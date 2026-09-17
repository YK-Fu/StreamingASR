from typing import Optional, List
import copy

import editdistance
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from transformers import StaticCache

from src.modules.transformer_decoder import DecoderRuntime

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, AbstractCTCDecoding
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.common.tokenizers.aggregate_tokenizer import DummyTokenizer
from nemo.collections.asr.metrics.wer import WER as NeMoWER
from nemo.utils import logging

class WER(NeMoWER):
    full_state_update: bool = True
    def __init__(
        self,
        decoding,
        use_cer=False,
        log_prediction=True,
        fold_consecutive=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
        sync_on_compute=True,
        **kwargs,
    ):
        Metric.__init__(self, dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute)
        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index

        self.decode = None
        if isinstance(self.decoding, RNNTDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, input_ids_lengths: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions,
                encoded_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
                input_ids=input_ids,
                input_ids_lengths=input_ids_lengths,
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, input_ids_lengths: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
            )
        else:
            raise TypeError(f"WER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        predictions_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Decode one batch and replace the local error-count state.

        ``input_ids_lengths`` carries the valid length of each right-padded
        transformer prompt. NeMo's base WER does not forward prompt lengths to
        its decoder, so the small override is required for heterogeneous
        prompts. As in NeMo, callers compute and reset after every batch.
        """
        references = []
        targets_cpu = targets.long().cpu()
        target_lengths_cpu = targets_lengths.long().cpu()
        if self.batch_dim_index != 0:
            dims = list(range(targets_cpu.ndim))
            targets_cpu = targets_cpu.permute(
                *([self.batch_dim_index] + dims[: self.batch_dim_index] + dims[self.batch_dim_index + 1 :])
            )

        with torch.no_grad():
            for sample_idx in range(targets_cpu.shape[0]):
                target_length = int(target_lengths_cpu[sample_idx].item())
                target = targets_cpu[sample_idx, :target_length].tolist()
                references.append(self.decoding.decode_ids_to_str(target))
            hypotheses = (
                self.decode(
                    predictions,
                    predictions_lengths,
                    predictions_mask,
                    input_ids,
                    input_ids_lengths,
                )
                if predictions.numel() > 0
                else []
            )

        if hypotheses and self.log_prediction:
            logging.info("\n")
            logging.info(f"WER reference:{references[0]}")
            logging.info(f"WER predicted:{hypotheses[0].text}")

        words = 0
        scores = 0
        for hypothesis, reference in zip(hypotheses, references):
            if isinstance(hypothesis, list):
                hypothesis = hypothesis[0]
            if self.use_cer:
                hypothesis_units = list(hypothesis.text)
                reference_units = list(reference)
            else:
                hypothesis_units = hypothesis.text.split()
                reference_units = reference.split()
            words += len(reference_units)
            scores += editdistance.eval(hypothesis_units, reference_units)

        self.scores = torch.tensor(
            scores, device=self.scores.device, dtype=self.scores.dtype
        )
        self.words = torch.tensor(
            words, device=self.words.device, dtype=self.words.dtype
        )

class CTCDecoding(CTCBPEDecoding):
    def __init__(self, decoding_cfg, tokenizer, blank_id=0):
        self.tokenizer = tokenizer

        AbstractCTCDecoding.__init__(self, decoding_cfg, blank_id=blank_id)

        # Finalize Beam Search Decoding framework
        if hasattr(self.decoding, "set_decoding_type"):
            if hasattr(self.tokenizer.tokenizer, 'get_vocab'):
                vocab_dict = self.tokenizer.tokenizer.get_vocab()
                if isinstance(self.tokenizer.tokenizer, DummyTokenizer):  # AggregateTokenizer.DummyTokenizer
                    vocab = vocab_dict
                else:
                    vocab = list(vocab_dict.keys())
                self.decoding.set_vocabulary(vocab)
                self.decoding.set_tokenizer(tokenizer)
            else:
                logging.warning("Could not resolve the vocabulary of the tokenizer !")

            self.decoding.set_decoding_type('subword')

class RNNTInfer:
    def __init__(
        self,
        prediction_network,
        joint_network,
        bos_idx: int = 0,
        blank_idx: int = 0,
        max_length: int = 1024,  # Total decoder-cache budget: prompt + emissions
        prefill_bucket_size: int = 0,
        max_symbols_per_step: int = 10,  # Max symbols per encoder frame (safety)
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
        compute_timestamps: bool = False,
    ):
        # TODO: Add timestamp/alignments/confidence computation implementation
        self.decoder_runtime = DecoderRuntime.eager(prediction_network)
        self.joint_network = joint_network
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx
        self.max_length = max_length
        self.prefill_bucket_size = prefill_bucket_size
        self.max_symbols_per_step = max_symbols_per_step
        self.compute_timestamps = compute_timestamps

    def set_decoder_runtime(self, runtime: DecoderRuntime):
        if runtime.base is not self.decoder_runtime.base:
            raise ValueError("DecoderRuntime.base must be the canonical prediction network")
        self.decoder_runtime = runtime

    def prefill_decoder_state(self, input_ids=None, attn_mask=None, position_ids=None, batch_size=None):
        base_decoder = self.decoder_runtime.base
        device = next(base_decoder.parameters()).device
        dtype = next(base_decoder.parameters()).dtype
        if input_ids is None:
            # No transcribed history: prefill only BOS. The StaticCache reserves
            # max_length slots, but the model should not execute 1024 padded
            # prompt positions just to initialize that cache.
            assert batch_size is not None, "batch_size must be provided if input_ids is None"
            input_ids = torch.full((batch_size, 1), self.bos_idx, device=device, dtype=torch.long)
            attn_mask = torch.ones((batch_size, 1), device=device, dtype=torch.int32)
            position_ids = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
        else:
            assert batch_size is None or batch_size == input_ids.size(0), "batch_size and input_ids.size(0) must be the same or batch_size is None"
            input_ids = input_ids.to(device)
            assert input_ids.size(1) <= self.max_length, "prompt length must not exceed max_length"
            if attn_mask is None:
                attn_mask = torch.ones_like(input_ids, dtype=torch.int32)
            else:
                attn_mask = attn_mask.to(device=device, dtype=torch.int32)
            if position_ids is None:
                # Logical positions remain contiguous within each right-padded
                # prompt even though all samples share physical cache slots.
                position_ids = attn_mask.long().cumsum(dim=1).sub(1).clamp(min=0)
            else:
                position_ids = position_ids.to(device=device, dtype=torch.long)
            assert attn_mask.shape == input_ids.shape
            assert position_ids.shape == input_ids.shape

        batch_size = input_ids.size(0)
        valid_lengths = attn_mask.sum(dim=-1).long()
        if torch.any(valid_lengths < 1):
            raise ValueError("Every decoder prompt must contain at least one token")
        expected_mask = (
            torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            < valid_lengths.unsqueeze(1)
        )
        if not torch.equal(attn_mask.bool(), expected_mask):
            raise ValueError("Decoder prompts must use contiguous right padding")
        prompt_length = input_ids.size(1)
        prefill_network = base_decoder
        if (
            self.decoder_runtime.decode_prefill is not None
            and self.prefill_bucket_size > 0
        ):
            bucket_length = min(
                self.max_length,
                ((prompt_length + self.prefill_bucket_size - 1) // self.prefill_bucket_size)
                * self.prefill_bucket_size,
            )
            if bucket_length < prompt_length:
                raise ValueError("Prompt is too long for the static decoder cache")
            pad_length = bucket_length - prompt_length
            if pad_length:
                input_ids = F.pad(input_ids, (0, pad_length), value=self.bos_idx)
                attn_mask = F.pad(attn_mask, (0, pad_length), value=0)
                # Padded positions are masked, so their logical position is not
                # observed. Actual emitted tokens subsequently overwrite these
                # cache slots using their contiguous logical position.
                position_ids = F.pad(position_ids, (0, pad_length), value=0)
            prefill_network = self.decoder_runtime.prefill_callable
        execution_length = input_ids.size(1)
        cache_position = torch.arange(execution_length, device=device, dtype=torch.int64)
        cache = StaticCache(
            config=base_decoder.config,
            max_batch_size=batch_size,
            max_cache_len=self.max_length,
            device=device,
            dtype=dtype
        )
        outputs, _ = prefill_network(
            input_ids=input_ids,
            attn_mask=attn_mask,
            position_ids=position_ids,
            cache=cache,
            cache_position=cache_position
        )
        # The state that predicts the first generated token is the LAST CONSUMED
        # token's hidden state, i.e. index valid_lengths - 1 (not valid_lengths,
        # which is the first padding slot).
        next_token_logits = outputs.transpose(1, 2)[torch.arange(batch_size), valid_lengths - 1].unsqueeze(1)
        # Grow this mask one column only when a non-blank token is
        # emitted. Its width then tracks the number of populated cache slots.
        if self.decoder_runtime.decode_step is not None:
            if attn_mask.size(1) > self.max_length:
                raise ValueError("Prompt attention mask exceeds the static decoder cache")
            next_attn_mask = F.pad(attn_mask, (0, self.max_length - attn_mask.size(1)), value=0)
        else:
            next_attn_mask = attn_mask
        # ``position_ids`` are per-sample logical positions. ``cache_position``
        # is a shared physical cursor, so shorter prompts leave masked cache
        # holes before generated tokens. This is the same mechanism used when
        # only part of an RNN-T batch emits on a decode iteration.
        next_position_ids = position_ids[torch.arange(batch_size), valid_lengths - 1].unsqueeze(-1)
        next_cache_position = valid_lengths.max().unsqueeze(0) - 1

        # Greedy decoding overwrites this one-token placeholder in-place. Clone it:
        # prompt input_ids can be a view into the training context, which compiled
        # embedding backward still needs unchanged at periodic training-WER steps.
        next_input_ids = input_ids[:, :1].clone()
        return next_input_ids, next_token_logits, next_attn_mask, next_position_ids, cache, next_cache_position


    def forward_decoder_one_step(self, input_ids, attn_mask, position_ids, cache, cache_position, decoder_logits=None):
        assert input_ids.size(1) == 1, "input_ids should have shape (batch_size, 1)"
        assert input_ids.size(1) == position_ids.size(1), "input_ids and position_ids should have the same length"
        if decoder_logits is None:
            prediction_network = self.decoder_runtime.step_callable
            decoder_logits, _ = prediction_network(
                input_ids=input_ids,
                attn_mask=attn_mask,
                position_ids=position_ids,
                cache=cache,
                cache_position=cache_position
            )
        return decoder_logits.transpose(1, 2)
    def kill_zombie_cache(self, cache, cache_position=None, attn_mask=None):
        # TODO: Implement zombie cache killing
        return 

    def decode(
        self,
        encoder_output,
        input_ids=None,
        attn_mask=None,
        position_ids=None,
        cache=None,
        cache_position=None,
        encoded_lengths=None,
    ):
        raise NotImplementedError("Subclass of RNNTInfer must implement the decode method")

class LoopLabelRNNTInfer(RNNTInfer):
    # Higher throughput decoding strategy, but if one sample contains long silence, it will incur longer latency
    def decode(
        self,
        encoder_output,
        input_ids=None,
        attn_mask=None,
        position_ids=None,
        cache=None,
        cache_position=None,
        encoded_lengths=None,
    ):
        encoder_output = encoder_output.transpose(1, 2)
        batch_size, max_time, _ = encoder_output.size()
        if encoded_lengths is None:
            encoded_lengths = torch.full(
                (batch_size,), max_time, dtype=torch.long, device=encoder_output.device
            )
        else:
            encoded_lengths = encoded_lengths.to(
                device=encoder_output.device, dtype=torch.long
            )
            if encoded_lengths.shape != (batch_size,):
                raise ValueError("encoded_lengths must have shape [batch]")
            if torch.any(encoded_lengths < 1) or torch.any(encoded_lengths > max_time):
                raise ValueError("encoded_lengths must be in [1, encoder_time]")
        if cache is None:
            input_ids, dec_out, attn_mask, position_ids, cache, cache_position = self.prefill_decoder_state(input_ids, attn_mask, position_ids, batch_size)
        else:
            assert input_ids.size(1) == 1, "input_ids should have shape (batch_size, 1)"
            dec_out = None
        b2active = encoded_lengths > 0
        b2time = torch.zeros(batch_size, device=encoder_output.device, dtype=torch.int64)
        safe_time = torch.zeros(batch_size, device=encoder_output.device, dtype=torch.int64)
        hyps = [
            rnnt_utils.Hypothesis(
                score=0.0,
                y_sequence=[],
                dec_state=None,
                last_token=None,
                length=0
            ) 
            for _ in range(batch_size)
        ]
        symbols_added = torch.zeros(batch_size, dtype=torch.int32, device=encoder_output.device)
        while b2active.any() and cache_position < self.max_length:
            find_next_token_or_end = torch.zeros(batch_size, dtype=torch.bool)
            while not find_next_token_or_end.all():
                if dec_out is None:
                    dec_out = self.forward_decoder_one_step(input_ids, attn_mask, position_ids, cache, cache_position)
                token_probs = self.joint_network(encoder_output[torch.arange(batch_size), safe_time, ].unsqueeze(1), dec_out)
                predictions = torch.argmax(token_probs, dim=-1).reshape(batch_size)
                # A token is written at cache_position + 1. Do not overrun the
                # static cache, and enforce the per-frame emission cap.
                cache_full = bool((cache_position + 1 >= self.max_length).item())
                force_blank = (symbols_added >= self.max_symbols_per_step) | ~b2active
                if cache_full:
                    force_blank.fill_(True)
                predictions[force_blank] = self.blank_idx
                blank_mask = predictions == self.blank_idx
                b2time[blank_mask & b2active] += 1

                # Reset symbols_added if blank token is added
                symbols_added[blank_mask] = 0
                b2active = b2time < encoded_lengths
                safe_time = torch.minimum(b2time, encoded_lengths - 1)
                find_next_token_or_end = ~b2active | ~blank_mask

            if not blank_mask.all():
                # The emitted token is consumed by the decoder on the next
                # forward at cache_position + 1.
                if self.decoder_runtime.decode_step is None:
                    attn_mask = F.pad(attn_mask, (0, 1), value=0)
                for b in range(batch_size):
                    if predictions[b] != self.blank_idx:
                        input_ids[b, 0] = predictions[b]
                        # Mark the slot this token will occupy on the NEXT forward
                        # (cache_position+1), not the last-consumed slot, so the
                        # token attends to itself (matching training's causal mask).
                        attn_mask[b, cache_position + 1] = 1
                        position_ids[b, 0] = position_ids[b, 0] + 1
                        symbols_added[b] += 1
                        hyps[b].y_sequence.append(predictions[b].item())
                    else:
                        attn_mask[b, cache_position + 1] = 0
                dec_out = None
                cache_position = cache_position + 1
        return hyps, input_ids, attn_mask, position_ids, cache, cache_position

class LoopFrameRNNTInfer(RNNTInfer):
    # Fixed latency decoding strategy with lower throughput and might generate more zombie cache
    def decode(
        self,
        encoder_output,
        input_ids=None,
        attn_mask=None,
        position_ids=None,
        cache=None,
        cache_position=None,
        encoded_lengths=None,
    ):
        encoder_output = encoder_output.transpose(1, 2)
        batch_size, max_time, _ = encoder_output.shape
        if encoded_lengths is None:
            encoded_lengths = torch.full(
                (batch_size,), max_time, dtype=torch.long, device=encoder_output.device
            )
        else:
            encoded_lengths = encoded_lengths.to(
                device=encoder_output.device, dtype=torch.long
            )
            if encoded_lengths.shape != (batch_size,):
                raise ValueError("encoded_lengths must have shape [batch]")
            if torch.any(encoded_lengths < 1) or torch.any(encoded_lengths > max_time):
                raise ValueError("encoded_lengths must be in [1, encoder_time]")
        if cache is None:
            input_ids, dec_out, attn_mask, position_ids, cache, cache_position = self.prefill_decoder_state(input_ids, attn_mask, position_ids, batch_size)
        else:
            assert input_ids.size(1) == 1, "input_ids should have shape (batch_size, 1)"
            dec_out = None
        hyps = [
            rnnt_utils.Hypothesis(
                score=0.0,
                y_sequence=[],
                dec_state=None,
                last_token=None,
                length=0
            ) 
            for _ in range(batch_size)
        ]
        symbols_added = torch.zeros(batch_size, dtype=torch.int32, device=encoder_output.device)
        for t in range(max_time):
            frame_active = t < encoded_lengths
            while frame_active.any():
                if dec_out is None:
                    dec_out = self.forward_decoder_one_step(input_ids, attn_mask, position_ids, cache, cache_position)
                token_probs = self.joint_network(encoder_output[torch.arange(batch_size), t:t+1, :], dec_out)
                predictions = torch.argmax(token_probs, dim=-1).reshape(batch_size)
                cache_full = bool((cache_position + 1 >= self.max_length).item())
                force_blank = (symbols_added >= self.max_symbols_per_step) | ~frame_active
                if cache_full:
                    force_blank.fill_(True)
                predictions[force_blank] = self.blank_idx
                blank_mask = predictions == self.blank_idx
                symbols_added[blank_mask & frame_active] = 0
                emit_mask = frame_active & ~blank_mask
                frame_active = emit_mask
                if not frame_active.any():
                    break
                else:
                    if self.decoder_runtime.decode_step is None:
                        attn_mask = F.pad(attn_mask, (0, 1), value=0)
                    for b in range(batch_size):
                        if emit_mask[b]:
                            input_ids[b, 0] = predictions[b]
                            attn_mask[b, cache_position + 1] = 1
                            position_ids[b, 0] = position_ids[b, 0] + 1
                            symbols_added[b] += 1
                            hyps[b].y_sequence.append(predictions[b].item())
                dec_out = None
                cache_position = cache_position + 1
        return hyps, input_ids, attn_mask, position_ids, cache, cache_position


class RNNTDecoding(ConfidenceMethodMixin):
    """
    RNN-T Decoding for BPE/Subword tokenizers.
    
    This class inherits from AbstractRNNTDecoding and provides decoding functionality
    for RNN-T models with BPE or subword tokenizers.
    """
    def __init__(
        self,
        decoding_cfg,
        decoder=None,
        joint=None,
        tokenizer=None,
        blank_id=0,
        decoder_runtime: Optional[DecoderRuntime] = None,
    ):
        """
        Args:
            decoding_cfg: DictConfig with decoding configuration
            decoder: Canonical decoder module (legacy eager construction path)
            decoder_runtime: Explicit decoder execution variants
            joint: The Joint network module  
            tokenizer: The tokenizer which will be used for decoding
            supported_punctuation: Optional set of punctuation marks in the vocabulary
        """
        super().__init__()
        if decoder_runtime is None:
            if decoder is None:
                raise ValueError("decoder or decoder_runtime must be provided")
            decoder_runtime = DecoderRuntime.eager(decoder)
        elif decoder is not None and decoder is not decoder_runtime.base:
            raise ValueError("decoder must match decoder_runtime.base")
        self.decoder_runtime = decoder_runtime
        decoder = decoder_runtime.base
        self.cfg = decoding_cfg
        self.blank_id = blank_id
        self.tokenizer = tokenizer
        self.compute_timestamps = self.cfg.get('compute_timestamps', None)
        self.preserve_alignments = self.cfg.get('preserve_alignments', None)
        self.preserve_frame_confidence = self.cfg.get('preserve_frame_confidence', None)
        self.max_length = int(self.cfg.get('max_length', 1024))
        self.prefill_bucket_size = int(self.cfg.get('prefill_bucket_size', 0))
        self.max_symbols_per_step = int(self.cfg.get('max_symbols_per_step', 10))
        if self.max_length < 3:
            raise ValueError("decoding.max_length must reserve BOS, language, and at least one output slot")
        if self.prefill_bucket_size < 0:
            raise ValueError("decoding.prefill_bucket_size cannot be negative")
        # ``max_length`` is the single total-token budget. Limit prompt history
        # to the first half so at least the other half remains for emissions.
        self.max_prompt_length = (self.max_length + 1) // 2
        
        # Override decoding strategy instantiation for greedy_batch
        if self.cfg.strategy == "LoopLabel":
            self.decoding = LoopLabelRNNTInfer(
                prediction_network=decoder,
                joint_network=joint,
                bos_idx=self.blank_id,
                blank_idx=self.blank_id,
                max_length=self.max_length,
                prefill_bucket_size=self.prefill_bucket_size,
                max_symbols_per_step=self.max_symbols_per_step,
                preserve_alignments=self.preserve_alignments,
                preserve_frame_confidence=self.preserve_frame_confidence,
                compute_timestamps=self.compute_timestamps,
            )
        elif self.cfg.strategy == "LoopFrame":
            self.decoding = LoopFrameRNNTInfer(
                prediction_network=decoder,
                joint_network=joint,
                bos_idx=self.blank_id,
                blank_idx=self.blank_id,
                max_length=self.max_length,
                prefill_bucket_size=self.prefill_bucket_size,
                max_symbols_per_step=self.max_symbols_per_step,
                preserve_alignments=self.preserve_alignments,
                preserve_frame_confidence=self.preserve_frame_confidence,
                compute_timestamps=self.compute_timestamps,
            )
        else:
            raise ValueError(f"Invalid strategy: {self.cfg.strategy}")
        self.decoding.set_decoder_runtime(decoder_runtime)

    def set_decoder_runtime(self, runtime: DecoderRuntime) -> None:
        self.decoding.set_decoder_runtime(runtime)
        self.decoder_runtime = runtime

    def prepare_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Bound a single utterance prompt without dropping BOS or language.

        Dataset contexts are already left-truncated for training. Validation can
        still receive a longer prompt than the fixed StaticCache permits, so keep
        the most recent history while retaining the two required special tokens.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] < 1:
            raise ValueError("RNN-T prompts must have shape [batch, prompt_length]")
        if input_ids.shape[1] < 2:
            raise ValueError("RNN-T prompts must contain BOS and a language token")
        if input_ids.shape[1] <= self.max_prompt_length:
            return input_ids
        history_budget = self.max_prompt_length - 2
        if history_budget == 0:
            return input_ids[:, :2]
        return torch.cat((input_ids[:, :2], input_ids[:, -history_budget:]), dim=1)

    def prepare_batched_prompts(
        self,
        input_ids: torch.Tensor,
        input_ids_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate and right-pad sample-specific prompts for one decode call."""
        if input_ids.ndim != 2:
            raise ValueError("RNN-T prompts must have shape [batch, prompt_length]")
        batch_size = input_ids.shape[0]
        if input_ids_lengths.shape != (batch_size,):
            raise ValueError("input_ids_lengths must have shape [batch]")
        lengths = input_ids_lengths.to(device=input_ids.device, dtype=torch.long)
        if torch.any(lengths < 2) or torch.any(lengths > input_ids.shape[1]):
            raise ValueError(
                "Every RNN-T prompt must contain BOS and language tokens and fit input_ids"
            )

        prepared = []
        for sample_idx, prompt_length in enumerate(lengths.detach().cpu().tolist()):
            prepared.append(
                self.prepare_prompt(
                    input_ids[sample_idx : sample_idx + 1, :prompt_length]
                ).squeeze(0)
            )

        prepared_lengths = torch.tensor(
            [prompt.numel() for prompt in prepared],
            dtype=torch.long,
            device=input_ids.device,
        )
        max_prompt_length = int(prepared_lengths.max().item())
        padded = input_ids.new_full(
            (batch_size, max_prompt_length), self.blank_id
        )
        for sample_idx, prompt in enumerate(prepared):
            padded[sample_idx, : prompt.numel()] = prompt
        attention_mask = (
            torch.arange(max_prompt_length, device=input_ids.device).unsqueeze(0)
            < prepared_lengths.unsqueeze(1)
        ).to(dtype=torch.int32)
        return padded, attention_mask


    
    def _aggregate_token_confidence(self, hypothesis: rnnt_utils.Hypothesis) -> List[float]:
        """
        Aggregate token confidence to word-level confidence.
        
        Args:
            hypothesis: Hypothesis object with token confidence scores
            
        Returns:
            A list of word-level confidence scores.
        """
        return self._aggregate_token_confidence_chars(hypothesis.words, hypothesis.token_confidence)

    def decode_ids_to_str(self, tokens: List[int], divide_asia_token_by_space: bool = True) -> str:
        """
        Decode token IDs to string.
        
        Args:
            tokens: List of token IDs
            divide_asia_token_by_space: if True, divide the asia token by space. e.g., "你好嗎" -> "你 好 嗎"
            
        Returns:
            Decoded string
        """
        hypothesis = self.tokenizer.ids_to_text(tokens)
        if divide_asia_token_by_space:
            # TODO
            pass
        return hypothesis
    
    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Decode token IDs to token strings.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            List of decoded token strings
        """
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list

    def decode_hypothesis(self, hypotheses_list: List[rnnt_utils.Hypothesis]) -> List[rnnt_utils.Hypothesis]:
        """
        Decode hypotheses to text.
        
        Args:
            hypotheses_list: List of Hypothesis objects
            
        Returns:
            List of Hypothesis objects with decoded text
        """
        for ind in range(len(hypotheses_list)):
            # Extract the integer encoded hypothesis
            prediction = hypotheses_list[ind].y_sequence

            if type(prediction) != list:
                prediction = prediction.tolist()

            prediction = [p for p in prediction if p != self.blank_id]


            if self.compute_timestamps is True:
                # keep the original predictions, wrap with the number of repetitions per token and alignments
                # this is done so that `rnnt_decoder_predictions_tensor()` can process this hypothesis
                # in order to compute exact time stamps.
                alignments = copy.deepcopy(hypotheses_list[ind].alignments)
                token_repetitions = [1] * len(alignments)  # preserve number of repetitions per token
                hypothesis = (prediction, alignments, token_repetitions)
            else:
                hypothesis = self.decode_ids_to_str(prediction)

            # De-tokenize the integer tokens
            hypotheses_list[ind].text = hypothesis

        return hypotheses_list

    def compute_confidence(self, hypotheses_list: List[rnnt_utils.Hypothesis]) -> List[rnnt_utils.Hypothesis]:
        # TODO
        pass
    def compute_rnnt_timestamps(self, hypothesis: rnnt_utils.Hypothesis, timestamp_type: str = 'all'):
        # TODO
        pass
    def rnnt_decoder_predictions_tensor(
        self,
        encoder_output,
        encoded_lengths,
        return_hypotheses: bool = False,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
        input_ids=None,
        input_ids_lengths=None,
        **kwargs,
    ):
        del partial_hypotheses, kwargs
        with torch.inference_mode():
            # input_ids is the prompt prefix ([bos, <language>, ...]) that the
            # predictor/joint were trained on; seeding it is required, otherwise
            # the bos-only initial state is out-of-distribution for the joint.
            prompt_attention_mask = None
            if input_ids is not None and input_ids_lengths is not None:
                input_ids, prompt_attention_mask = self.prepare_batched_prompts(
                    input_ids=input_ids,
                    input_ids_lengths=input_ids_lengths,
                )
            elif input_ids is not None:
                input_ids = self.prepare_prompt(input_ids)
            hypotheses_list = self.decoding.decode(
                encoder_output=encoder_output,
                encoded_lengths=encoded_lengths,
                input_ids=input_ids,
                attn_mask=prompt_attention_mask,
            )  # type: [List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]
        prediction_list = hypotheses_list

        hypotheses = self.decode_hypothesis(prediction_list)  # type: List[str]

        # If computing timestamps
        if self.compute_timestamps is True:
            timestamp_type = self.cfg.get('rnnt_timestamp_type', 'all')
            for hyp_idx in range(len(hypotheses)):
                hypotheses[hyp_idx] = self.compute_rnnt_timestamps(hypotheses[hyp_idx], timestamp_type)

        if return_hypotheses:
            # greedy decoding, can get high-level confidence scores
            if self.preserve_frame_confidence and (
                self.preserve_word_confidence or self.preserve_token_confidence
            ):
                hypotheses = self.compute_confidence(hypotheses)
            return hypotheses

        return [rnnt_utils.Hypothesis(h.score, h.y_sequence, h.text) for h in hypotheses]
