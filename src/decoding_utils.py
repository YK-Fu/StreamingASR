from typing import Optional, List
import copy

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from transformers import StaticCache

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, AbstractCTCDecoding
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.common.tokenizers.aggregate_tokenizer import DummyTokenizer
from nemo.collections.asr.metrics.wer import WER as NeMoWER

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
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions,
                encoded_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
            )
        else:
            raise TypeError(f"WER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

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
        max_length: int = 1024,
        max_symbols_per_step: int = 10,  # Max symbols per encoder frame (safety)
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
        compute_timestamps: bool = False,
    ):
        # TODO: Add timestamp/alignments/confidence computation implementation
        self.prediction_network = prediction_network
        self.joint_network = joint_network
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx
        self.max_length = max_length
        self.max_symbols_per_step = max_symbols_per_step
        self.compute_timestamps = compute_timestamps

    def prefill_decoder_state(self, input_ids=None, attn_mask=None, position_ids=None, batch_size=None):
        device = next(self.prediction_network.parameters()).device
        dtype = next(self.prediction_network.parameters()).dtype
        if input_ids is None:
            # No transcribed history, begin with BOS
            assert batch_size is not None, "batch_size must be provided if input_ids is None"
            input_ids = torch.zeros(batch_size, self.max_length, device=device, dtype=torch.long).fill_(self.bos_idx)
            attn_mask = torch.zeros(batch_size, self.max_length, device=device, dtype=torch.int32)
            attn_mask[:, 0] = 1
            position_ids = torch.zeros(batch_size, self.max_length, device=device, dtype=torch.long)
        else:
            assert batch_size is None or batch_size == input_ids.size(0), "batch_size and input_ids.size(0) must be the same or batch_size is None"
            assert attn_mask is not None and position_ids is not None, "attn_mask and position_ids must be provided if input_ids is provided"
            input_ids = input_ids.to(device)
            if input_ids.size(1) != self.max_length:
                assert input_ids.size(1) < self.max_length, "input_ids.size(1) must be less than max_length - 1"
                input_ids = F.pad(input_ids, (0, self.max_length - input_ids.size(1)), value=self.bos_idx)
                attn_mask = F.pad(attn_mask, (0, self.max_length - attn_mask.size(1)), value=0)
                position_ids = F.pad(position_ids, (0, self.max_length - position_ids.size(1)), value=0)
            else:
                assert attn_mask.size(1) == position_ids.size(1) == self.max_length, "attn_mask.size(1) must be equal to max_length - 1 if input_ids.size(1) is equal to max_length - 1"
        valid_lengths = attn_mask.sum(dim=-1)
        cache_position = torch.arange(self.max_length, device=device, dtype=torch.int64)
        cache = StaticCache(
            config=self.prediction_network.config,
            max_batch_size=batch_size,
            max_cache_len=self.max_length,
            device=device,
            dtype=dtype
        )
        outputs, _ = self.prediction_network(
            input_ids=input_ids,
            attn_mask=attn_mask,
            position_ids=position_ids,
            cache=cache,
            cache_position=cache_position
        )
        next_token_logits = outputs.transpose(1, 2)[torch.arange(batch_size), valid_lengths].unsqueeze(1)
        next_attn_mask = F.pad(attn_mask, (0, 1), value=1)      # (b, max_length) -> (b, max_length + 1)
        next_position_ids = position_ids[torch.arange(batch_size), valid_lengths].unsqueeze(-1) + 1
        next_cache_position = valid_lengths.max().unsqueeze(0) - 1

        # The content of input_ids will not be used again, we return it as the placeholder for the one step decode stage
        return input_ids[:, :1], next_token_logits, next_attn_mask, next_position_ids, cache, next_cache_position


    def forward_decoder_one_step(self, input_ids, attn_mask, position_ids, cache, cache_position, decoder_logits=None):
        assert input_ids.size(1) == 1, "input_ids should have shape (batch_size, 1)"
        assert input_ids.size(1) == position_ids.size(1), "input_ids and position_ids should have the same length"
        if decoder_logits is None:
            decoder_logits, _ = self.prediction_network(
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

    def decode(self, encoder_output, input_ids=None, attn_mask=None, position_ids=None, cache=None, cache_position=None):
        raise NotImplementedError("Subclass of RNNTInfer must implement the decode method")

class LoopLabelRNNTInfer(RNNTInfer):
    # Higher throughput decoding strategy, but if one sample contains long silence, it will incur longer latency
    def decode(self, encoder_output, input_ids=None, attn_mask=None, position_ids=None, cache=None, cache_position=None):
        encoder_output = encoder_output.transpose(1, 2)
        batch_size, max_time, _ = encoder_output.size()
        if cache is None:
            input_ids, dec_out, attn_mask, position_ids, cache, cache_position = self.prefill_decoder_state(input_ids, attn_mask, position_ids, batch_size)
        else:
            assert input_ids.size(1) == 1, "input_ids should have shape (batch_size, 1)"
            dec_out = None
        b2active = torch.ones(batch_size, device=encoder_output.device, dtype=torch.bool)
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
                predictions = torch.argmax(token_probs, dim=-1).squeeze(1)
                # Force to add blank token if symbols_added >= self.max_symbols_per_step
                predictions[symbols_added >= self.max_symbols_per_step] = self.blank_idx
                blank_mask = (predictions == self.blank_idx).squeeze(1)
                b2time[blank_mask] += 1

                # Reset symbols_added if blank token is added
                symbols_added[blank_mask] = 0
                b2active = b2time < max_time
                safe_time = b2time.clamp(max=max_time - 1)
                find_next_token_or_end = ~b2active | ~blank_mask

            if not blank_mask.all():
                for b in range(batch_size):
                    if predictions[b] != self.blank_idx:
                        input_ids[b, 0] = predictions[b]
                        attn_mask[b, cache_position] = 1
                        position_ids[b, 0] = position_ids[b, 0] + 1
                        symbols_added[b] += 1
                        hyps[b].y_sequence.append(predictions[b].item())
                    else:
                        attn_mask[b, cache_position] = 0
                dec_out = None
                cache_position = cache_position + 1
        return hyps, input_ids, attn_mask, position_ids, cache, cache_position

class LoopFrameRNNTInfer(RNNTInfer):
    # Fixed latency decoding strategy with lower throughput and might generate more zombie cache
    def decode(self, encoder_output, input_ids=None, attn_mask=None, position_ids=None, cache=None, cache_position=None):
        batch_size, max_time, _ = encoder_output.shape
        device = encoder_output.device
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
            while True:
                if dec_out is None:
                    dec_out = self.forward_decoder_one_step(input_ids, attn_mask, position_ids, cache, cache_position)
                token_probs = self.joint_network(encoder_output[torch.arange(batch_size), t:t+1, :], dec_out)
                predictions = torch.argmax(token_probs, dim=-1).squeeze(1)
                predictions[symbols_added >= self.max_symbols_per_step] = self.blank_idx
                blank_mask = predictions == self.blank_idx
                symbols_added[blank_mask] = 0
                if blank_mask.all():
                    break
                else:
                    for b in range(batch_size):
                        if predictions[b] != self.blank_idx:
                            input_ids[b, 0] = predictions[b]
                            attn_mask[b, cache_position] = 1
                            position_ids[b, 0] = position_ids[b, 0] + 1
                            symbols_added[b] += 1
                            hyps[b].y_sequence.append(predictions[b].item())
                        else:
                            attn_mask[b, cache_position] = 0
                dec_out = None
                cache_position = cache_position + 1
        return hyps, input_ids, attn_mask, position_ids, cache, cache_position


class RNNTDecoding(ConfidenceMethodMixin):
    """
    RNN-T Decoding for BPE/Subword tokenizers.
    
    This class inherits from AbstractRNNTDecoding and provides decoding functionality
    for RNN-T models with BPE or subword tokenizers.
    """
    def __init__(self, decoding_cfg, decoder, joint, tokenizer, blank_id=0):
        """
        Args:
            decoding_cfg: DictConfig with decoding configuration
            decoder: The Decoder/Prediction network module
            joint: The Joint network module  
            tokenizer: The tokenizer which will be used for decoding
            supported_punctuation: Optional set of punctuation marks in the vocabulary
        """
        super().__init__()
        self.cfg = decoding_cfg
        self.blank_id = blank_id
        self.tokenizer = tokenizer
        self.compute_timestamps = self.cfg.get('compute_timestamps', None)
        self.preserve_alignments = self.cfg.get('preserve_alignments', None)
        self.preserve_frame_confidence = self.cfg.get('preserve_frame_confidence', None)
        self.max_length = self.cfg.get('max_length', 1024)
        self.max_symbols_per_step = self.cfg.get('max_symbols_per_step', 10)
        
        # Override decoding strategy instantiation for greedy_batch
        if self.cfg.strategy == "LoopLabel":
            self.decoding = LoopLabelRNNTInfer(
                prediction_network=decoder,
                joint_network=joint,
                blank_idx=self.blank_id,
                max_length=self.max_length,
                max_symbols_per_step=self.max_symbols_per_step,
                preserve_alignments=self.preserve_alignments,
                preserve_frame_confidence=self.preserve_frame_confidence,
                compute_timestamps=self.compute_timestamps,
            )
        elif self.cfg.strategy == "LoopFrame":
            self.decoding = LoopFrameRNNTInfer(
                prediction_network=decoder,
                joint_network=joint,
                blank_idx=self.blank_id,
                max_length=self.max_length,
                max_symbols_per_step=self.max_symbols_per_step,
                preserve_alignments=self.preserve_alignments,
                preserve_frame_confidence=self.preserve_frame_confidence,
                compute_timestamps=self.compute_timestamps,
            )
        else:
            raise ValueError(f"Invalid strategy: {self.cfg.strategy}")


    
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
    def rnnt_decoder_predictions_tensor(self, encoder_output, encoded_lengths, return_hypotheses: bool = False, partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None, **kwargs):
        with torch.inference_mode():
            hypotheses_list = self.decoding.decode(encoder_output=encoder_output)  # type: [List[Hypothesis]]

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
