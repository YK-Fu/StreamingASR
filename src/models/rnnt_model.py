# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Modifications Copyright (c) 2026, Iven Fu. All rights reserved.

from typing import Dict, Optional

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, InterCTCMixin
from nemo.utils import logging
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.core.classes.mixins import AccessMixin

from src.loss import CTCLoss, NLLLoss
from src.decoding_utils import CTCDecoding, RNNTDecoding, WER
from src.datasets import get_asr_dataset, ResumableDataloader, ResumableSampler

class HybridRNNTCTCWhisperLMModel(EncDecHybridRNNTCTCModel, ASRBPEMixin, InterCTCMixin):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""
    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        tokenizer = AutoTokenizer(tokenizer_cfg.path, pad_token=tokenizer_cfg.blank_token, bos_token=tokenizer_cfg.blank_token, trust_remote_code=True)
        self.tokenizer = tokenizer

    @classmethod
    def from_config_dict(cls, cfg: DictConfig):
        # recursively load _module_ with the target module
        import hydra
        return hydra.utils.instantiate(cfg)

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        self.rank = 0
        if trainer is not None:
            self.world_size = trainer.world_size
            self.rank = trainer.global_rank
        self._setup_tokenizer(cfg.tokenizer)

        ASRModel.__init__(self, cfg=cfg, trainer=trainer)

        num_vocab = self.tokenizer.vocab_size
        self.blank_id = self.tokenizer.token_to_id(self.cfg.tokenizer.blank_token)
        with open_dict(self.cfg.decoder.config):
            self.cfg.decoder.config.vocab_size = num_vocab
            self.cfg.decoder.config.bos_token_id = self.tokenizer.tokenizer.bos_token_id
            self.cfg.decoder.config.eos_token_id = self.tokenizer.tokenizer.eos_token_id
            self.cfg.decoder.projection.num_classes = num_vocab


        with open_dict(self.cfg.joint):
            self.cfg.joint.num_classes = num_vocab - 1    #  minus the blank token
            self.cfg.joint.jointnet.encoder_hidden = cfg.encoder.d_model
            self.cfg.joint.jointnet.pred_hidden = cfg.decoder.config.hidden_size

        self.preprocessor = HybridRNNTCTCWhisperLMModel.from_config_dict(self.cfg.preprocessor)
        self.encoder = HybridRNNTCTCWhisperLMModel.from_config_dict(self.cfg.encoder)
        self.decoder = HybridRNNTCTCWhisperLMModel.from_config_dict(self.cfg.decoder)

        if hasattr(self.cfg, 'aux_llm') and self.cfg.aux_llm.llm_loss_weight > 0:
            self.llm_loss = NLLLoss(
                ignore_index=self.tokenizer.tokenizer.pad_token_id,
                reduction=self.cfg.aux_llm.get('llm_loss_reduction', 'mean'),
            )
            self.llm_loss_weight = self.cfg.aux_llm.llm_loss_weight
        else:
            self.llm_loss = None
            self.llm_loss_weight = 0
        # We use LLM prediction head and ctc_decoder_head as simple_lm_proj and simple_am_proj to save parameters
        self.llm_head = HybridRNNTCTCWhisperLMModel.from_config_dict(self.cfg.decoder.projection)
        if self.llm_head.tie_weights:
            self.llm_head_weights = self.decoder.prediction.embed_tokens.weight
        self.joint = HybridRNNTCTCWhisperLMModel.from_config_dict(self.cfg.joint)

        if hasattr(self.cfg, 'spec_augment') and self.cfg.spec_augment is not None:
            self.spec_augmentation = HybridRNNTCTCWhisperLMModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        if 'compute_eval_loss' in self.cfg:
            self.compute_eval_loss = self.cfg.compute_eval_loss
        else:
            self.compute_eval_loss = True

        # Setup decoding object
        self.decoding = RNNTDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
            blank_id=self.blank_id,
        )

        # Setup wer object
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self.cfg.get('use_cer', False),
            log_prediction=self.cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        self.joint.set_wer(self.wer)

        # setup auxiliary CTC decoder
        if 'aux_ctc' not in self.cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )

        if self.cfg.aux_ctc.decoder["num_classes"] < 1:
            logging.info(
                "\nReplacing placholder number of classes ({}) with actual number of classes - {}".format(
                    self.cfg.aux_ctc.decoder["num_classes"], num_vocab
                )
            )
            self.cfg.aux_ctc.decoder["num_classes"] = num_vocab

        # Setup CTC decoding
        self.ctc_decoder = HybridRNNTCTCWhisperLMModel.from_config_dict(self.cfg.aux_ctc.decoder)     # This is also used as simple_am_proj
        self.ctc_loss_weight = self.cfg.aux_ctc.get("ctc_loss_weight", 0.5)
        self.ctc_loss = CTCLoss(
                    num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                    zero_infinity=True,
                    reduction=self.cfg.aux_ctc.get("ctc_loss_reduction", "mean"),
                    blank_id=self.blank_id,
                )

        self.ctc_decoding = CTCDecoding(self.cfg.aux_ctc.decoding, tokenizer=self.tokenizer, blank_id=self.blank_id)

        # Setup CTC WER
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )
        # Setup optimization normalization (if provided in config)
        self.setup_optim_normalization()

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # setting the RNNT decoder as the default one
        self.cur_decoder = "rnnt"

        self.setup_interctc(decoder_name='ctc_decoder', loss_name='ctc_loss', wer_name='ctc_wer')

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        dataset = get_asr_dataset(
            manifest_filepath=config.manifest_filepath,
            tokenizer=self.tokenizer,
            batch_size=config['batch_size'],
            language_mapping=self.encoder.language_mapping,
            language_drop_rate=config.get('language_drop_rate', 0.0),
            sample_rate=config['sample_rate'],
            max_duration=config['max_duration'],
            min_duration=config['min_duration'],
            bucket_by=config.get('bucket_by', 'audio'),
            drop_last=config.get('drop_last', True),
        )
        if dataset is None:
            return None
        sampler = ResumableSampler(
            dataset=dataset, 
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=config.get('shuffle', True)
        )
        return ResumableDataloader(
            dataset=dataset,
            sampler=sampler,
            batch_size=None,
            collate_fn=dataset.collate_fn,
            drop_last=None,
            shuffle=None,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def change_decoding_strategy(
        self, decoding_cfg: DictConfig = None, decoder_type: str = None, verbose: bool = True
    ):
        """
        Changes decoding strategy used during RNNT decoding process.
        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having both RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
            verbose: bool whether to display change of decoder config or not.
        """
        if decoder_type is None or decoder_type == 'rnnt':
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.decoding

            self.decoding = RNNTDecoding(
                decoding_cfg=decoding_cfg,
                decoder=self.decoder,
                joint=self.joint,
                tokenizer=self.tokenizer,
                blank_id=self.blank_id,
            )

            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=self.wer.batch_dim_index,
                use_cer=self.wer.use_cer,
                log_prediction=self.wer.log_prediction,
                dist_sync_on_step=True,
            )

            self.joint.set_wer(self.wer)

            self.joint.temperature = decoding_cfg.get('temperature', 1.0)

            # Update config
            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            self.cur_decoder = "rnnt"
            if verbose:
                logging.info(
                    f"Changed decoding strategy of the RNNT decoder to \n{OmegaConf.to_yaml(self.cfg.decoding)}"
                )

        elif decoder_type == 'ctc':
            if not hasattr(self, 'ctc_decoding'):
                raise ValueError("The model does not have the ctc_decoding module and does not support ctc decoding.")
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.aux_ctc.decoding

            self.ctc_decoding = CTCDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer, blank_id=self.blank_id)

            self.ctc_wer = WER(
                decoding=self.ctc_decoding,
                use_cer=self.ctc_wer.use_cer,
                log_prediction=self.ctc_wer.log_prediction,
                dist_sync_on_step=True,
            )

            self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

            # Update config
            with open_dict(self.cfg.aux_ctc.decoding):
                self.cfg.aux_ctc.decoding = decoding_cfg

            self.cur_decoder = "ctc"
            if verbose:
                logging.info(
                    f"Changed decoding strategy of the CTC decoder to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}"
                )
        else:
            raise ValueError(f"decoder_type={decoder_type} is not supported. Supported values: [ctc,rnnt]")

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        context, target, attn_mask, position_ids, target_start, target_end, waveform, language_ids = batch

        # Do not pass length to the preprocessor, it will be computed in the preprocessor (padding as blank training)
        signal, signal_length = self.preprocessor(raw_speech=waveform, length=None)
        if self.spec_augmentation is not None and self.training:
            signal = self.spec_augmentation(input_spec=signal, length=signal_length)
        # forward() only performs encoder forward
        encoded = self.forward(input_signal=signal, language_ids=language_ids)
        encoded_len = torch.full((encoded.shape[0],), encoded.shape[2], device=encoded.device)

        if self.ctc_loss_weight > 0:
            simple_am, ctc_output = self.ctc_decoder(encoded, return_logits=True, return_softmax=True)
        else:
            simple_am = self.ctc_decoder(encoded, return_logits=True, return_softmax=False)

        # do not include the last token in the context for the decoder (this model does not predict eos token)
        decoded, _ = self.decoder(input_ids=context, attn_mask=attn_mask, position_ids=position_ids)
        if self.llm_head.tie_weights:
            if self.llm_loss is not None:
                simple_lm, lm_output = self.llm_head(decoded, return_logits=True, return_softmax=True, weights=self.llm_head_weights)
            else:
                simple_lm = self.llm_head(decoded, return_logits=True, return_softmax=False, weights=self.llm_head_weights)
        else:
            if self.llm_loss is not None:
                simple_lm, lm_output = self.llm_head(decoded, return_logits=True, return_softmax=True)
            else:
                simple_lm = self.llm_head(decoded, return_logits=True, return_softmax=False)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        if self.llm_loss is not None:
            _, _, v = simple_lm.size()
            # We do not predict the eos token, so we does not need the last output token 
            llm_loss = self.llm_loss(lm_output[:, :-1].reshape(-1, v), context[..., 1:].reshape(-1), position_ids, target_start, target_end)
        else:
            llm_loss = 0

        # Fused joint step
        simple_loss, rnnt_loss, wer, _, _ = self.joint.forward_fused_loss(
            encoder_outputs=encoded,
            decoder_outputs=decoded,
            simple_am=simple_am,
            simple_lm=simple_lm,
            am_only_scale=self.cfg.loss.get("am_only_scale", 0.0),
            lm_only_scale=self.cfg.loss.get("lm_only_scale", 0.25),
            s_range=self.cfg.loss.get("s_range", 5),
            delay_penalty=self.cfg.loss.get("delay_penalty", 0.0),
            blank_symbol=self.blank_id,
            encoder_lengths=encoded_len,
            transcripts=context,
            targets=target,
            target_start=target_start,
            target_end=target_end,
            compute_wer=compute_wer,
        )

        # Add auxiliary losses, if registered
        rnnt_loss = self.add_auxiliary_losses(rnnt_loss)

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs = {
            'train_simple_loss': simple_loss.detach().cpu().item(),
            'train_rnnt_loss': rnnt_loss.detach().cpu().item(),
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }
        if self.llm_loss is not None:
            tensorboard_logs.update({'train_llm_loss': llm_loss.detach().cpu().item()})

        if compute_wer:
            tensorboard_logs.update({'training_batch_wer': wer})

        simple_loss_weight = 0.5 if self.trainer.global_step > self.cfg.optim.sched.warmup_steps else 1.0 - 0.5 * (self.trainer.global_step / self.cfg.optim.sched.warmup_steps)
        rnnt_loss_weight = 1.0 if self.trainer.global_step > self.cfg.optim.sched.warmup_steps else 0.1 + 0.9 * (self.trainer.global_step / self.cfg.optim.sched.warmup_steps)
        if self.ctc_loss_weight > 0:
            ctc_loss = self.ctc_loss(
                log_probs=ctc_output, targets=target, input_lengths=encoded_len, target_lengths=target_end - target_start
            )
            tensorboard_logs.update({'train_ctc_loss': ctc_loss.detach().cpu().item()})
            loss_value = (1 - self.ctc_loss_weight - self.llm_loss_weight) * (simple_loss_weight * simple_loss + rnnt_loss_weight * rnnt_loss) + self.ctc_loss_weight * ctc_loss + self.llm_loss_weight * llm_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=ctc_output, 
                    predictions_lengths=encoded_len,
                    targets=target, 
                    targets_lengths=target_end - target_start,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})
        else:
            loss_value = (1 - self.llm_loss_weight) * (simple_loss_weight * simple_loss + rnnt_loss_weight * rnnt_loss) + self.llm_loss_weight * llm_loss

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, target, target_end - target_start, compute_wer=compute_wer
        )

        tensorboard_logs.update({'train_loss': loss_value.detach().cpu().item()})
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        return {'loss': loss_value}

    def forward(self, input_signal):
        encoded = self.encoder(audio_signal=input_signal)
        return encoded

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        context, target, attn_mask, position_ids, target_start, target_end, signal, language_ids = batch
        signal, _ = self.preprocessor(raw_speech=signal, length=None)
        encoded = self.forward(input_signal=signal, language_ids=language_ids)
        encoded_len = torch.full((encoded.shape[0],), encoded.shape[2], device=encoded.device)

        tensorboard_logs = {}
        
        compute_wer = True

        if self.compute_eval_loss:
            simple_am, ctc_output = self.ctc_decoder(encoded, return_logits=True, return_softmax=True)
            decoded, _ = self.decoder(input_ids=context[..., :-1], attn_mask=attn_mask, position_ids=position_ids)
            if self.llm_head.tie_weights:
                simple_lm = self.llm_head(decoded, return_logits=True, return_softmax=False, weights=self.llm_head_weights)
            else:
                simple_lm = self.llm_head(decoded, return_logits=True, return_softmax=False)
            ctc_loss = self.ctc_loss(
                log_probs=ctc_output, targets=target, input_lengths=encoded_len, target_lengths=target_end - target_start
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss.detach().cpu().item()
        else:
            ctc_output = self.ctc_decoder(encoded, return_logits=False, return_softmax=True)
            decoded = None
            simple_am = None
            simple_lm = None

        # Fused joint step
        encoded_len = torch.full((encoded.shape[0],), encoded.shape[2], device=encoded.device)
        simple_loss, rnnt_loss, wer, wer_num, wer_denom = self.joint.forward_fused_loss(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                simple_am=simple_am,
                simple_lm=simple_lm,
                am_only_scale=self.cfg.loss.get("am_only_scale", 0.0),
                lm_only_scale=self.cfg.loss.get("lm_only_scale", 0.25),
                s_range=self.cfg.loss.get("s_range", 5),
                encoder_lengths=encoded_len,
                transcripts=context,
                targets=target,
                target_start=target_start,
                target_end=target_end,
                compute_wer=compute_wer,
            )

        if simple_loss is not None:
            simple_loss_weight = 0.5 if self.trainer.global_step > self.cfg.optim.sched.warmup_steps else 1.0 - 0.5 * (self.trainer.global_step / self.cfg.optim.sched.warmup_steps)
            rnnt_loss_weight = 1.0 if self.trainer.global_step > self.cfg.optim.sched.warmup_steps else 0.1 + 0.9 * (self.trainer.global_step / self.cfg.optim.sched.warmup_steps)
            tensorboard_logs['val_rnnt_loss'] = rnnt_loss.detach().cpu().item()
            tensorboard_logs['val_simple_loss'] = simple_loss.detach().cpu().item()
            tensorboard_logs['val_ctc_loss'] = ctc_loss.detach().cpu().item()
            tensorboard_logs['val_loss'] = (1 - self.ctc_loss_weight) * (simple_loss_weight * simple_loss + rnnt_loss_weight * rnnt_loss) + self.ctc_loss_weight * ctc_loss.detach().cpu().item()

        tensorboard_logs['val_wer_num'] = wer_num
        tensorboard_logs['val_wer_denom'] = wer_denom
        tensorboard_logs['val_wer'] = wer


        self.ctc_wer.update(
            predictions=ctc_output,
            targets=target,
            targets_lengths=target_end - target_start,
            predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer
        tensorboard_logs['global_step'] = self.trainer.global_step

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        return tensorboard_logs

    def on_save_checkpoint(self, state_dict):
        # in order to resume training from the same point, we need this to prevent from dataloader prefetching the next batch
        actual_updated_samples = state_dict['global_step'] * self.trainer.accumulate_grad_batches
        current_batch_progress = actual_updated_samples % self.trainer.num_training_batches
        state_dict['loops']['fit_loop']['state_dict']['combined_loader'][0]['consumed_batches'] = actual_updated_samples
        # To resume from the actual updated samples, we need to set the batch progress to the actual updated samples
        state_dict['loops']['fit_loop']['epoch_loop.batch_progress']['total'] = {
            'ready': actual_updated_samples,
            'started': actual_updated_samples,
            'processed': actual_updated_samples,
            'completed': actual_updated_samples,
        }
        state_dict['loops']['fit_loop']['epoch_loop.batch_progress']['current'] = {
            'ready': current_batch_progress,
            'started': current_batch_progress,
            'processed': current_batch_progress,
            'completed': current_batch_progress,
        }

        super().on_save_checkpoint(state_dict)
        
