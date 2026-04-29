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
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, InterCTCMixin
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.utils import logging
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.core.classes.mixins import AccessMixin
from nemo.utils.model_utils import maybe_update_config_version

from src.loss import CTCLoss, CosineSimilarityLoss, MSELoss
from src.decoding_utils import CTCDecoding
from src.datasets import get_asr_dataset, ResumableDataloader, ResumableSampler
from src.modules.evaluation import MixErrorRate, MixErrorRateMetric


class CausalWhisperDistilModel(ASRModel, ASRBPEMixin, InterCTCMixin):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""
    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        tokenizer = AutoTokenizer(tokenizer_cfg.path, pad_token=tokenizer_cfg.blank_token, bos_token=tokenizer_cfg.blank_token, trust_remote_code=True)
        self.tokenizer = tokenizer

    @classmethod
    def from_config_dict(cls, config: DictConfig, **kwargs):
        # recursively load _module_ with the target module
        import hydra
        config = maybe_update_config_version(config)
        if "_target_" in config:
            return hydra.utils.instantiate(config, **kwargs)
        return cls(cfg=config, **kwargs)

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        self.rank = 0
        if trainer is not None:
            self.world_size = trainer.world_size
            self.rank = trainer.global_rank
        self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)
        num_vocab = self.tokenizer.vocab_size
        self.blank_id = self.tokenizer.token_to_id(self.cfg.tokenizer.blank_token)
        self.preprocessor = CausalWhisperDistilModel.from_config_dict(self.cfg.preprocessor)
        with open_dict(self.cfg.teacher):
            self.cfg.teacher.freeze = True
        with open_dict(self.cfg.student):
            self.cfg.student.freeze = False
        self.teacher = CausalWhisperDistilModel.from_config_dict(self.cfg.teacher)
        self.teacher.eval()
        self.student = CausalWhisperDistilModel.from_config_dict(self.cfg.student)

        loss_type = self.cfg.distil_loss.get('type', 'cosine')
        self.distil_loss_scale = self.cfg.distil_loss.get('scale', 10.0)
        if loss_type == 'cosine':
            self.distil_loss = CosineSimilarityLoss(dim=-1, scale=self.distil_loss_scale, reduction=self.cfg.distil_loss.get('reduction', 'mean'))
        elif loss_type == 'mse':
            self.distil_loss = MSELoss(reduction=self.cfg.distil_loss.get('reduction', 'mean'))
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        if hasattr(self.cfg, 'spec_augment') and self.cfg.spec_augment is not None:
            self.spec_augmentation = CausalWhisperDistilModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        if 'compute_eval_loss' in self.cfg:
            self.compute_eval_loss = self.cfg.compute_eval_loss
        else:
            self.compute_eval_loss = True

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
        self.ctc_decoder = CausalWhisperDistilModel.from_config_dict(self.cfg.aux_ctc.decoder)     # This is also used as simple_am_proj
        self.ctc_loss_weight = self.cfg.aux_ctc.get("ctc_loss_weight", 0.5)
        self.ctc_loss = CTCLoss(
                    num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                    zero_infinity=True,
                    reduction="mean",
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

        # ************** 0429, Forbes: Setup MER for evaluation (DDP-safe) **************
        # MixErrorRate handles Chinese / English / code-switched references
        # (Chinese counted by character, English by word). MixErrorRateMetric
        # wraps it as a torchmetrics.Metric so its err / nref states are
        # all-reduced (sum) across ranks at compute() time, giving a true
        # micro-averaged MER instead of an average of per-batch ratios.
        self.mer_metric = MixErrorRate(to_simplified_chinese=True)
        self.ctc_mer = MixErrorRateMetric(self.mer_metric, dist_sync_on_step=False)
        # **********************************************************************


        # Setup optional Optimization flags
        self.setup_optimization_flags()

        self.setup_interctc(decoder_name='ctc_decoder', loss_name='ctc_loss', wer_name='ctc_wer')

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        dataset = get_asr_dataset(
            manifest_filepath=config.manifest_filepath,
            tokenizer=self.tokenizer,
            batch_size=config['batch_size'],
            sample_rate=config['sample_rate'],
            max_duration=config['max_duration'],
            min_duration=config['min_duration'],
            bucket_by=config.get('bucket_by', 'audio'),
            audio_chunk_size=config.get('audio_chunk_size', None),
            drop_last=config.get('drop_last', True),
            language_file=config.get('language_file', ""),
            language_drop_rate=config.get('language_drop_rate', 0.0),
            never_drop_language=config.get('never_drop_language', []),
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
        
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        _, target, _, _, target_start, target_end, waveform, language_id = batch
        target_len = target_end - target_start

        # Do not pass length to the preprocessor, it will be computed in the preprocessor (padding as blank training)
        signal, signal_length = self.preprocessor(raw_speech=waveform, length=None)
        if self.spec_augmentation is not None and self.training:
            student_signal = self.spec_augmentation(input_spec=signal, length=signal_length)
            teacher_signal = signal
        else:
            student_signal = teacher_signal = signal
        # forward() only performs encoder forward
        student_encoded = self.forward(input_signal=student_signal, language_ids=language_id)
        teacher_encoded = self.forward(input_signal=teacher_signal, mode='teacher')
        encoded_len = torch.full((student_encoded.shape[0],), student_encoded.shape[2], device=student_encoded.device)

        
        distil_loss = self.distil_loss(student_encoded, teacher_encoded)

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs = {
            'train_distil_loss': distil_loss.detach().cpu().item(),
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

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

        ctc_output = self.ctc_decoder(student_encoded, return_logits=False, return_softmax=True)

        if self.ctc_loss_weight > 0:
            ctc_loss = self.ctc_loss(
                log_probs=ctc_output, targets=target, input_lengths=encoded_len, target_lengths=target_len
            )
            tensorboard_logs.update({'train_ctc_loss': ctc_loss.detach().cpu().item()})
            loss_value = (1 - self.ctc_loss_weight) * distil_loss + self.ctc_loss_weight * ctc_loss
            # ************** 0429, Forbes: training-time MER instead of WER **************
            # Per-batch local-rank MER via compute_num_denom (no DDP sync — this
            # is just trend monitoring during training, not the final eval). Same
            # token-counting rule as val_wer_ctc, so the two numbers are directly
            # comparable. Only runs every log_every_n_steps (compute_wer gate).
            if compute_wer:
                pred_result = self.ctc_decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=ctc_output,
                    decoder_lengths=encoded_len,
                )
                pred_hyps = pred_result[0] if isinstance(pred_result, tuple) else pred_result
                pred_strs = [h.text if hasattr(h, 'text') else h for h in pred_hyps]
                ref_strs = [
                    self.tokenizer.ids_to_text(t[:l].tolist())
                    for t, l in zip(target, target_len)
                ]
                ctc_err, ctc_nref = self.mer_metric.compute_num_denom(
                    predictions=pred_strs, references=ref_strs,
                )
                if ctc_nref > 0:
                    tensorboard_logs.update({'training_batch_wer_ctc': ctc_err / ctc_nref})
            # **********************************************************************
        else:
            loss_value = distil_loss

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, target, target_len, compute_wer=compute_wer
        )

        tensorboard_logs.update({'train_loss': loss_value.detach().cpu().item()})
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        return {'loss': loss_value}

    def forward(self, input_signal, language_ids=None, mode='student'):
        if mode == 'student':
            encoded = self.student(audio_signal=input_signal, language_ids=language_ids)
        elif mode == 'teacher':
            encoded = self.teacher(audio_signal=input_signal, mode='teacher')
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return encoded

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        self.log_dict(logs, sync_dist=True)
        # ************** 0429, Forbes: surface per-batch MER on the tqdm bar **************
        # Lets you watch val_batch_wer_ctc tick down per step on long val sets,
        # while val_wer_ctc (epoch-level micro-average) remains the canonical
        # number logged in on_validation_epoch_end.
        if 'val_batch_wer_ctc' in logs:
            self.log(
                'val_batch_wer_ctc', logs['val_batch_wer_ctc'],
                prog_bar=True, on_step=True, on_epoch=False, sync_dist=False,
            )
        # **********************************************************************
        return logs

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        _, target, _, _, target_start, target_end, waveform, language_ids = batch
        target_len = target_end - target_start
        signal, _ = self.preprocessor(raw_speech=waveform, length=None)
        student_encoded = self.forward(input_signal=signal, language_ids=language_ids)
        teacher_encoded = self.forward(input_signal=signal, mode='teacher')
        encoded_len = torch.full((student_encoded.shape[0],), student_encoded.shape[2], device=student_encoded.device)

        tensorboard_logs = {}
        distil_loss = self.distil_loss_scale * self.distil_loss(student_encoded, teacher_encoded)
        tensorboard_logs['val_distil_loss'] = distil_loss.detach().cpu().item()

        compute_wer = True
        ctc_output = self.ctc_decoder(student_encoded, return_logits=False, return_softmax=True)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=ctc_output, targets=target, input_lengths=encoded_len, target_lengths=target_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss.detach().cpu().item()
        else:
            ctc_output = self.ctc_decoder(student_encoded, return_logits=False, return_softmax=True)



        # ************** 0429, Forbes: Mixed Error Rate (MER) instead of WER for evaluation **************
        # Two MER values are produced from the same decoded strings:
        #   * self.ctc_mer.update(...) accumulates err / nref into the metric
        #     state. compute() is called once per epoch in
        #     on_validation_epoch_end and returns the canonical, micro-averaged,
        #     DDP-synchronised val_wer_ctc.
        #   * val_batch_wer_ctc is the per-batch local-rank MER, computed
        #     directly from compute_num_denom (no DDP sync). It is just for
        #     progress visibility — it is mathematically a per-batch ratio,
        #     not the micro-average. Use val_wer_ctc as the canonical metric.
        pred_result = self.ctc_decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=ctc_output,
            decoder_lengths=encoded_len,
        )
        pred_hyps = pred_result[0] if isinstance(pred_result, tuple) else pred_result
        pred_strs = [h.text if hasattr(h, 'text') else h for h in pred_hyps]
        ref_strs = [
            self.tokenizer.ids_to_text(t[:l].tolist())
            for t, l in zip(target, target_len)
        ]
        self.ctc_mer.update(predictions=pred_strs, references=ref_strs)
        batch_err, batch_nref = self.mer_metric.compute_num_denom(
            predictions=pred_strs, references=ref_strs,
        )
        if batch_nref > 0:
            tensorboard_logs['val_batch_wer_ctc'] = batch_err / batch_nref
        # **********************************************************************
        tensorboard_logs['global_step'] = self.trainer.global_step

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        return tensorboard_logs

    def on_validation_epoch_end(self):
        # ************** 0429, Forbes: epoch-level MER aggregation (DDP-safe) **************
        # compute() triggers an all-reduce sum on err / nref across ranks,
        # so every rank sees the same micro-averaged MER. We then log once
        # with sync_dist=False (already synchronized).
        mer, err, nref = self.ctc_mer.compute()
        if int(nref.item()) > 0:
            self.log('val_wer_ctc',       mer,                            sync_dist=False, prog_bar=True)
            self.log('val_wer_num_ctc',   err.to(torch.float32),          sync_dist=False)
            self.log('val_wer_denom_ctc', nref.to(torch.float32),         sync_dist=False)
        self.ctc_mer.reset()
        # **********************************************************************
        parent = getattr(super(), 'on_validation_epoch_end', None)
        if callable(parent):
            return parent()

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

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