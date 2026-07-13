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

import gc

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
from src.modules.projection import DistilTimeUpsampler


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

        # If the student subsamples more than the teacher (e.g. student 8x vs teacher
        # 2x), the student output is shorter in time and cannot be compared frame-for-
        # frame with the teacher. Add a time-upsampling projector (activation + Linear +
        # sub-pixel reshape) applied ONLY to the distillation loss path.
        teacher_sf = getattr(self.teacher, 'subsampling_factor', 2)
        student_sf = getattr(self.student, 'subsampling_factor', 2)
        if student_sf % teacher_sf != 0:
            raise ValueError(
                f"student subsampling factor ({student_sf}) must be a multiple of the "
                f"teacher's ({teacher_sf}) so student frames upsample to teacher frames")
        distil_ratio = student_sf // teacher_sf
        if distil_ratio > 1:
            proj_act = self.cfg.distil_loss.get('projector_activation', 'gelu')
            self.distil_projector = DistilTimeUpsampler(self.cfg.student.d_model, distil_ratio, proj_act)
            logging.info(
                f"Distillation frame-rate ratio student/teacher = {distil_ratio}; added "
                f"time-upsampling projector Linear({self.cfg.student.d_model}, "
                f"{distil_ratio * self.cfg.student.d_model}) on the distil-loss path.")
        else:
            self.distil_projector = None

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
            dist_sync_on_step=False,
            log_prediction=self.cfg.get("log_prediction", False),
        )

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
            audio_chunk_step=config.get('audio_chunk_step', None),
            drop_last=config.get('drop_last', True),
            language_file=config.get('language_file', ""),
            language_drop_rate=config.get('language_drop_rate', 0.0),
            never_drop_language=config.get('never_drop_language', []),
            augmentation=config.get('augmentation', None),
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
            persistent_workers=config.get('num_workers', 0) > 0,
        )
        
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        _, target, _, target_start, target_end, waveform, language_id = batch
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

        # Match the student frame rate to the teacher for the distillation loss only.
        # The CTC head below still consumes the low-rate `student_encoded`.
        if self.distil_projector is not None:
            student_for_distil = self.distil_projector(student_encoded)
            # conv floor-rounding can leave the upsampled length a few frames off the
            # teacher's; align on the shorter length before the elementwise/cosine loss.
            if student_for_distil.shape[2] != teacher_encoded.shape[2]:
                L = min(student_for_distil.shape[2], teacher_encoded.shape[2])
                student_for_distil, teacher_encoded = student_for_distil[..., :L], teacher_encoded[..., :L]
        else:
            student_for_distil = student_encoded
        distil_loss = self.distil_loss(student_for_distil, teacher_encoded)
        del teacher_encoded, student_for_distil  # free before next large alloc

        # NOTE: do NOT reset the access registry here. The interctc layer-15 tensor
        # captured during the student forward must survive until add_interctc_losses()
        # below. reset_registry() also disables access, which would make
        # add_interctc_losses() early-return and silently drop the interctc loss.
        # The end-of-step reset (after add_interctc_losses) handles cleanup.

        tensorboard_logs = {
            'train_distil_loss': distil_loss.detach(),
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
        }

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        compute_wer = (sample_id + 1) % log_every_n_steps == 0

        ctc_output = self.ctc_decoder(student_encoded, return_logits=False, return_softmax=True)
        del student_encoded  # free encoder output; ctc_output already holds the projected result

        if self.ctc_loss_weight > 0:
            ctc_loss = self.ctc_loss(
                log_probs=ctc_output, targets=target, input_lengths=encoded_len, target_lengths=target_len
            )
            tensorboard_logs.update({'train_ctc_loss': ctc_loss.detach()})
            loss_value = (1 - self.ctc_loss_weight) * distil_loss + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=ctc_output,
                    predictions_lengths=encoded_len,
                    targets=target,
                    targets_lengths=target_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer.detach()})
        else:
            loss_value = distil_loss
        del ctc_output

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, target, target_len, compute_wer=compute_wer
        )
        # add_interctc_losses returns the per-layer interctc metrics (inter_ctc_loss_l15,
        # inter_wer_l15, final_loss) but does NOT log them itself. Merge them in so they
        # reach wandb/tensorboard — matches NeMo's stock training_step. The loss itself is
        # already folded into loss_value above; this only surfaces the breakdown.
        tensorboard_logs.update(additional_logs)

        tensorboard_logs.update({'train_loss': loss_value.detach()})
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        return {'loss': loss_value}

    @property
    def encoder(self):
        # InterCTCMixin.get_captured_interctc_tensors() reads the captured
        # interctc tensors from self.encoder's submodule registries. Our encoder
        # is the student. Expose it as a property (not a registered submodule) so
        # the mixin finds the capture without double-registering student's params.
        #
        # If torch.compile wrapped the student in an OptimizedModule, return the
        # underlying module: named_modules() on the wrapper surfaces the same
        # _registry under both '' and '_orig_mod', which trips the mixin's
        # "layer ... has been logged multiple times!" dedup guard.
        return getattr(self.student, '_orig_mod', self.student)

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
        # No self.log here — logging happens in on_validation_epoch_end to avoid
        # Lightning's "same key twice with different arguments" error across dataloaders.
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            # NeMo's property only initializes per-dataloader sublists on first access if
            # _validation_dl is already a list>1 at that moment; and our epoch-end resets
            # the cache. Lazy-extend here so dataloader_idx is always valid.
            outputs = self.validation_step_outputs
            while len(outputs) <= dataloader_idx:
                outputs.append([])
            outputs[dataloader_idx].append(logs)
        else:
            self.validation_step_outputs.append(logs)
        return logs

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            super().on_validation_epoch_end()
            return

        if isinstance(outputs[0], list):
            # Per-dataloader val_wer_ctc for wandb
            for i, dl_outputs in enumerate(outputs):
                if dl_outputs:
                    dl_wer = sum(o['val_wer_ctc'] for o in dl_outputs) / len(dl_outputs)
                    self.log(f'val_wer_ctc_dl{i}', dl_wer, sync_dist=True)
            all_outputs = [o for dl in outputs for o in dl]
        else:
            all_outputs = list(outputs)

        if all_outputs:
            keys = [k for k in all_outputs[0] if k != 'val_wer_ctc']
            # Macro avg of val_wer_ctc (bare key) for checkpoint monitor
            self.log('val_wer_ctc', sum(o['val_wer_ctc'] for o in all_outputs) / len(all_outputs), sync_dist=True)
            # Macro avg of all other metrics (no suffix), sit alongside training logs in wandb
            for key in keys:
                self.log(key, sum(o[key] for o in all_outputs) / len(all_outputs), sync_dist=True)

        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()
        super().on_validation_epoch_end()

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()

    def on_after_backward(self):
        # Replaces NeMo ASRModel.on_after_backward (Python-loop scan over every
        # named_parameter with per-param isnan/isinf + .any() syncs). One fused
        # _foreach_norm call propagates nan/inf into a single scalar; one host
        # sync at the end. Intentionally does NOT call super() to skip NeMo's
        # slow path — Lightning's base on_after_backward is a no-op.
        if not getattr(self, '_skip_nan_grad', False):
            return
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return
        norms = torch._foreach_norm(grads, 2.0)
        total = torch.stack(norms).sum()
        valid = torch.isfinite(total).to(torch.float32).view(1)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
        if valid.item() < 1:
            logging.warning('detected inf or nan values in gradients! Setting gradients to zero.')
            self.zero_grad()

    @torch.no_grad()
    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        _, target, _, target_start, target_end, waveform, language_ids = batch
        target_len = target_end - target_start
        signal, _ = self.preprocessor(raw_speech=waveform, length=None)
        student_encoded = self.forward(input_signal=signal, language_ids=language_ids)
        teacher_encoded = self.forward(input_signal=signal, mode='teacher')
        encoded_len = torch.full((student_encoded.shape[0],), student_encoded.shape[2], device=student_encoded.device)

        tensorboard_logs = {}
        # distil_loss already incorporates the scale internally (CosineSimilarityLoss(scale=...))
        if self.distil_projector is not None:
            student_for_distil = self.distil_projector(student_encoded)
            if student_for_distil.shape[2] != teacher_encoded.shape[2]:
                L = min(student_for_distil.shape[2], teacher_encoded.shape[2])
                student_for_distil, teacher_encoded = student_for_distil[..., :L], teacher_encoded[..., :L]
        else:
            student_for_distil = student_encoded
        distil_loss = self.distil_loss(student_for_distil, teacher_encoded)
        tensorboard_logs['val_distil_loss'] = distil_loss.detach()
        del teacher_encoded, student_for_distil

        ctc_output = self.ctc_decoder(student_encoded, return_logits=False, return_softmax=True)
        del student_encoded

        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=ctc_output, targets=target, input_lengths=encoded_len, target_lengths=target_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss.detach()

        self.ctc_wer.update(
            predictions=ctc_output,
            targets=target,
            targets_lengths=target_len,
            predictions_lengths=encoded_len,
        )
        del ctc_output
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num.detach()
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom.detach()
        tensorboard_logs['val_wer_ctc'] = ctc_wer.detach()
        tensorboard_logs['global_step'] = self.trainer.global_step

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        return tensorboard_logs

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

        state_dict['rng_state'] = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
        }

        super().on_save_checkpoint(state_dict)

    def on_load_checkpoint(self, state_dict):
        if 'rng_state' in state_dict:
            torch.set_rng_state(state_dict['rng_state']['torch'])
            torch.cuda.set_rng_state_all(state_dict['rng_state']['cuda'])
        super().on_load_checkpoint(state_dict)