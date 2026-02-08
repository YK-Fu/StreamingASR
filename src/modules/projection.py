import torch
from typing import Optional, Union, List

from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.collections.asr.parts.submodules.jasper import init_weights

import k2

class SimpleProj(NeuralModule, Exportable):
    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None, tie_weights=False):
        super().__init__()
        self.__vocabulary = vocabulary
        self._feat_in = feat_in
        self._num_classes = num_classes
        self.tie_weights = tie_weights
        if not self.tie_weights:
            self.decoder_layers = torch.nn.Linear(self._feat_in, self._num_classes, bias=False)

            self.apply(lambda x: init_weights(x, mode=init_mode))

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    def forward(self, encoder_output, return_logits=False, return_softmax=True, weights=None):
        assert return_logits or return_softmax, "Either return_logits or return_softmax must be True"
        if not self.tie_weights:
            logits = self.decoder_layers(encoder_output.transpose(1, 2))
        else:
            assert weights is not None, "weights must be provided if tie_weights is True"
            logits = torch.nn.functional.linear(encoder_output.transpose(1, 2), weights)
        if return_softmax:
            # cast to float to prevent from spiky logits (stablize training)
            softmax = torch.nn.functional.log_softmax(logits.float() / self.temperature, dim=-1)
            if return_logits:
                return logits, softmax
            else:
                return softmax
        else:
            return logits

    @property
    def num_classes_with_blank(self):
        return self._num_classes


class PrunedRNNTJoint(RNNTJoint):
    def forward(self, f, g, project_input=True):
        if project_input:
            return self.joint_after_projection(self.project_encoder(f), self.project_prednet(g), log_softmax=not self.training)
        else:
            return self.joint_after_projection(f, g, log_softmax=not self.training)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor, log_softmax: bool = True) -> torch.Tensor:
        # f, g -> [B, T, R, H] for training, [B, 1, H] for transcribing
        if f.ndim == 3:
            assert g.ndim == 3
            f = f.unsqueeze(dim=1)  # (B, 1, 1, H)
            g = g.unsqueeze(dim=1)  # (B, 1, 1, H)
        assert g.ndim == 4
        assert f.ndim == 4

        inp = f + g  # [B, T, R, H] or [B, 1, 1, H]

        del f, g

        res = self.joint_net(inp)  # [B, T, R, V + 1] or [B, 1, 1, V + 1]

        del inp


        if not log_softmax:
            res = (res / self.temperature).log_softmax(dim=-1)
        else:
            res = (res / self.temperature)

        return res

    def forward_fused_loss(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        simple_am: Optional[torch.Tensor] = None,
        simple_lm: Optional[torch.Tensor] = None,
        am_only_scale: float = 0.0,
        lm_only_scale: float = 0.25,
        s_range: int = 5,
        delay_penalty: float = 0.0,
        blank_symbol: int = 0,
        encoder_lengths: Optional[torch.Tensor] = None,
        transcripts: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_start: Optional[torch.Tensor] = None,
        target_end: Optional[torch.Tensor] = None,
        compute_wer: bool = False,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # encoder = (B, D, T)
        # decoder = (B, D, U) if passed, else None
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)

        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        if not self._fuse_loss_wer:
            if decoder_outputs is None:
                raise ValueError(
                    "decoder_outputs passed is None, and `fuse_loss_wer` is not set. "
                    "decoder_outputs can only be None for fused step!"
                )

            out = self.forward(encoder_outputs, decoder_outputs)  # [B, T, U, V + 1]
            return out

        else:
            # At least the loss module must be supplied during fused joint
            if self._wer is None:
                raise ValueError("`fuse_loss_wer` flag is set, but `wer` modules were not provided! ")

            # When using fused joint step, both encoder and transcript lengths must be provided
            if (encoder_lengths is None) or (targets is None) or (target_start is None) or (target_end is None):
                raise ValueError(
                    "`fuse_loss_wer` is set, therefore encoder and target lengths " "must be provided as well!"
                )
            if simple_lm is not None and simple_am is not None:
                symbols = transcripts[..., 1:]
                boundary = torch.zeros((encoder_outputs.shape[0], 4), dtype=torch.int64, device=encoder_outputs.device)
                boundary[:, 0] = target_start - 1
                boundary[:, 2] = target_end - 1
                boundary[:, 3] = encoder_lengths

                # We should always cast the inputs to float32 to prevent numerical instability
                simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                    lm=simple_lm.float(),
                    am=simple_am.float(),
                    symbols=symbols,
                    termination_symbol=blank_symbol,
                    lm_only_scale=lm_only_scale,
                    am_only_scale=am_only_scale,
                    delay_penalty=delay_penalty,
                    boundary=boundary,
                    reduction='sum',
                    return_grad=True,
                )

                ranges = k2.get_rnnt_prune_ranges(
                    px_grad=px_grad,
                    py_grad=py_grad,
                    boundary=boundary,
                    s_range=s_range,
                )

                enc_pruned, dec_pruned = k2.do_rnnt_pruning(
                    am=self.project_encoder(encoder_outputs),
                    lm=self.project_prednet(decoder_outputs),
                    ranges=ranges,
                )
                joint = self.forward(enc_pruned, dec_pruned, project_input=False)
                # We should always cast the inputs to float32 to prevent numerical instability
                rnnt_loss = k2.rnnt_loss_pruned(
                    logits=joint.float(),
                    symbols=symbols,
                    ranges=ranges,
                    termination_symbol=blank_symbol,
                    boundary=boundary,
                    delay_penalty=delay_penalty,
                    reduction='sum',
                )
                rnnt_loss = rnnt_loss / (target_end - target_start).sum()
                del joint, ranges, boundary
            else:
                simple_loss = None
                rnnt_loss = None

            # Update WER for sub batch
            if compute_wer:
                encoder_outputs = encoder_outputs.transpose(1, 2)  # [B, T, D] -> [B, D, T]
                encoder_outputs = encoder_outputs.detach()
                targets = targets.detach()

                # Update WER on each process without syncing
                if self.training:
                    original_sync = self.wer._to_sync
                    self.wer._to_sync = False

                self.wer.update(
                    predictions=encoder_outputs,
                    predictions_lengths=encoder_lengths,
                    targets=targets,
                    targets_lengths=target_end - target_start,
                )
                # Sync and all_reduce on all processes, compute global WER
                wer, wer_num, wer_denom = self.wer.compute()
                self.wer.reset()

                if self.training:
                    self.wer._to_sync = original_sync
            else:
                wer = None
                wer_num = None
                wer_denom = None

            return simple_loss, rnnt_loss, wer, wer_num, wer_denom
