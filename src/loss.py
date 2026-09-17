"""Loss functions used by the CTC, distillation, and RNN-T models."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor

try:
    import k2
except ImportError:  # k2 is only required by the RNN-T losses below.
    k2 = None

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

__all__ = [
    'CTCLoss',
    'NLLLoss',
    'CosineSimilarityLoss',
    'MSELoss',
    'mutual_information_recursion',
    'rnnt_loss_pruned',
    'rnnt_loss_smoothed',
]


def _require_k2():
    if k2 is None:
        raise ImportError("k2 is required for RNN-T loss calculation")
    return k2


class _FastEmitMutualInformationRecursion(torch.autograd.Function):
    """Reuse k2's recursion and replace only its transition backward rule."""

    @staticmethod
    def forward(
        ctx,
        px: Tensor,
        py: Tensor,
        returned_grads: List[Optional[Tensor]],
        boundary: Optional[Tensor],
        fastemit_lambda: float,
    ) -> Tensor:
        k2_module = _require_k2()
        scores, (px_grad, py_grad) = k2_module.mutual_information_recursion(
            px=px,
            py=py,
            boundary=boundary,
            return_grad=True,
        )

        # px contains label transitions and py contains blank/frame transitions.
        px_grad = px_grad * (1.0 + fastemit_lambda)
        returned_grads[0] = px_grad
        returned_grads[1] = py_grad
        ctx.save_for_backward(px_grad, py_grad)
        return scores

    @staticmethod
    def backward(ctx, scores_grad: Tensor):
        px_grad, py_grad = ctx.saved_tensors
        scores_grad = scores_grad.reshape(-1, 1, 1)
        return (
            px_grad * scores_grad,
            py_grad * scores_grad,
            None,
            None,
            None,
        )


def mutual_information_recursion(
    px: Tensor,
    py: Tensor,
    boundary: Optional[Tensor] = None,
    fastemit_lambda: float = 0.0,
    return_grad: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
    """Run k2's recursion with FastEmit's non-blank gradient scaling.

    The forward score is the ordinary RNN-T score. FastEmit changes only the
    backward rule by multiplying label-transition gradients by
    ``1 + fastemit_lambda``; blank-transition gradients remain unchanged.
    """
    fastemit_lambda = float(fastemit_lambda)
    if fastemit_lambda < 0.0:
        raise ValueError(
            f"fastemit_lambda must be non-negative, got {fastemit_lambda}"
        )

    returned_grads: List[Optional[Tensor]] = [None, None]
    scores = _FastEmitMutualInformationRecursion.apply(
        px,
        py,
        returned_grads,
        boundary,
        fastemit_lambda,
    )
    if not return_grad:
        return scores

    px_grad, py_grad = returned_grads
    assert px_grad is not None and py_grad is not None
    return scores, (px_grad, py_grad)


def _add_delay_penalty(
    px: Tensor,
    boundary: Optional[Tensor],
    rnnt_type: str,
    delay_penalty: float,
) -> Tensor:
    """Apply the same label-transition delay penalty as k2's losses."""
    if delay_penalty <= 0.0:
        return px

    batch_size, _, time_size = px.shape
    num_frames = time_size if rnnt_type != "regular" else time_size - 1
    if boundary is None:
        offset = torch.tensor(
            (num_frames - 1) / 2,
            dtype=px.dtype,
            device=px.device,
        ).expand(batch_size, 1, 1)
    else:
        offset = ((boundary[:, 3] - 1) / 2).reshape(batch_size, 1, 1)

    penalty = offset - torch.arange(
        time_size, device=px.device
    ).reshape(1, 1, time_size)
    return px + penalty.to(px.dtype) * delay_penalty


def _loss_from_scores(scores: Tensor, reduction: Optional[str]) -> Tensor:
    if reduction == "none":
        return -scores
    if reduction == "mean":
        return -scores.mean()
    if reduction == "sum":
        return -scores.sum()
    raise ValueError(
        f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
    )


def rnnt_loss_smoothed(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    lm_only_scale: float = 0.1,
    am_only_scale: float = 0.1,
    boundary: Optional[Tensor] = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    fastemit_lambda: float = 0.0,
    reduction: Optional[str] = "mean",
    return_grad: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
    """Equivalent to ``k2.rnnt_loss_smoothed`` with FastEmit support."""
    k2_module = _require_k2()
    px, py = k2_module.get_rnnt_logprobs_smoothed(
        lm=lm,
        am=am,
        symbols=symbols,
        termination_symbol=termination_symbol,
        lm_only_scale=lm_only_scale,
        am_only_scale=am_only_scale,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )
    px = _add_delay_penalty(px, boundary, rnnt_type, delay_penalty)
    scores_and_grads = mutual_information_recursion(
        px=px,
        py=py,
        boundary=boundary,
        fastemit_lambda=fastemit_lambda,
        return_grad=return_grad,
    )
    if return_grad:
        scores, grads = scores_and_grads
        return _loss_from_scores(scores, reduction), grads
    return _loss_from_scores(scores_and_grads, reduction)


def rnnt_loss_pruned(
    logits: Tensor,
    symbols: Tensor,
    ranges: Tensor,
    termination_symbol: int,
    boundary: Optional[Tensor] = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    fastemit_lambda: float = 0.0,
    reduction: Optional[str] = "mean",
) -> Tensor:
    """Equivalent to ``k2.rnnt_loss_pruned`` with FastEmit support."""
    k2_module = _require_k2()
    px, py = k2_module.get_rnnt_logprobs_pruned(
        logits=logits,
        symbols=symbols,
        ranges=ranges,
        termination_symbol=termination_symbol,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )
    px = _add_delay_penalty(px, boundary, rnnt_type, delay_penalty)
    scores = mutual_information_recursion(
        px=px,
        py=py,
        boundary=boundary,
        fastemit_lambda=fastemit_lambda,
    )
    return _loss_from_scores(scores, reduction)

class MSELoss(nn.MSELoss, Serialization, Typing):
    def __init__(self, reduction='mean', **kwargs):
        super().__init__(reduction='none', **kwargs)
        self.finegrained_reduction = reduction
    @property
    def input_types(self):
        """Input types definitions for MSELoss.
        """
        return {
            "x": NeuralType(('B, T, D'), LogprobsType()),
            "y": NeuralType(('B, T, D'), LogprobsType()),
        }
    @property
    def output_types(self):
        """Output types definitions for MSELoss.
        """
        return {"loss": NeuralType(elements_type=LossType())}
    def forward(self, x, y):
        """Forward pass for MSELoss.
        """
        loss = super().forward(x, y)
        b, t = loss.shape
        if self.finegrained_reduction == 'mean':
            loss = loss / (b * t)
        elif self.finegrained_reduction == 'sum':
            loss = loss
        elif self.finegrained_reduction == 'mean_batch':
            loss = loss / b
        else:
            raise ValueError(f"Invalid reduction: {self.finegrained_reduction}")
        return loss

class CosineSimilarityLoss(nn.CosineSimilarity, Serialization, Typing):
    def __init__(self, dim=-1, scale=1000.0, reduction='mean', **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.finegrained_reduction = reduction
        self.scale = scale if scale < 0 else -1 * scale

    @property
    def input_types(self):
        """Input types definitions for CosineSimilarityLoss.
        """
        return {
            "x": NeuralType(('B, T, D'), LogprobsType()),
            "y": NeuralType(('B, T, D'), LogprobsType()),
        }
    @property
    def output_types(self):
        """Output types definitions for CosineSimilarityLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def forward(self, x, y):
        """Forward pass for CosineSimilarityLoss.
        """
        loss = super().forward(x, y).sum() * self.scale
        b, t, _ = x.shape
        if self.finegrained_reduction == 'mean':
            loss = loss / (b * t)
        elif self.finegrained_reduction == 'sum':
            loss = loss
        elif self.finegrained_reduction == 'mean_batch':
            loss = loss / b
        else:
            raise ValueError(f"Invalid reduction: {self.finegrained_reduction}")
        return loss

class NLLLoss(nn.NLLLoss, Serialization, Typing):
    def __init__(self, reduction='mean', ignore_index=-100, **kwargs):
        super().__init__(ignore_index=ignore_index, reduction="none", **kwargs)
        self.finegrained_reduction = reduction

    @property
    def input_types(self):
        """Input types definitions for NLLLoss.
        """
        return {
            "log_probs": NeuralType(('B * T * D'), LogprobsType()),
            "targets": NeuralType(('B * T'), LabelsType()),
            "target_start": NeuralType(('B'), LengthsType()),
            "target_end": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for NLLLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def forward(self, log_probs, targets, target_start, target_end):
        loss = super().forward(log_probs, targets)
        # Offset the target start and end by 1 because of next unit prediction (target is the shift one of input).
        # log_probs is pre-flattened from (B, L-1, V); recover (B, L-1) shape for the mask via arange.
        batch_size = target_start.shape[0]
        seq_len = log_probs.shape[0] // batch_size
        arange = torch.arange(seq_len, device=log_probs.device).unsqueeze(0)  # (1, L-1)
        target_mask = ((arange >= (target_start - 1).unsqueeze(1)) & (arange < (target_end - 1).unsqueeze(1))).reshape(-1)
        loss = (loss * target_mask).sum()
        if self.finegrained_reduction == 'mean':
            loss = loss / target_mask.sum()
        elif self.finegrained_reduction == 'sum':
            loss = loss
        elif self.finegrained_reduction == 'mean_batch':
            loss = loss / batch_size
            raise ValueError(f"Invalid reduction: {self.finegrained_reduction}")
        return loss

class CTCLoss(nn.CTCLoss, Serialization, Typing):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, zero_infinity=False, reduction='mean_batch', blank_id=0):
        self._blank = blank_id
        # Don't forget to properly call base constructor
        if reduction not in ['none', 'mean', 'sum', 'mean_batch']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch]')

        self.config_reduction = reduction
        if reduction == 'mean_batch':
            ctc_reduction = 'none'
            self._apply_reduction = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_reduction = False
        super().__init__(blank=self._blank, reduction=ctc_reduction, zero_infinity=zero_infinity)

    def reduce(self, losses, target_lengths):
        if self.config_reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        return losses

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # override forward implementation
        # custom logic, if necessary
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        log_probs = log_probs.transpose(1, 0)
        loss = super().forward(
            log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        )
        if self._apply_reduction:
            loss = self.reduce(loss, target_lengths)
        return loss
