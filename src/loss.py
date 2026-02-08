import torch
from torch import nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

__all__ = ['CTCLoss', 'NLLLoss', 'CosineSimilarityLoss', 'MSELoss']

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
            "position_ids": NeuralType(('B * T'), LengthsType()),
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

    def forward(self, log_probs, targets, position_ids, target_start, target_end):
        loss = super().forward(log_probs, targets)
        # Offset the target start and end by 1 because of next unit prediction (target is the shift one of input)
        target_mask = ((position_ids[:, :-1] >= (target_start - 1).unsqueeze(1)) & (position_ids[:, :-1] < (target_end - 1).unsqueeze(1))).reshape(-1)
        loss = (loss * target_mask).sum()
        if self.finegrained_reduction == 'mean':
            loss = loss / target_mask.sum()
        elif self.finegrained_reduction == 'sum':
            loss = loss
        elif self.finegrained_reduction == 'mean_batch':
            loss = loss / position_ids.shape[0]
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