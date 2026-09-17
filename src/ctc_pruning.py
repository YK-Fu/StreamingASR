"""CTC-posterior frame selection and compaction for pruned RNN-T."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FrameReductionResult:
    encoded: torch.Tensor
    lengths: torch.Tensor
    original_indices: torch.Tensor
    retained_frames: torch.Tensor
    valid_frames: torch.Tensor


@torch.no_grad()
def select_ctc_frames(
    ctc_log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: Optional[torch.Tensor],
    blank_id: int,
    blank_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded original-frame indices and the retained lengths."""
    if ctc_log_probs.ndim != 3:
        raise ValueError("ctc_log_probs must have shape [batch, time, vocabulary]")
    if not 0.0 <= blank_threshold <= 1.0:
        raise ValueError("blank_threshold must be in [0, 1]")
    batch_size, max_time, vocab_size = ctc_log_probs.shape
    if not 0 <= blank_id < vocab_size:
        raise ValueError("blank_id is outside the CTC vocabulary")
    if input_lengths.shape != (batch_size,):
        raise ValueError("input_lengths must have shape [batch]")
    if target_lengths is not None and target_lengths.shape != (batch_size,):
        raise ValueError("target_lengths must have shape [batch]")

    blank_probs = ctc_log_probs[..., blank_id].exp()
    retained: list[torch.Tensor] = []
    retained_lengths: list[int] = []
    for batch_idx in range(batch_size):
        input_length = int(input_lengths[batch_idx].item())
        if not 1 <= input_length <= max_time:
            raise ValueError("input_lengths must be in [1, max_time]")
        keep = torch.nonzero(
            blank_probs[batch_idx, :input_length] <= blank_threshold,
            as_tuple=False,
        ).squeeze(1)
        minimum = 1
        if target_lengths is not None:
            minimum = max(
                minimum,
                min(int(target_lengths[batch_idx].item()), input_length),
            )
        if keep.numel() < minimum:
            # Lowest blank-posterior frames are the most CTC-informative.
            keep = torch.topk(
                blank_probs[batch_idx, :input_length],
                k=minimum,
                largest=False,
                sorted=False,
            ).indices.sort().values
        retained.append(keep)
        retained_lengths.append(int(keep.numel()))

    max_retained = max(retained_lengths)
    indices = torch.zeros(
        (batch_size, max_retained), dtype=torch.long, device=ctc_log_probs.device
    )
    for batch_idx, keep in enumerate(retained):
        indices[batch_idx, : keep.numel()] = keep
        if keep.numel() < max_retained:
            indices[batch_idx, keep.numel() :] = keep[-1]
    return indices, torch.tensor(
        retained_lengths, dtype=torch.long, device=ctc_log_probs.device
    )


def reduce_ctc_frames(
    encoded: torch.Tensor,
    ctc_log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: Optional[torch.Tensor],
    blank_id: int,
    blank_threshold: float = 0.9,
) -> FrameReductionResult:
    """Physically compact encoder output using detached CTC blank posteriors."""
    if encoded.ndim != 3:
        raise ValueError("encoded must have shape [batch, channels, time]")
    if (
        encoded.shape[0] != ctc_log_probs.shape[0]
        or encoded.shape[2] != ctc_log_probs.shape[1]
    ):
        raise ValueError("encoded and ctc_log_probs shapes do not agree")
    indices, lengths = select_ctc_frames(
        ctc_log_probs=ctc_log_probs.detach(),
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id,
        blank_threshold=blank_threshold,
    )
    gathered = encoded.gather(
        2, indices.unsqueeze(1).expand(-1, encoded.shape[1], -1)
    )
    return FrameReductionResult(
        encoded=gathered,
        lengths=lengths,
        original_indices=indices,
        retained_frames=lengths.sum(),
        valid_frames=input_lengths.sum(),
    )
