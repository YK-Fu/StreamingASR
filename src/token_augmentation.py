import torch
from torchaudio.functional import forced_align as torchaudio_forced_align


@torch.no_grad()
def ctc_viterbi_token_frames(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate one CTC frame for every reference token with Viterbi alignment.

    The alignment is constrained to the reference sequence, but the returned
    frame indices can subsequently be used to read *unconstrained* CTC token
    predictions. Repeated adjacent reference tokens are handled by the standard
    inter-token blank state.

    Returns:
        ``token_frames`` with shape ``[B, U_max]`` and a boolean
        ``alignment_valid`` mask with shape ``[B]``. Invalid alignments use
        frame zero and should retain clean transcript history in the caller.
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape [batch, time, vocabulary]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, target_time]")
    batch_size, max_time, vocab_size = log_probs.shape
    if targets.shape[0] != batch_size:
        raise ValueError("targets and log_probs must have the same batch size")
    if input_lengths.shape != (batch_size,) or target_lengths.shape != (batch_size,):
        raise ValueError("input_lengths and target_lengths must have shape [batch]")
    if max_time == 0:
        raise ValueError("log_probs must contain at least one time step")
    if not 0 <= blank_id < vocab_size:
        raise ValueError("blank_id is outside the CTC vocabulary")
    if torch.any(input_lengths < 1) or torch.any(input_lengths > max_time):
        raise ValueError("input_lengths must be in [1, max_time]")
    if torch.any(target_lengths < 0) or torch.any(target_lengths > targets.shape[1]):
        raise ValueError("target_lengths are outside the padded target width")
    if targets.numel() and (torch.any(targets < 0) or torch.any(targets >= vocab_size)):
        raise ValueError("targets contain token IDs outside the CTC vocabulary")

    device = log_probs.device
    input_lengths = input_lengths.to(device=device, dtype=torch.long)
    target_lengths = target_lengths.to(device=device, dtype=torch.long)
    targets = targets.to(device=device, dtype=torch.long)
    max_target = targets.shape[1]

    if max_target == 0:
        return targets.new_zeros((batch_size, 0)), torch.ones(
            batch_size, dtype=torch.bool, device=device
        )

    token_positions = torch.arange(max_target, device=device).unsqueeze(0)
    valid_target_tokens = token_positions < target_lengths.unsqueeze(1)
    repeated = torch.zeros_like(valid_target_tokens)
    repeated[:, 1:] = (
        (targets[:, 1:] == targets[:, :-1])
        & valid_target_tokens[:, 1:]
    )
    minimum_input_lengths = target_lengths + repeated.sum(dim=1)
    contains_blank = ((targets == blank_id) & valid_target_tokens).any(dim=1)
    alignment_valid = (input_lengths >= minimum_input_lengths) & ~contains_blank

    # The compiled torchaudio CPU/CUDA operator currently accepts B=1. Calling
    # it once per microbatch sample still avoids a Python loop over acoustic
    # frames, which is the performance-sensitive dimension.
    paths = torch.full(
        (batch_size, max_time), blank_id, dtype=torch.long, device=device
    )
    path_scores = log_probs.new_full((batch_size, max_time), float("-inf"))
    input_length_values = input_lengths.detach().cpu().tolist()
    target_length_values = target_lengths.detach().cpu().tolist()
    structurally_valid_values = alignment_valid.detach().cpu().tolist()
    for batch_idx, structurally_valid in enumerate(structurally_valid_values):
        input_length = input_length_values[batch_idx]
        target_length = target_length_values[batch_idx]
        if not structurally_valid or target_length == 0:
            continue
        sample_path, sample_scores = torchaudio_forced_align(
            log_probs=log_probs[
                batch_idx : batch_idx + 1, :input_length
            ],
            targets=targets[batch_idx : batch_idx + 1, :target_length],
            blank=blank_id,
        )
        paths[batch_idx, :input_length] = sample_path[0]
        path_scores[batch_idx, :input_length] = sample_scores[0]

    # Every nonblank run in a valid CTC path corresponds to exactly one target
    # position. Repeated reference labels are separate runs because CTC requires
    # an intervening blank.
    previous_paths = torch.cat(
        (paths.new_full((batch_size, 1), blank_id), paths[:, :-1]), dim=1
    )
    time_positions = torch.arange(max_time, device=device).unsqueeze(0)
    active_frames = time_positions < input_lengths.unsqueeze(1)
    path_is_token = (paths != blank_id) & active_frames
    run_starts = path_is_token & (
        (previous_paths == blank_id) | (previous_paths != paths)
    )
    token_slots = run_starts.cumsum(dim=1) - 1
    token_slots = torch.where(
        path_is_token, token_slots, token_slots.new_full((), max_target)
    )
    recovered_lengths = run_starts.sum(dim=1)
    alignment_valid &= recovered_lengths == target_lengths
    path_is_token &= alignment_valid.unsqueeze(1)
    token_slots = torch.where(
        path_is_token, token_slots, token_slots.new_full((), max_target)
    )
    path_scores = path_scores.masked_fill(~path_is_token, float("-inf"))

    best_scores = log_probs.new_full((batch_size, max_target + 1), float("-inf"))
    best_scores.scatter_reduce_(1, token_slots, path_scores, reduce="amax", include_self=True)
    is_best = path_is_token & (path_scores == best_scores.gather(1, token_slots))
    candidate_frames = torch.where(
        is_best, time_positions.expand(batch_size, -1), time_positions.new_full((), max_time)
    )
    token_frames = torch.full(
        (batch_size, max_target + 1), max_time, dtype=torch.long, device=device
    )
    token_frames.scatter_reduce_(
        1, token_slots, candidate_frames, reduce="amin", include_self=True
    )
    token_frames = token_frames[:, :max_target].clamp(max=max_time - 1)
    return token_frames, alignment_valid


def _ctc_best_nonblank_ids(log_probs: torch.Tensor, blank_id: int) -> torch.Tensor:
    """Return the highest-scoring nonblank token without copying B*T*V logits."""
    vocab_size = log_probs.shape[-1]
    if vocab_size < 2:
        raise ValueError("CTC vocabulary must contain blank and at least one token")
    if blank_id == 0:
        return log_probs[..., 1:].argmax(dim=-1) + 1
    if blank_id == vocab_size - 1:
        return log_probs[..., :-1].argmax(dim=-1)

    left_score, left_id = log_probs[..., :blank_id].max(dim=-1)
    right_score, right_offset = log_probs[..., blank_id + 1 :].max(dim=-1)
    right_id = right_offset + blank_id + 1
    return torch.where(left_score >= right_score, left_id, right_id)


@torch.no_grad()
def ctc_aligned_token_substitution(
    input_ids: torch.Tensor,
    target_start: torch.Tensor,
    target_end: torch.Tensor,
    targets: torch.Tensor,
    ctc_log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace reference-history tokens with length-matched CTC predictions.

    CTC Viterbi forced alignment supplies exactly one acoustic frame per clean
    reference token. The unrestricted nonblank argmax at that frame supplies
    the candidate predictor input. Clean targets and all non-transcript input
    positions remain unchanged.

    Returns:
        Augmented input IDs; selected, changed, disagreeing, and eligible token
        counts; and the number of utterances without a valid CTC alignment.
    """
    if input_ids.ndim != 2 or targets.ndim != 2:
        raise ValueError("input_ids and targets must both have shape [batch, sequence]")
    if input_ids.shape[0] != targets.shape[0] or input_ids.shape[0] != ctc_log_probs.shape[0]:
        raise ValueError("input_ids, targets, and ctc_log_probs batch sizes must match")
    if target_start.shape != target_end.shape or target_start.shape != (input_ids.shape[0],):
        raise ValueError("target_start and target_end must have shape [batch]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")

    target_lengths = (target_end - target_start).to(dtype=torch.long)
    if torch.any(target_start < 0) or torch.any(target_end < target_start):
        raise ValueError("target boundaries must satisfy 0 <= start <= end")
    if torch.any(target_end > input_ids.shape[1]):
        raise ValueError("target_end exceeds the decoder input width")
    if torch.any(target_lengths > targets.shape[1]):
        raise ValueError("target lengths exceed the padded targets width")

    token_frames, alignment_valid = ctc_viterbi_token_frames(
        log_probs=ctc_log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id,
    )
    max_target = targets.shape[1]
    token_positions = torch.arange(max_target, device=input_ids.device).unsqueeze(0)
    eligible = (
        token_positions < target_lengths.to(device=input_ids.device).unsqueeze(1)
    ) & alignment_valid.to(device=input_ids.device).unsqueeze(1)
    eligible_count = eligible.sum()
    alignment_failures = (~alignment_valid).sum()
    if probability == 0.0 or eligible_count == 0:
        zero = eligible_count.new_zeros(())
        return input_ids, zero, zero, zero, eligible_count, alignment_failures

    best_nonblank = _ctc_best_nonblank_ids(ctc_log_probs, blank_id)
    aligned_predictions = best_nonblank.gather(1, token_frames)
    disagreements = eligible & (aligned_predictions != targets.to(aligned_predictions.device))
    selected = eligible & (
        torch.rand(eligible.shape, device=input_ids.device) < probability
    )

    decoder_positions = target_start.to(input_ids.device).unsqueeze(1) + token_positions
    selected_batch, selected_token = selected.nonzero(as_tuple=True)
    augmented = input_ids.clone()
    augmented[
        selected_batch, decoder_positions[selected_batch, selected_token]
    ] = aligned_predictions[selected_batch, selected_token]
    changed = selected & disagreements
    return (
        augmented,
        selected.sum(),
        changed.sum(),
        disagreements.sum(),
        eligible_count,
        alignment_failures,
    )


@torch.no_grad()
def mask_token_attention(
    attention_mask: torch.Tensor,
    token_end: torch.Tensor,
    probability: float,
    token_start: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hide randomly selected text tokens from causal attention.

    The fixed ``[BOS, language]`` prefix occupies positions before
    ``token_start``. Manifest context and transcript tokens are both eligible
    in ``[token_start, token_end)``. Selected positions remain in ``input_ids``
    so the decoder and RNN-T target axes keep the same length, but their
    key/value entries are unavailable to subsequent decoder positions. This is
    an inexpensive approximation of token deletion rather than physical
    sequence compaction.

    Returns:
        The augmented attention mask, selected-token count, and eligible-token
        count. Selection is independent for every eligible token.
    """
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if token_end.shape != (attention_mask.shape[0],):
        raise ValueError("token_end must have shape [batch]")
    if token_start < 0 or torch.any(token_end < token_start):
        raise ValueError("token boundaries must satisfy 0 <= start <= end")
    if torch.any(token_end > attention_mask.shape[1]):
        raise ValueError("token_end exceeds the attention-mask width")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")

    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    eligible = (
        (positions >= token_start)
        & (positions < token_end.to(attention_mask.device).unsqueeze(1))
        & attention_mask.bool()
    )
    eligible_count = eligible.sum()
    if probability == 0.0 or eligible_count == 0:
        return attention_mask, eligible_count.new_zeros(()), eligible_count

    selected = eligible & (
        torch.rand(attention_mask.shape, device=attention_mask.device) < probability
    )
    augmented = attention_mask.clone()
    augmented.masked_fill_(selected, 0)
    return augmented, selected.sum(), eligible_count


def _levenshtein_insertions(
    reference: list[int], hypothesis: list[int]
) -> list[tuple[int, int]]:
    """Return hypothesis insertions as ``(reference_boundary, token_id)``.

    ``reference_boundary`` is the number of reference tokens consumed before
    the extra hypothesis token. Diagonal substitutions are preferred over a
    delete-plus-insert pair when several minimum-cost alignments are possible.
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    costs = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for ref_idx in range(1, ref_len + 1):
        costs[ref_idx][0] = ref_idx
    for hyp_idx in range(1, hyp_len + 1):
        costs[0][hyp_idx] = hyp_idx

    for ref_idx in range(1, ref_len + 1):
        ref_token = reference[ref_idx - 1]
        for hyp_idx in range(1, hyp_len + 1):
            substitution_cost = int(ref_token != hypothesis[hyp_idx - 1])
            costs[ref_idx][hyp_idx] = min(
                costs[ref_idx - 1][hyp_idx] + 1,
                costs[ref_idx][hyp_idx - 1] + 1,
                costs[ref_idx - 1][hyp_idx - 1] + substitution_cost,
            )

    insertions = []
    ref_idx = ref_len
    hyp_idx = hyp_len
    while ref_idx > 0 or hyp_idx > 0:
        if (
            ref_idx > 0
            and hyp_idx > 0
            and reference[ref_idx - 1] == hypothesis[hyp_idx - 1]
            and costs[ref_idx][hyp_idx] == costs[ref_idx - 1][hyp_idx - 1]
        ):
            ref_idx -= 1
            hyp_idx -= 1
        elif (
            ref_idx > 0
            and hyp_idx > 0
            and costs[ref_idx][hyp_idx]
            == costs[ref_idx - 1][hyp_idx - 1] + 1
        ):
            ref_idx -= 1
            hyp_idx -= 1
        elif (
            ref_idx > 0
            and costs[ref_idx][hyp_idx] == costs[ref_idx - 1][hyp_idx] + 1
        ):
            ref_idx -= 1
        else:
            if hyp_idx == 0:
                raise RuntimeError("Invalid Levenshtein backtrace")
            insertions.append((ref_idx, hypothesis[hyp_idx - 1]))
            hyp_idx -= 1

    insertions.reverse()
    return insertions


def _collapse_ctc_path(
    frame_ids: list[int], blank_id: int, protected_ids: set[int]
) -> list[int]:
    """Collapse repeats and blanks from a greedy CTC frame path."""
    collapsed = []
    previous = None
    for token_id in frame_ids:
        if token_id != previous and token_id != blank_id and token_id not in protected_ids:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


@torch.no_grad()
def ctc_insertion_recovery(
    input_ids: torch.Tensor,
    target_start: torch.Tensor,
    target_end: torch.Tensor,
    targets: torch.Tensor,
    ctc_log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int,
    pad_token_id: int,
    sample_probability: float,
    text_bucket_size: int | None,
    protected_ids: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Insert one genuine CTC hypothesis insertion into selected samples.

    The collapsed greedy CTC hypothesis is Levenshtein-aligned to the clean
    transcript. When a selected sample has insertion errors, one is chosen
    uniformly and inserted into the Qwen input at its aligned reference
    boundary. ``state_indices`` maps Qwen outputs back to the original clean
    predictor axis: the state after the inserted token replaces the state at
    that boundary, teaching recovery without adding an RNN-T target.

    The augmented width is rounded up to ``text_bucket_size`` after insertion,
    never merely extended by one position.

    Returns:
        Augmented input IDs, augmented attention mask, clean-axis output-state
        indices, selected-sample count, candidate-insertion count, and applied-
        sample count. At most one insertion is applied per sample.
    """
    if input_ids.ndim != 2 or targets.ndim != 2 or ctc_log_probs.ndim != 3:
        raise ValueError(
            "input_ids, targets, and ctc_log_probs must have shapes "
            "[batch, sequence], [batch, target], and [batch, time, vocabulary]"
        )
    batch_size, original_width = input_ids.shape
    if targets.shape[0] != batch_size or ctc_log_probs.shape[0] != batch_size:
        raise ValueError("input_ids, targets, and ctc_log_probs batch sizes must match")
    if target_start.shape != (batch_size,) or target_end.shape != (batch_size,):
        raise ValueError("target_start and target_end must have shape [batch]")
    if input_lengths.shape != (batch_size,):
        raise ValueError("input_lengths must have shape [batch]")
    if torch.any(target_start < 1) or torch.any(target_end < target_start):
        raise ValueError("target boundaries must satisfy 1 <= start <= end")
    if torch.any(target_end > original_width):
        raise ValueError("target_end exceeds the decoder input width")
    target_lengths = target_end - target_start
    if torch.any(target_lengths > targets.shape[1]):
        raise ValueError("target lengths exceed the padded target width")
    max_ctc_time = ctc_log_probs.shape[1]
    if torch.any(input_lengths < 1) or torch.any(input_lengths > max_ctc_time):
        raise ValueError("input_lengths must be in [1, ctc_time]")
    if not 0.0 <= sample_probability <= 1.0:
        raise ValueError("sample_probability must be in [0, 1]")
    if text_bucket_size is not None and text_bucket_size < 1:
        raise ValueError("text_bucket_size must be positive or None")

    selected = torch.rand(batch_size, device=input_ids.device) < sample_probability
    selected_values = selected.detach().cpu().tolist()
    selected_count = selected.sum()
    identity_indices = torch.arange(
        original_width, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    if not any(selected_values):
        attention_mask = (input_ids != pad_token_id).long()
        attention_mask[:, 0] = 1
        zero = selected_count.new_zeros(())
        return input_ids, attention_mask, identity_indices, selected_count, zero, zero

    protected = {blank_id}
    if protected_ids is not None:
        protected.update(protected_ids.detach().cpu().tolist())
    greedy_paths = ctc_log_probs.argmax(dim=-1).detach().cpu()
    targets_cpu = targets.detach().cpu()
    input_length_values = input_lengths.detach().cpu().tolist()
    target_length_values = target_lengths.detach().cpu().tolist()

    chosen_insertions: list[tuple[int, int] | None] = [None] * batch_size
    candidate_count = 0
    for batch_idx, is_selected in enumerate(selected_values):
        if not is_selected:
            continue
        reference = targets_cpu[
            batch_idx, : target_length_values[batch_idx]
        ].tolist()
        hypothesis = _collapse_ctc_path(
            greedy_paths[batch_idx, : input_length_values[batch_idx]].tolist(),
            blank_id=blank_id,
            protected_ids=protected,
        )
        candidates = _levenshtein_insertions(reference, hypothesis)
        candidate_count += len(candidates)
        if candidates:
            candidate_idx = int(
                torch.randint(len(candidates), (), device=input_ids.device).item()
            )
            chosen_insertions[batch_idx] = candidates[candidate_idx]

    applied_count_value = sum(item is not None for item in chosen_insertions)
    candidate_count_tensor = selected_count.new_tensor(candidate_count)
    applied_count = selected_count.new_tensor(applied_count_value)
    if applied_count_value == 0:
        attention_mask = (input_ids != pad_token_id).long()
        attention_mask[:, 0] = 1
        return (
            input_ids,
            attention_mask,
            identity_indices,
            selected_count,
            candidate_count_tensor,
            applied_count,
        )

    required_width = max(
        int(target_end[batch_idx].item())
        + int(chosen_insertions[batch_idx] is not None)
        for batch_idx in range(batch_size)
    )
    if text_bucket_size is not None:
        required_width = (
            (required_width + text_bucket_size - 1) // text_bucket_size
        ) * text_bucket_size
    augmented_width = max(original_width, required_width)
    augmented = input_ids.new_full(
        (batch_size, augmented_width), pad_token_id
    )
    state_indices = identity_indices.clone()

    for batch_idx, insertion in enumerate(chosen_insertions):
        sequence_end = int(target_end[batch_idx].item())
        if insertion is None:
            augmented[batch_idx, :sequence_end] = input_ids[
                batch_idx, :sequence_end
            ]
            continue

        reference_boundary, insertion_token = insertion
        insertion_position = int(target_start[batch_idx].item()) + reference_boundary
        augmented[batch_idx, :insertion_position] = input_ids[
            batch_idx, :insertion_position
        ]
        augmented[batch_idx, insertion_position] = insertion_token
        augmented[batch_idx, insertion_position + 1 : sequence_end + 1] = input_ids[
            batch_idx, insertion_position:sequence_end
        ]

        # The clean state immediately before this boundary is replaced by the
        # state after the inserted token. Later clean states shift right by one.
        state_indices[batch_idx, insertion_position - 1] = insertion_position
        if insertion_position < sequence_end:
            clean_positions = torch.arange(
                insertion_position,
                sequence_end,
                device=input_ids.device,
            )
            state_indices[batch_idx, clean_positions] = clean_positions + 1
        state_indices[batch_idx].clamp_(max=augmented_width - 1)

    attention_mask = (augmented != pad_token_id).long()
    attention_mask[:, 0] = 1
    return (
        augmented,
        attention_mask,
        state_indices,
        selected_count,
        candidate_count_tensor,
        applied_count,
    )
