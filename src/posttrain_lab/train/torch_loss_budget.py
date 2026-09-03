"""Batched PyTorch implementation of loss masks and exact token budgets.

The functions in this module are the tensor-facing API for trainer adapters.
They preserve device placement, reject ambiguous mask metadata, and match the
scalar semantics in :mod:`posttrain_lab.train.loss_budget`.

No model forward or optimizer is performed here.  A caller plans one complete
logical optimizer update, applies the returned Boolean mask to its objective,
and commits only after it knows whether the optimizer step executed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .loss_budget import (
    BudgetReservation,
    BudgetStateError,
    BudgetStepRecord,
    LossTokenBudget,
    MaskValidationError,
)

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_integer_tensor(value: Tensor, *, name: str, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise MaskValidationError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise MaskValidationError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    if value.dtype not in _INTEGER_DTYPES:
        raise MaskValidationError(f"{name} must have an integer dtype, got {value.dtype}")


def _binary_mask(
    value: Tensor,
    *,
    name: str,
    shape: torch.Size,
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise MaskValidationError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise MaskValidationError(
            f"{name} shape {tuple(value.shape)} does not match expected {tuple(shape)}"
        )
    if value.device != device:
        raise MaskValidationError(
            f"{name} is on {value.device}, expected the input device {device}"
        )
    if value.dtype == torch.bool:
        return value
    if value.dtype not in _INTEGER_DTYPES:
        raise MaskValidationError(f"{name} must have bool or integer 0/1 dtype")
    if bool(torch.any((value != 0) & (value != 1)).item()):
        raise MaskValidationError(f"{name} contains values outside 0/1")
    return value.to(dtype=torch.bool)


def _attention_mask(input_tensor: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return torch.ones_like(input_tensor, dtype=torch.bool)
    return _binary_mask(
        attention_mask,
        name="attention_mask",
        shape=input_tensor.shape,
        device=input_tensor.device,
    )


def _validate_optional_token_id(token_id: int | None, *, name: str) -> None:
    if token_id is not None and (
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
    ):
        raise MaskValidationError(f"{name} must be a non-negative integer or None")


def _completion_starts(
    completion_start: int | Sequence[int] | Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    if isinstance(completion_start, bool):
        raise MaskValidationError("completion_start must contain integer indices")
    if isinstance(completion_start, int):
        starts = torch.full((batch_size,), completion_start, dtype=torch.int64, device=device)
    elif isinstance(completion_start, Tensor):
        if completion_start.device != device:
            raise MaskValidationError(
                "completion_start tensor must be on the same device as input_ids"
            )
        if completion_start.dtype not in _INTEGER_DTYPES:
            raise MaskValidationError("completion_start tensor must have an integer dtype")
        if completion_start.ndim == 0:
            starts = completion_start.to(dtype=torch.int64).expand(batch_size)
        elif completion_start.ndim == 1 and completion_start.shape[0] == batch_size:
            starts = completion_start.to(dtype=torch.int64)
        else:
            raise MaskValidationError(
                "completion_start tensor must be scalar or have shape [batch]"
            )
    else:
        try:
            starts = torch.as_tensor(completion_start, device=device)
        except (TypeError, ValueError) as error:
            raise MaskValidationError(
                "completion_start must be an int or integer sequence"
            ) from error
        if starts.dtype not in _INTEGER_DTYPES or starts.ndim != 1:
            raise MaskValidationError(
                "completion_start sequence must contain one integer per batch row"
            )
        if starts.shape[0] != batch_size:
            raise MaskValidationError(
                f"completion_start has {starts.shape[0]} rows, expected {batch_size}"
            )
        starts = starts.to(dtype=torch.int64)

    if bool(torch.any((starts < 0) | (starts > sequence_length)).item()):
        raise MaskValidationError(f"completion_start values must be in [0, {sequence_length}]")
    return starts


def _through_first_eos_mask(
    base_mask: Tensor,
    token_ids: Tensor,
    *,
    eos_token_id: int | None,
    include_eos: bool,
) -> Tensor:
    if not isinstance(include_eos, bool):
        raise MaskValidationError("include_eos must be bool")
    _validate_optional_token_id(eos_token_id, name="eos_token_id")
    if eos_token_id is None:
        return base_mask

    eos_positions = base_mask & token_ids.eq(eos_token_id)
    eos_count = eos_positions.to(dtype=torch.int64).cumsum(dim=1)
    before_first_eos = eos_count.eq(0)
    if include_eos:
        first_eos = eos_positions & eos_count.eq(1)
        return base_mask & (before_first_eos | first_eos)
    return base_mask & before_first_eos


def torch_completion_loss_mask(
    input_ids: Tensor,
    *,
    completion_start: int | Sequence[int] | Tensor,
    attention_mask: Tensor | None = None,
    eos_token_id: int | None = None,
    include_eos: bool = True,
) -> Tensor:
    """Return a batched completion-through-first-EOS Boolean mask.

    ``input_ids`` must be ``[batch, sequence]``.  ``completion_start`` uses
    absolute padded-sequence indices and may be a scalar or one value per row.
    Left padding before the start is allowed; attention after completion
    padding begins is rejected.
    """

    _require_integer_tensor(input_ids, name="input_ids", ndim=2)
    active = _attention_mask(input_ids, attention_mask)
    batch_size, sequence_length = input_ids.shape
    starts = _completion_starts(
        completion_start,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=input_ids.device,
    )
    positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
    completion_region = positions >= starts.unsqueeze(1)

    padding_seen = (completion_region & ~active).to(dtype=torch.int64).cumsum(dim=1).gt(0)
    if bool(torch.any(completion_region & active & padding_seen).item()):
        raise MaskValidationError("attention_mask must be right-padded after each completion_start")

    base_mask = completion_region & active
    return _through_first_eos_mask(
        base_mask,
        input_ids,
        eos_token_id=eos_token_id,
        include_eos=include_eos,
    )


def torch_assistant_target_loss_mask(
    labels: Tensor,
    *,
    attention_mask: Tensor | None = None,
    ignore_index: int = -100,
    eos_token_id: int | None = None,
    include_eos: bool = True,
) -> Tensor:
    """Return the batched SFT target mask for one assistant completion per row."""

    _require_integer_tensor(labels, name="labels", ndim=2)
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise MaskValidationError("ignore_index must be an integer")
    active = _attention_mask(labels, attention_mask)
    base_mask = active & labels.ne(ignore_index)
    return _through_first_eos_mask(
        base_mask,
        labels,
        eos_token_id=eos_token_id,
        include_eos=include_eos,
    )


def torch_intersect_masks(*masks: Tensor) -> Tensor:
    """Strictly intersect batched masks without changing their device."""

    if not masks:
        raise MaskValidationError("at least one mask is required")
    first = masks[0]
    if not isinstance(first, Tensor) or first.ndim != 2:
        raise MaskValidationError("mask[0] must be a rank-2 torch.Tensor")
    normalized = [
        _binary_mask(
            mask,
            name=f"mask[{index}]",
            shape=first.shape,
            device=first.device,
        )
        for index, mask in enumerate(masks)
    ]
    result = normalized[0].clone()
    for mask in normalized[1:]:
        result = result & mask
    return result


@dataclass(frozen=True)
class TorchGroupMaskResult:
    """Tensor result after exact zero-reward-variance group filtering."""

    masks: Tensor
    active_row_mask: Tensor
    active_group_ids: Tensor
    skipped_group_ids: Tensor

    @property
    def active_token_count(self) -> int:
        return int(self.masks.sum(dtype=torch.int64).item())


def torch_exclude_zero_variance_grpo_groups(
    masks: Tensor,
    *,
    group_ids: Tensor,
    rewards: Tensor,
) -> TorchGroupMaskResult:
    """Remove every row in groups whose rewards are exactly identical.

    ``group_ids`` are non-negative integer prompt/group identifiers.  Rewards
    may be integer or floating point but must be finite.  Exact equality is
    intentional for the project's exact/symbolic 0/1 verifier reward.
    """

    if not isinstance(masks, Tensor) or masks.ndim != 2:
        raise MaskValidationError("masks must be a rank-2 torch.Tensor")
    normalized_masks = _binary_mask(
        masks,
        name="masks",
        shape=masks.shape,
        device=masks.device,
    )
    _require_integer_tensor(group_ids, name="group_ids", ndim=1)
    if group_ids.device != masks.device:
        raise MaskValidationError("group_ids and masks must be on the same device")
    if group_ids.shape[0] != masks.shape[0]:
        raise MaskValidationError("group_ids must contain one value per mask row")
    if bool(torch.any(group_ids < 0).item()):
        raise MaskValidationError("group_ids must be non-negative")

    if not isinstance(rewards, Tensor) or rewards.ndim != 1:
        raise MaskValidationError("rewards must be a rank-1 torch.Tensor")
    if rewards.device != masks.device:
        raise MaskValidationError("rewards and masks must be on the same device")
    if rewards.shape[0] != masks.shape[0]:
        raise MaskValidationError("rewards must contain one value per mask row")
    if rewards.dtype == torch.bool or not (
        rewards.dtype.is_floating_point or rewards.dtype in _INTEGER_DTYPES
    ):
        raise MaskValidationError("rewards must have a real numeric dtype")
    if rewards.dtype.is_floating_point and not bool(torch.isfinite(rewards).all().item()):
        raise MaskValidationError("rewards must be finite")

    if group_ids.numel() == 0:
        empty_rows = torch.empty((0,), dtype=torch.bool, device=masks.device)
        empty_groups = group_ids.clone()
        return TorchGroupMaskResult(
            masks=normalized_masks.clone(),
            active_row_mask=empty_rows,
            active_group_ids=empty_groups,
            skipped_group_ids=empty_groups.clone(),
        )

    unique_group_ids, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    group_count = unique_group_ids.shape[0]
    if rewards.dtype.is_floating_point:
        lower_fill = torch.inf
        upper_fill = -torch.inf
    else:
        dtype_info = torch.iinfo(rewards.dtype)
        lower_fill = dtype_info.max
        upper_fill = dtype_info.min

    group_min = torch.full((group_count,), lower_fill, dtype=rewards.dtype, device=rewards.device)
    group_max = torch.full((group_count,), upper_fill, dtype=rewards.dtype, device=rewards.device)
    group_min.scatter_reduce_(0, inverse, rewards, reduce="amin", include_self=True)
    group_max.scatter_reduce_(0, inverse, rewards, reduce="amax", include_self=True)
    active_groups = group_min.ne(group_max)
    active_rows = active_groups[inverse]

    return TorchGroupMaskResult(
        masks=normalized_masks & active_rows.unsqueeze(1),
        active_row_mask=active_rows,
        active_group_ids=unique_group_ids[active_groups],
        skipped_group_ids=unique_group_ids[~active_groups],
    )


@dataclass(frozen=True)
class TorchBudgetSelection:
    """Candidate and selected masks paired with an immutable reservation."""

    candidate_mask: Tensor
    loss_mask: Tensor
    row_keys: tuple[tuple[str, int], ...]
    reservation: BudgetReservation

    @property
    def selection_id(self) -> str:
        return self.reservation.selection_id

    @property
    def selected_tokens(self) -> int:
        return self.reservation.selected_tokens

    @property
    def candidate_tokens(self) -> int:
        return self.reservation.candidate_tokens

    @property
    def truncated(self) -> bool:
        return self.reservation.truncated


def _generation_index_list(
    generation_indices: Sequence[int] | Tensor,
    *,
    batch_size: int,
) -> list[int]:
    if isinstance(generation_indices, Tensor):
        _require_integer_tensor(generation_indices, name="generation_indices", ndim=1)
        if generation_indices.shape[0] != batch_size:
            raise MaskValidationError("generation_indices must contain one value per mask row")
        values = generation_indices.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        values = list(generation_indices)
    if len(values) != batch_size:
        raise MaskValidationError("generation_indices must contain one value per mask row")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise MaskValidationError("generation_indices must be non-negative integers")
    return values


def _row_keys(
    sample_ids: Sequence[str],
    generation_indices: Sequence[int],
    *,
    batch_size: int,
) -> tuple[tuple[str, int], ...]:
    if len(sample_ids) != batch_size:
        raise MaskValidationError("sample_ids must contain one value per mask row")
    keys = []
    for index, (sample_id, generation_index) in enumerate(
        zip(sample_ids, generation_indices, strict=True)
    ):
        if not isinstance(sample_id, str) or not sample_id:
            raise MaskValidationError(f"sample_ids[{index}] must be a non-empty string")
        keys.append((sample_id, generation_index))
    if len(keys) != len(set(keys)):
        raise MaskValidationError(
            "(sample_id, generation_index) pairs must be unique within an update"
        )
    return tuple(keys)


def _stable_row_order(row_keys: Sequence[tuple[str, int]], *, device: torch.device) -> Tensor:
    order = sorted(range(len(row_keys)), key=row_keys.__getitem__)
    return torch.tensor(order, dtype=torch.int64, device=device)


def _select_canonical_prefix(
    candidate_mask: Tensor,
    *,
    row_keys: Sequence[tuple[str, int]],
    selected_tokens: int,
) -> Tensor:
    candidate_tokens = int(candidate_mask.sum(dtype=torch.int64).item())
    if selected_tokens < 0 or selected_tokens > candidate_tokens:
        raise BudgetStateError("selected token count is outside the candidate mask")
    if selected_tokens == candidate_tokens:
        return candidate_mask.clone()
    if selected_tokens == 0:
        return torch.zeros_like(candidate_mask)

    row_order = _stable_row_order(row_keys, device=candidate_mask.device)
    ordered = candidate_mask.index_select(0, row_order)
    flat = ordered.reshape(-1)
    prefix = flat & flat.to(dtype=torch.int64).cumsum(dim=0).le(selected_tokens)
    selected_ordered = prefix.reshape_as(ordered)
    selected = torch.empty_like(candidate_mask)
    selected.index_copy_(0, row_order, selected_ordered)
    return selected


def _mask_bytes(mask: Tensor) -> bytes:
    return mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes(order="C")


def _selection_content_digest(
    candidate_mask: Tensor,
    selected_mask: Tensor,
    *,
    row_keys: Sequence[tuple[str, int]],
) -> str:
    row_order = _stable_row_order(row_keys, device=candidate_mask.device)
    canonical_candidate = candidate_mask.index_select(0, row_order)
    canonical_selected = selected_mask.index_select(0, row_order)
    canonical_keys = [row_keys[index] for index in row_order.to(device="cpu").tolist()]
    metadata = {
        "schema_version": 1,
        "shape": list(candidate_mask.shape),
        "row_keys": [list(key) for key in canonical_keys],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    )
    digest.update(b"\x00candidate-mask\x00")
    digest.update(_mask_bytes(canonical_candidate))
    digest.update(b"\x00selected-mask\x00")
    digest.update(_mask_bytes(canonical_selected))
    return digest.hexdigest()


def plan_torch_loss_budget(
    budget: LossTokenBudget,
    objective_mask: Tensor,
    *,
    sample_ids: Sequence[str],
    generation_indices: Sequence[int] | Tensor,
) -> TorchBudgetSelection:
    """Apply the exact global budget to one logical optimizer update.

    Trainer integrations must call this once for the globally aggregated
    logical update, then scatter the resulting row masks to ranks/microbatches.
    Selection is vectorized; only compact mask bytes and row metadata cross to
    the host to build the audit digest.
    """

    if not isinstance(budget, LossTokenBudget):
        raise BudgetStateError("budget must be a LossTokenBudget")
    if not isinstance(objective_mask, Tensor) or objective_mask.ndim != 2:
        raise MaskValidationError("objective_mask must be a rank-2 torch.Tensor")
    normalized_mask = _binary_mask(
        objective_mask,
        name="objective_mask",
        shape=objective_mask.shape,
        device=objective_mask.device,
    )
    batch_size = normalized_mask.shape[0]
    normalized_generation_indices = _generation_index_list(
        generation_indices,
        batch_size=batch_size,
    )
    row_keys = _row_keys(
        sample_ids,
        normalized_generation_indices,
        batch_size=batch_size,
    )
    candidate_mask = normalized_mask.detach().clone()
    candidate_tokens = int(candidate_mask.sum(dtype=torch.int64).item())
    selected_tokens = min(candidate_tokens, budget.remaining_tokens)
    selected_mask = _select_canonical_prefix(
        candidate_mask,
        row_keys=row_keys,
        selected_tokens=selected_tokens,
    )
    content_digest = _selection_content_digest(
        candidate_mask,
        selected_mask,
        row_keys=row_keys,
    )
    reservation = budget.reserve(
        candidate_tokens,
        content_digest=content_digest,
    )
    if reservation.selected_tokens != int(selected_mask.sum(dtype=torch.int64).item()):
        raise BudgetStateError("selected mask and budget reservation disagree")
    return TorchBudgetSelection(
        candidate_mask=candidate_mask,
        loss_mask=selected_mask,
        row_keys=row_keys,
        reservation=reservation,
    )


def commit_torch_loss_budget(
    budget: LossTokenBudget,
    selection: TorchBudgetSelection,
    *,
    optimizer_step_executed: bool,
    objective: str,
    step_id: str,
) -> BudgetStepRecord:
    """Verify the applied tensor mask and commit its accounting transaction."""

    if not isinstance(budget, LossTokenBudget):
        raise BudgetStateError("budget must be a LossTokenBudget")
    if not isinstance(selection, TorchBudgetSelection):
        raise BudgetStateError("selection must be a TorchBudgetSelection")
    if (
        not isinstance(selection.candidate_mask, Tensor)
        or selection.candidate_mask.ndim != 2
        or selection.candidate_mask.dtype != torch.bool
    ):
        raise BudgetStateError("candidate mask must remain a rank-2 Boolean tensor")
    if (
        not isinstance(selection.loss_mask, Tensor)
        or selection.loss_mask.shape != selection.candidate_mask.shape
        or selection.loss_mask.device != selection.candidate_mask.device
        or selection.loss_mask.dtype != torch.bool
    ):
        raise BudgetStateError("loss mask no longer matches candidate mask metadata")
    if not isinstance(selection.row_keys, tuple):
        raise BudgetStateError("selection row keys must remain an immutable tuple")
    sample_ids = []
    generation_indices = []
    for key in selection.row_keys:
        if not isinstance(key, tuple) or len(key) != 2:
            raise BudgetStateError("selection contains malformed row keys")
        sample_ids.append(key[0])
        generation_indices.append(key[1])
    try:
        row_keys = _row_keys(
            sample_ids,
            generation_indices,
            batch_size=selection.candidate_mask.shape[0],
        )
    except MaskValidationError as error:
        raise BudgetStateError("selection contains invalid row keys") from error
    expected_mask = _select_canonical_prefix(
        selection.candidate_mask,
        row_keys=row_keys,
        selected_tokens=selection.reservation.selected_tokens,
    )
    if not torch.equal(selection.loss_mask.detach(), expected_mask):
        raise BudgetStateError("loss mask is not the canonical budget prefix")
    content_digest = _selection_content_digest(
        selection.candidate_mask,
        selection.loss_mask,
        row_keys=row_keys,
    )
    if content_digest != selection.reservation.content_digest:
        raise BudgetStateError("tensor selection no longer matches its reservation digest")
    return budget.commit(
        selection.reservation,
        optimizer_step_executed=optimizer_step_executed,
        objective=objective,
        step_id=step_id,
    )
