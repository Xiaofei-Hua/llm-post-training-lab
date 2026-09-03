from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from posttrain_lab.train import (
    BudgetStateError,
    LossTokenBudget,
    MaskValidationError,
    commit_torch_loss_budget,
    plan_torch_loss_budget,
    torch_assistant_target_loss_mask,
    torch_completion_loss_mask,
    torch_exclude_zero_variance_grpo_groups,
    torch_intersect_masks,
)


def completion_mask_oracle(
    token_ids: list[int],
    *,
    completion_start: int,
    attention_mask: list[int],
    eos_token_id: int,
    include_eos: bool,
) -> list[bool]:
    result = [False] * len(token_ids)
    saw_eos = False
    for token_index in range(completion_start, len(token_ids)):
        if not attention_mask[token_index] or saw_eos:
            continue
        is_eos = token_ids[token_index] == eos_token_id
        result[token_index] = include_eos or not is_eos
        saw_eos = is_eos
    return result


def assistant_mask_oracle(
    labels: list[int],
    *,
    attention_mask: list[int],
    ignore_index: int,
    eos_token_id: int,
    include_eos: bool,
) -> list[bool]:
    result = []
    saw_eos = False
    for label, attended in zip(labels, attention_mask, strict=True):
        enabled = bool(attended) and label != ignore_index and not saw_eos
        is_eos = enabled and label == eos_token_id
        result.append(enabled and (include_eos or not is_eos))
        saw_eos = saw_eos or is_eos
    return result


@st.composite
def completion_batches(draw: st.DrawFn) -> tuple[list[list[int]], list[list[int]], list[int], bool]:
    batch_size = draw(st.integers(min_value=1, max_value=5))
    sequence_length = draw(st.integers(min_value=1, max_value=24))
    include_eos = draw(st.booleans())
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    completion_starts: list[int] = []

    for _ in range(batch_size):
        completion_start = draw(st.integers(min_value=0, max_value=sequence_length))
        active_end = draw(st.integers(min_value=completion_start, max_value=sequence_length))
        prefix_attention = draw(
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=completion_start,
                max_size=completion_start,
            )
        )
        row = draw(
            st.lists(
                st.integers(min_value=0, max_value=15),
                min_size=sequence_length,
                max_size=sequence_length,
            )
        )
        input_ids.append(row)
        attention_masks.append(
            prefix_attention
            + [1] * (active_end - completion_start)
            + [0] * (sequence_length - active_end)
        )
        completion_starts.append(completion_start)

    return input_ids, attention_masks, completion_starts, include_eos


@st.composite
def assistant_batches(draw: st.DrawFn) -> tuple[list[list[int]], list[list[int]], bool]:
    batch_size = draw(st.integers(min_value=1, max_value=5))
    sequence_length = draw(st.integers(min_value=1, max_value=24))
    labels = [
        draw(
            st.lists(
                st.integers(min_value=-2, max_value=8),
                min_size=sequence_length,
                max_size=sequence_length,
            )
        )
        for _ in range(batch_size)
    ]
    attention = [
        draw(
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=sequence_length,
                max_size=sequence_length,
            )
        )
        for _ in range(batch_size)
    ]
    return labels, attention, draw(st.booleans())


@settings(max_examples=100, deadline=None)
@given(completion_batches())
def test_batched_completion_mask_matches_scalar_oracle(
    case: tuple[list[list[int]], list[list[int]], list[int], bool],
) -> None:
    input_ids, attention_masks, completion_starts, include_eos = case
    actual = torch_completion_loss_mask(
        torch.tensor(input_ids, dtype=torch.int64),
        completion_start=torch.tensor(completion_starts, dtype=torch.int64),
        attention_mask=torch.tensor(attention_masks, dtype=torch.int64),
        eos_token_id=2,
        include_eos=include_eos,
    )
    expected = [
        completion_mask_oracle(
            row,
            completion_start=start,
            attention_mask=attention,
            eos_token_id=2,
            include_eos=include_eos,
        )
        for row, start, attention in zip(input_ids, completion_starts, attention_masks, strict=True)
    ]
    assert actual.dtype == torch.bool
    assert actual.tolist() == expected


@settings(max_examples=100, deadline=None)
@given(assistant_batches())
def test_batched_assistant_mask_matches_scalar_oracle(
    case: tuple[list[list[int]], list[list[int]], bool],
) -> None:
    labels, attention, include_eos = case
    actual = torch_assistant_target_loss_mask(
        torch.tensor(labels, dtype=torch.int64),
        attention_mask=torch.tensor(attention, dtype=torch.int64),
        ignore_index=-1,
        eos_token_id=2,
        include_eos=include_eos,
    )
    expected = [
        assistant_mask_oracle(
            row,
            attention_mask=row_attention,
            ignore_index=-1,
            eos_token_id=2,
            include_eos=include_eos,
        )
        for row, row_attention in zip(labels, attention, strict=True)
    ]
    assert actual.tolist() == expected


def test_completion_mask_rejects_non_contiguous_attention() -> None:
    with pytest.raises(MaskValidationError, match="right-padded"):
        torch_completion_loss_mask(
            torch.tensor([[10, 20, 0, 21]], dtype=torch.int64),
            completion_start=1,
            attention_mask=torch.tensor([[1, 1, 0, 1]], dtype=torch.int64),
        )


def test_tensor_masks_require_binary_integer_values() -> None:
    with pytest.raises(MaskValidationError, match="bool or integer"):
        torch_intersect_masks(torch.tensor([[1.0, 0.0]]))
    with pytest.raises(MaskValidationError, match="outside 0/1"):
        torch_intersect_masks(torch.tensor([[1, 2]], dtype=torch.int64))


def test_grpo_group_filter_runs_on_tensor_batch() -> None:
    masks = torch.tensor(
        [[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]],
        dtype=torch.bool,
    )
    result = torch_exclude_zero_variance_grpo_groups(
        masks,
        group_ids=torch.tensor([10, 10, 20, 20, 30, 30]),
        rewards=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    assert result.active_group_ids.tolist() == [20]
    assert result.skipped_group_ids.tolist() == [10, 30]
    assert result.active_row_mask.tolist() == [False, False, True, True, False, False]
    assert result.masks.tolist() == [
        [False, False, False],
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [False, False, False],
        [False, False, False],
    ]
    assert result.active_token_count == 3


def test_grpo_group_filter_rejects_non_finite_rewards() -> None:
    with pytest.raises(MaskValidationError, match="finite"):
        torch_exclude_zero_variance_grpo_groups(
            torch.ones((2, 2), dtype=torch.bool),
            group_ids=torch.tensor([0, 0]),
            rewards=torch.tensor([0.0, torch.nan]),
        )


def test_torch_budget_closes_exactly_and_preserves_device() -> None:
    budget = LossTokenBudget(target_tokens=5, consumed_tokens=2, version=3)
    objective_mask = torch.tensor(
        [[False, True, True, True], [False, True, True, True]], dtype=torch.bool
    )
    selection = plan_torch_loss_budget(
        budget,
        objective_mask,
        sample_ids=["sample-b", "sample-a"],
        generation_indices=torch.tensor([0, 0]),
    )
    assert selection.loss_mask.device == objective_mask.device
    assert selection.loss_mask.tolist() == [
        [False, False, False, False],
        [False, True, True, True],
    ]
    assert selection.candidate_tokens == 6
    assert selection.selected_tokens == 3
    assert selection.truncated

    record = commit_torch_loss_budget(
        budget,
        selection,
        optimizer_step_executed=True,
        objective="opd",
        step_id="stage-1-final",
    )
    assert record.complete
    assert record.consumed_after == 5
    assert budget.remaining_tokens == 0


def test_torch_budget_selection_is_independent_of_batch_order() -> None:
    first = plan_torch_loss_budget(
        LossTokenBudget(2),
        torch.tensor([[True, True, True], [True, True, False]]),
        sample_ids=["sample-b", "sample-a"],
        generation_indices=[0, 0],
    )
    second = plan_torch_loss_budget(
        LossTokenBudget(2),
        torch.tensor([[True, True, False], [True, True, True]]),
        sample_ids=["sample-a", "sample-b"],
        generation_indices=[0, 0],
    )
    assert first.selection_id == second.selection_id
    assert first.loss_mask.tolist() == [[False, False, False], [True, True, False]]
    assert second.loss_mask.tolist() == [[True, True, False], [False, False, False]]


def test_torch_budget_digest_binds_unselected_candidates() -> None:
    first = plan_torch_loss_budget(
        LossTokenBudget(1),
        torch.tensor([[True], [True]]),
        sample_ids=["sample-a", "sample-b"],
        generation_indices=[0, 0],
    )
    second = plan_torch_loss_budget(
        LossTokenBudget(1),
        torch.tensor([[True], [True]]),
        sample_ids=["sample-a", "sample-c"],
        generation_indices=[0, 0],
    )
    assert first.loss_mask.tolist() == second.loss_mask.tolist()
    assert first.selection_id != second.selection_id


def test_torch_budget_failed_step_does_not_advance_and_can_retry() -> None:
    budget = LossTokenBudget(target_tokens=2)
    selection = plan_torch_loss_budget(
        budget,
        torch.tensor([[True, True]]),
        sample_ids=["sample-a"],
        generation_indices=[0],
    )
    failed = commit_torch_loss_budget(
        budget,
        selection,
        optimizer_step_executed=False,
        objective="sft",
        step_id="update-0-attempt-0",
    )
    assert failed.counted_tokens == 0
    assert budget.consumed_tokens == 0

    succeeded = commit_torch_loss_budget(
        budget,
        selection,
        optimizer_step_executed=True,
        objective="sft",
        step_id="update-0-attempt-1",
    )
    assert succeeded.counted_tokens == 2
    assert budget.complete


def test_torch_budget_detects_in_place_mask_mutation() -> None:
    budget = LossTokenBudget(target_tokens=1)
    selection = plan_torch_loss_budget(
        budget,
        torch.tensor([[True, True]]),
        sample_ids=["sample-a"],
        generation_indices=[0],
    )
    selection.loss_mask[0, 0] = False
    selection.loss_mask[0, 1] = True
    with pytest.raises(BudgetStateError, match="canonical budget prefix"):
        commit_torch_loss_budget(
            budget,
            selection,
            optimizer_step_executed=True,
            objective="sft",
            step_id="mutated-mask",
        )


def test_torch_budget_rejects_forged_accounting() -> None:
    budget = LossTokenBudget(target_tokens=2)
    selection = plan_torch_loss_budget(
        budget,
        torch.tensor([[True, True]]),
        sample_ids=["sample-a"],
        generation_indices=[0],
    )
    forged = replace(
        selection,
        reservation=replace(selection.reservation, selected_tokens=1),
    )
    with pytest.raises(BudgetStateError):
        commit_torch_loss_budget(
            budget,
            forged,
            optimizer_step_executed=True,
            objective="sft",
            step_id="forged-accounting",
        )


@settings(max_examples=75, deadline=None)
@given(
    target=st.integers(min_value=1, max_value=200),
    candidate_sizes=st.lists(st.integers(min_value=1, max_value=32), min_size=1, max_size=20),
)
def test_tensor_budget_property_never_overshoots(
    target: int,
    candidate_sizes: list[int],
) -> None:
    budget = LossTokenBudget(target_tokens=target)
    update = 0
    while not budget.complete:
        candidate_size = candidate_sizes[update % len(candidate_sizes)]
        selection = plan_torch_loss_budget(
            budget,
            torch.ones((1, candidate_size), dtype=torch.bool),
            sample_ids=[f"sample-{update}"],
            generation_indices=[0],
        )
        before = budget.consumed_tokens
        record = commit_torch_loss_budget(
            budget,
            selection,
            optimizer_step_executed=True,
            objective="property-check",
            step_id=f"update-{update}",
        )
        assert record.counted_tokens == min(candidate_size, target - before)
        assert budget.consumed_tokens <= target
        update += 1
    assert budget.consumed_tokens == target
