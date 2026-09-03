from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from posttrain_lab.train import BudgetStateError, LossTokenBudget


def content_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("target", "consumed", "version"),
    [
        (0, 0, 0),
        (True, 0, 0),
        (4, -1, 0),
        (4, 5, 0),
        (4, 0, -1),
    ],
)
def test_counter_rejects_invalid_state(target: int, consumed: int, version: int) -> None:
    with pytest.raises(ValueError):
        LossTokenBudget(target, consumed_tokens=consumed, version=version)


def test_reservation_closes_remaining_budget_without_overshoot() -> None:
    budget = LossTokenBudget(5, consumed_tokens=3, version=7)
    reservation = budget.reserve(8, content_digest=content_digest("final-update"))
    assert reservation.candidate_tokens == 8
    assert reservation.selected_tokens == 2
    assert reservation.truncated

    record = budget.commit(
        reservation,
        optimizer_step_executed=True,
        objective="opd",
        step_id="stage-final",
    )
    assert record.complete
    assert record.counted_tokens == 2
    assert budget.consumed_tokens == 5
    assert budget.remaining_tokens == 0


def test_failed_step_does_not_advance_and_same_reservation_can_retry() -> None:
    budget = LossTokenBudget(4)
    reservation = budget.reserve(2, content_digest=content_digest("update-0"))
    failed = budget.commit(
        reservation,
        optimizer_step_executed=False,
        objective="sft",
        step_id="update-0-attempt-0",
    )
    assert failed.counted_tokens == 0
    assert budget.state_dict()["consumed_tokens"] == 0

    succeeded = budget.commit(
        reservation,
        optimizer_step_executed=True,
        objective="sft",
        step_id="update-0-attempt-1",
    )
    assert succeeded.counted_tokens == 2
    assert budget.consumed_tokens == 2


def test_executed_update_invalidates_other_reservations_from_old_state() -> None:
    budget = LossTokenBudget(4)
    stale = budget.reserve(1, content_digest=content_digest("stale"))
    current = budget.reserve(1, content_digest=content_digest("current"))
    budget.commit(
        current,
        optimizer_step_executed=True,
        objective="grpo",
        step_id="current",
    )
    with pytest.raises(BudgetStateError, match="stale"):
        budget.commit(
            stale,
            optimizer_step_executed=True,
            objective="grpo",
            step_id="stale",
        )


def test_canonical_selected_count_cannot_be_forged() -> None:
    budget = LossTokenBudget(4)
    reservation = budget.reserve(3, content_digest=content_digest("update"))
    forged = replace(reservation, selected_tokens=2)
    with pytest.raises(BudgetStateError, match="canonical"):
        budget.commit(
            forged,
            optimizer_step_executed=True,
            objective="sft",
            step_id="forged",
        )


def test_content_digest_and_selection_id_are_bound() -> None:
    budget = LossTokenBudget(4)
    first = budget.reserve(2, content_digest=content_digest("batch-a"))
    repeated = budget.reserve(2, content_digest=content_digest("batch-a"))
    different = budget.reserve(2, content_digest=content_digest("batch-b"))
    assert first.selection_id == repeated.selection_id
    assert first.selection_id != different.selection_id

    forged = replace(first, content_digest=content_digest("batch-b"))
    with pytest.raises(BudgetStateError, match="selection_id"):
        budget.commit(
            forged,
            optimizer_step_executed=True,
            objective="sft",
            step_id="forged-digest",
        )


def test_zero_token_update_can_be_logged_but_not_executed() -> None:
    budget = LossTokenBudget(2)
    reservation = budget.reserve(0, content_digest=content_digest("empty"))
    record = budget.commit(
        reservation,
        optimizer_step_executed=False,
        objective="grpo",
        step_id="zero-variance-batch",
    )
    assert record.counted_tokens == 0
    with pytest.raises(BudgetStateError, match="zero selected"):
        budget.commit(
            reservation,
            optimizer_step_executed=True,
            objective="grpo",
            step_id="invalid-empty-step",
        )


def test_state_round_trip_is_schema_versioned() -> None:
    budget = LossTokenBudget(3)
    reservation = budget.reserve(2, content_digest=content_digest("update"))
    record = budget.commit(
        reservation,
        optimizer_step_executed=True,
        objective="opd",
        step_id="update-1",
    )
    restored = LossTokenBudget.from_state_dict(budget.state_dict())
    assert restored.state_dict() == budget.state_dict()
    assert json.loads(json.dumps(record.to_dict()))["counted_tokens"] == 2

    invalid_schema = {**budget.state_dict(), "schema_version": 999}
    with pytest.raises(ValueError, match="unsupported"):
        LossTokenBudget.from_state_dict(invalid_schema)


def test_state_schema_rejects_missing_or_extra_fields() -> None:
    with pytest.raises(ValueError, match="invalid state keys"):
        LossTokenBudget.from_state_dict(
            {
                "schema_version": 1,
                "target_tokens": 3,
                "consumed_tokens": 0,
                "version": 0,
                "extra": 1,
            }
        )


def test_commit_metadata_is_strictly_validated() -> None:
    budget = LossTokenBudget(2)
    reservation = budget.reserve(1, content_digest=content_digest("update"))
    with pytest.raises(ValueError, match="must be bool"):
        budget.commit(
            reservation,
            optimizer_step_executed=1,  # type: ignore[arg-type]
            objective="sft",
            step_id="invalid-flag",
        )
    with pytest.raises(ValueError, match="objective"):
        budget.commit(
            reservation,
            optimizer_step_executed=False,
            objective=" ",
            step_id="invalid-objective",
        )
