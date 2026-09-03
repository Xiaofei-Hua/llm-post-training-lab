"""Transactional accounting for exact Student backward-token budgets.

The counter is intentionally host-side: it owns stage state, checkpoint
serialization, retry semantics, and immutable reservations. Tensor selection
and mask validation live in ``torch_loss_budget``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass


class MaskValidationError(ValueError):
    """Raised when tensor metadata cannot define an unambiguous loss mask."""


class BudgetStateError(RuntimeError):
    """Raised when a reservation is malformed or committed against stale state."""


def _is_non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class BudgetReservation:
    """Immutable token reservation tied to one counter version and mask digest."""

    candidate_tokens: int
    selected_tokens: int
    consumed_before: int
    target_tokens: int
    counter_version: int
    content_digest: str
    selection_id: str

    @property
    def truncated(self) -> bool:
        return self.selected_tokens < self.candidate_tokens


@dataclass(frozen=True)
class BudgetStepRecord:
    """Serializable accounting record for one attempted optimizer update."""

    step_id: str
    objective: str
    selection_id: str
    optimizer_step_executed: bool
    candidate_tokens: int
    selected_tokens: int
    counted_tokens: int
    consumed_before: int
    consumed_after: int
    target_tokens: int
    counter_version_after: int

    @property
    def complete(self) -> bool:
        return self.consumed_after == self.target_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "objective": self.objective,
            "selection_id": self.selection_id,
            "optimizer_step_executed": self.optimizer_step_executed,
            "candidate_tokens": self.candidate_tokens,
            "selected_tokens": self.selected_tokens,
            "counted_tokens": self.counted_tokens,
            "consumed_before": self.consumed_before,
            "consumed_after": self.consumed_after,
            "target_tokens": self.target_tokens,
            "counter_version_after": self.counter_version_after,
            "complete": self.complete,
        }


class LossTokenBudget:
    """Reserve and commit an exact per-stage Student loss-token budget.

    ``reserve`` never mutates state. ``commit`` advances the counter only when
    ``optimizer_step_executed`` is true, so failed/overflowed attempts can be
    logged and retried without double counting.
    """

    STATE_SCHEMA_VERSION = 1
    RESERVATION_SCHEMA_VERSION = 1

    def __init__(
        self,
        target_tokens: int,
        *,
        consumed_tokens: int = 0,
        version: int = 0,
    ) -> None:
        if not _is_non_negative_integer(target_tokens) or target_tokens == 0:
            raise ValueError("target_tokens must be a positive integer")
        if not _is_non_negative_integer(consumed_tokens):
            raise ValueError("consumed_tokens must be a non-negative integer")
        if consumed_tokens > target_tokens:
            raise ValueError("consumed_tokens cannot exceed target_tokens")
        if not _is_non_negative_integer(version):
            raise ValueError("version must be a non-negative integer")
        self._target_tokens = target_tokens
        self._consumed_tokens = consumed_tokens
        self._version = version

    @property
    def target_tokens(self) -> int:
        return self._target_tokens

    @property
    def consumed_tokens(self) -> int:
        return self._consumed_tokens

    @property
    def remaining_tokens(self) -> int:
        return self._target_tokens - self._consumed_tokens

    @property
    def version(self) -> int:
        return self._version

    @property
    def complete(self) -> bool:
        return self.remaining_tokens == 0

    def reserve(self, candidate_tokens: int, *, content_digest: str) -> BudgetReservation:
        """Reserve the canonical prefix that fits in the remaining stage budget."""

        if not _is_non_negative_integer(candidate_tokens):
            raise ValueError("candidate_tokens must be a non-negative integer")
        if not _is_sha256(content_digest):
            raise ValueError("content_digest must be a lowercase SHA-256 hex digest")
        selected_tokens = min(candidate_tokens, self.remaining_tokens)
        selection_id = self._reservation_digest(
            candidate_tokens=candidate_tokens,
            selected_tokens=selected_tokens,
            content_digest=content_digest,
        )
        return BudgetReservation(
            candidate_tokens=candidate_tokens,
            selected_tokens=selected_tokens,
            consumed_before=self._consumed_tokens,
            target_tokens=self._target_tokens,
            counter_version=self._version,
            content_digest=content_digest,
            selection_id=selection_id,
        )

    def commit(
        self,
        reservation: BudgetReservation,
        *,
        optimizer_step_executed: bool,
        objective: str,
        step_id: str,
    ) -> BudgetStepRecord:
        """Commit a verified reservation after the optimizer outcome is known."""

        if not isinstance(optimizer_step_executed, bool):
            raise ValueError("optimizer_step_executed must be bool")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError("step_id must be a non-empty string")
        self._validate_reservation(reservation)

        counted_tokens = 0
        if optimizer_step_executed:
            if reservation.selected_tokens == 0:
                raise BudgetStateError("cannot execute an update with zero selected tokens")
            counted_tokens = reservation.selected_tokens
            self._consumed_tokens += counted_tokens
            self._version += 1

        return BudgetStepRecord(
            step_id=step_id,
            objective=objective,
            selection_id=reservation.selection_id,
            optimizer_step_executed=optimizer_step_executed,
            candidate_tokens=reservation.candidate_tokens,
            selected_tokens=reservation.selected_tokens,
            counted_tokens=counted_tokens,
            consumed_before=reservation.consumed_before,
            consumed_after=self._consumed_tokens,
            target_tokens=self._target_tokens,
            counter_version_after=self._version,
        )

    def state_dict(self) -> dict[str, int]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "target_tokens": self._target_tokens,
            "consumed_tokens": self._consumed_tokens,
            "version": self._version,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> LossTokenBudget:
        if not isinstance(state, Mapping):
            raise ValueError("state must be a mapping")
        required = {"schema_version", "target_tokens", "consumed_tokens", "version"}
        missing = required - set(state)
        extra = set(state) - required
        if missing or extra:
            raise ValueError(
                f"invalid state keys; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if (
            not _is_non_negative_integer(state["schema_version"])
            or state["schema_version"] != cls.STATE_SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported state schema_version={state['schema_version']!r}")
        return cls(
            target_tokens=state["target_tokens"],  # type: ignore[arg-type]
            consumed_tokens=state["consumed_tokens"],  # type: ignore[arg-type]
            version=state["version"],  # type: ignore[arg-type]
        )

    def _reservation_digest(
        self,
        *,
        candidate_tokens: int,
        selected_tokens: int,
        content_digest: str,
    ) -> str:
        payload = {
            "schema_version": self.RESERVATION_SCHEMA_VERSION,
            "target_tokens": self._target_tokens,
            "consumed_tokens": self._consumed_tokens,
            "counter_version": self._version,
            "candidate_tokens": candidate_tokens,
            "selected_tokens": selected_tokens,
            "content_digest": content_digest,
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _validate_reservation(self, reservation: BudgetReservation) -> None:
        if not isinstance(reservation, BudgetReservation):
            raise BudgetStateError("reservation must be a BudgetReservation")
        integer_fields = (
            reservation.candidate_tokens,
            reservation.selected_tokens,
            reservation.consumed_before,
            reservation.target_tokens,
            reservation.counter_version,
        )
        if not all(_is_non_negative_integer(value) for value in integer_fields):
            raise BudgetStateError("reservation contains an invalid integer field")
        if reservation.target_tokens != self._target_tokens:
            raise BudgetStateError("reservation target does not match counter target")
        if reservation.consumed_before != self._consumed_tokens:
            raise BudgetStateError("reservation was planned from a stale consumed-token state")
        if reservation.counter_version != self._version:
            raise BudgetStateError("reservation was planned from a stale counter version")
        expected_selected = min(reservation.candidate_tokens, self.remaining_tokens)
        if reservation.selected_tokens != expected_selected:
            raise BudgetStateError("reservation is not the canonical remaining-budget prefix")
        if not _is_sha256(reservation.content_digest):
            raise BudgetStateError("reservation contains an invalid content digest")
        expected_id = self._reservation_digest(
            candidate_tokens=reservation.candidate_tokens,
            selected_tokens=reservation.selected_tokens,
            content_digest=reservation.content_digest,
        )
        if reservation.selection_id != expected_id:
            raise BudgetStateError("selection_id does not match reservation contents")
