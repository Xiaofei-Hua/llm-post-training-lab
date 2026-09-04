"""Deterministic item-level metric contracts for D07 evaluation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any

from posttrain_lab.data import canonical_json_bytes, strict_json_loads, write_json_atomic

from .contracts import (
    BenchmarkDescriptor,
    BenchmarkTask,
    CheckpointIdentity,
    EvaluationContractError,
    FinishReason,
    GenerationBatch,
    GenerationProtocol,
    GenerationRecord,
    _require_exact_keys,
    _require_identifier,
    _require_int,
    _require_mapping,
    _require_sha256,
)

EVALUATOR_CONTRACT_SCHEMA_VERSION = "d07-evaluator-contract-v1"
SAMPLE_SCORE_SCHEMA_VERSION = "d07-sample-score-v1"
ITEM_SCORE_SCHEMA_VERSION = "d07-item-score-v1"
EVALUATION_REPORT_SCHEMA_VERSION = "d07-evaluation-report-v1"
SCORE_SCALE = 1_000_000
_MAX_REPORT_BYTES = 2 * 1024**3


class MetricContractError(EvaluationContractError):
    """Raised when a metric or scored result violates its frozen definition."""


class EvaluatorInfrastructureError(RuntimeError):
    """Raised when infrastructure/reference failure cannot be scored as model error."""


def fraction_to_ppm(value: Fraction) -> int:
    """Round an exact fraction to integer ppm using non-negative half-up rounding."""

    if value < 0 or value > 1:
        raise MetricContractError("metric fraction must be in [0, 1]")
    numerator = value.numerator * SCORE_SCALE
    denominator = value.denominator
    return (2 * numerator + denominator) // (2 * denominator)


def pass_at_k_fraction(*, total_samples: int, correct_samples: int, k: int) -> Fraction:
    """Exact unbiased pass@k estimator ``1-C(n-c,k)/C(n,k)``."""

    _require_int(total_samples, field="total_samples", minimum=1)
    _require_int(correct_samples, field="correct_samples", maximum=total_samples)
    _require_int(k, field="k", minimum=1, maximum=total_samples)
    denominator = math.comb(total_samples, k)
    misses = math.comb(total_samples - correct_samples, k)
    return Fraction(denominator - misses, denominator)


def pass_at_k_ppm(*, total_samples: int, correct_samples: int, k: int) -> int:
    return fraction_to_ppm(
        pass_at_k_fraction(
            total_samples=total_samples,
            correct_samples=correct_samples,
            k=k,
        )
    )


@dataclass(frozen=True, slots=True)
class EvaluatorContract:
    task: BenchmarkTask
    primary_metric: str
    pass_at_k: tuple[int, ...]
    verifier_policy_sha256: str
    score_scale: int = SCORE_SCALE
    strict_label_policy: str = "strip_nfc_case_sensitive_exact"
    schema_version: str = EVALUATOR_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATOR_CONTRACT_SCHEMA_VERSION:
            raise MetricContractError("unsupported evaluator contract schema")
        if self.primary_metric != "answer_accuracy":
            raise MetricContractError("D07 primary_metric must be answer_accuracy")
        if tuple(sorted(set(self.pass_at_k))) != self.pass_at_k or not self.pass_at_k:
            raise MetricContractError("pass_at_k must be non-empty, unique, and sorted")
        for k in self.pass_at_k:
            _require_int(k, field="pass_at_k", minimum=1, maximum=1_024)
        _require_sha256(self.verifier_policy_sha256, field="verifier_policy_sha256")
        if self.score_scale != SCORE_SCALE:
            raise MetricContractError(f"score_scale must be exactly {SCORE_SCALE}")
        if self.strict_label_policy != "strip_nfc_case_sensitive_exact":
            raise MetricContractError("unsupported strict_label_policy")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EvaluatorContract:
        required = {
            "schema_version",
            "task",
            "primary_metric",
            "pass_at_k",
            "verifier_policy_sha256",
            "score_scale",
            "strict_label_policy",
        }
        _require_exact_keys(raw, required=required, field="evaluator contract")
        pass_at_k = raw["pass_at_k"]
        if not isinstance(pass_at_k, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in pass_at_k
        ):
            raise MetricContractError("pass_at_k must be an array of integers")
        try:
            task = BenchmarkTask(raw["task"])
        except (TypeError, ValueError) as error:
            raise MetricContractError("invalid evaluator task") from error
        return cls(
            schema_version=raw["schema_version"],
            task=task,
            primary_metric=raw["primary_metric"],
            pass_at_k=tuple(pass_at_k),
            verifier_policy_sha256=raw["verifier_policy_sha256"],
            score_scale=raw["score_scale"],
            strict_label_policy=raw["strict_label_policy"],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task": self.task.value,
            "primary_metric": self.primary_metric,
            "pass_at_k": list(self.pass_at_k),
            "verifier_policy_sha256": self.verifier_policy_sha256,
            "score_scale": self.score_scale,
            "strict_label_policy": self.strict_label_policy,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_record())).hexdigest()

    def assert_compatible(
        self,
        descriptor: BenchmarkDescriptor,
        protocol: GenerationProtocol,
    ) -> None:
        if self.task is not descriptor.task:
            raise MetricContractError("evaluator task does not match benchmark descriptor")
        if self.pass_at_k[-1] > protocol.samples_per_item:
            raise MetricContractError("pass_at_k exceeds samples_per_item")
        if protocol.samples_per_item == 1 and self.pass_at_k != (1,):
            raise MetricContractError("greedy evaluation requires pass_at_k=(1,)")


def evaluator_version_sha256(
    contract: EvaluatorContract,
    *,
    backend_versions: Mapping[str, str],
) -> str:
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in backend_versions.items()
    ):
        raise MetricContractError("backend_versions must map strings to strings")
    payload = {
        "schema_version": "d07-evaluator-version-v1",
        "contract_sha256": contract.digest,
        "backend_versions": dict(sorted(backend_versions.items())),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class SampleScore:
    request_id: str
    item_id: str
    item_index: int
    sample_index: int
    generation_record_sha256: str
    correct: bool
    verification_status: str
    extraction_status: str
    parse_status: str
    answer_source: str | None
    marker_count: int
    completion_token_count: int
    finish_reason: str
    score_sha256: str
    schema_version: str = SAMPLE_SCORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SAMPLE_SCORE_SCHEMA_VERSION:
            raise MetricContractError("unsupported sample score schema")
        _require_sha256(self.request_id, field="request_id")
        _require_identifier(self.item_id, field="item_id")
        _require_int(self.item_index, field="item_index", maximum=999_999)
        _require_int(self.sample_index, field="sample_index", maximum=1_023)
        _require_sha256(
            self.generation_record_sha256,
            field="generation_record_sha256",
        )
        if not isinstance(self.correct, bool):
            raise MetricContractError("correct must be Boolean")
        for field in ("verification_status", "extraction_status", "parse_status"):
            _require_identifier(getattr(self, field), field=field)
        if self.answer_source is not None:
            _require_identifier(self.answer_source, field="answer_source")
        _require_int(self.marker_count, field="marker_count", maximum=1_000_000)
        _require_int(
            self.completion_token_count,
            field="completion_token_count",
            maximum=65_536,
        )
        _require_identifier(self.finish_reason, field="finish_reason")
        if self.finish_reason not in {reason.value for reason in FinishReason}:
            raise MetricContractError("finish_reason is not a supported generation reason")
        if self.correct != (self.verification_status == "match"):
            raise MetricContractError("correct must agree with verification_status=match")
        _require_sha256(self.score_sha256, field="score_sha256")
        expected = hashlib.sha256(canonical_json_bytes(self.unsigned_record())).hexdigest()
        if self.score_sha256 != expected:
            raise MetricContractError("sample score_sha256 mismatch")

    @classmethod
    def build(
        cls,
        generation: GenerationRecord,
        *,
        correct: bool,
        verification_status: str,
        extraction_status: str,
        parse_status: str,
        answer_source: str | None,
        marker_count: int,
    ) -> SampleScore:
        if generation.finish_reason is None:
            raise MetricContractError("cannot score generation without finish_reason")
        unsigned = {
            "schema_version": SAMPLE_SCORE_SCHEMA_VERSION,
            "request_id": generation.request_id,
            "item_id": generation.item_id,
            "item_index": generation.item_index,
            "sample_index": generation.sample_index,
            "generation_record_sha256": generation.record_sha256,
            "correct": correct,
            "verification_status": verification_status,
            "extraction_status": extraction_status,
            "parse_status": parse_status,
            "answer_source": answer_source,
            "marker_count": marker_count,
            "completion_token_count": len(generation.output_token_ids),
            "finish_reason": generation.finish_reason.value,
        }
        return cls(
            request_id=generation.request_id,
            item_id=generation.item_id,
            item_index=generation.item_index,
            sample_index=generation.sample_index,
            generation_record_sha256=generation.record_sha256,
            correct=correct,
            verification_status=verification_status,
            extraction_status=extraction_status,
            parse_status=parse_status,
            answer_source=answer_source,
            marker_count=marker_count,
            completion_token_count=len(generation.output_token_ids),
            finish_reason=generation.finish_reason.value,
            score_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SampleScore:
        required = {
            "schema_version",
            "request_id",
            "item_id",
            "item_index",
            "sample_index",
            "generation_record_sha256",
            "correct",
            "verification_status",
            "extraction_status",
            "parse_status",
            "answer_source",
            "marker_count",
            "completion_token_count",
            "finish_reason",
            "score_sha256",
        }
        _require_exact_keys(raw, required=required, field="sample score")
        return cls(
            schema_version=raw["schema_version"],
            request_id=raw["request_id"],
            item_id=raw["item_id"],
            item_index=raw["item_index"],
            sample_index=raw["sample_index"],
            generation_record_sha256=raw["generation_record_sha256"],
            correct=raw["correct"],
            verification_status=raw["verification_status"],
            extraction_status=raw["extraction_status"],
            parse_status=raw["parse_status"],
            answer_source=raw["answer_source"],
            marker_count=raw["marker_count"],
            completion_token_count=raw["completion_token_count"],
            finish_reason=raw["finish_reason"],
            score_sha256=raw["score_sha256"],
        )

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "item_id": self.item_id,
            "item_index": self.item_index,
            "sample_index": self.sample_index,
            "generation_record_sha256": self.generation_record_sha256,
            "correct": self.correct,
            "verification_status": self.verification_status,
            "extraction_status": self.extraction_status,
            "parse_status": self.parse_status,
            "answer_source": self.answer_source,
            "marker_count": self.marker_count,
            "completion_token_count": self.completion_token_count,
            "finish_reason": self.finish_reason,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "score_sha256": self.score_sha256}


@dataclass(frozen=True, slots=True)
class ItemScore:
    item_id: str
    item_index: int
    sample_score_sha256: tuple[str, ...]
    sample_correctness: tuple[bool, ...]
    correct_count: int
    pass_at_k_ppm: tuple[tuple[int, int], ...]
    item_score_sha256: str
    schema_version: str = ITEM_SCORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ITEM_SCORE_SCHEMA_VERSION:
            raise MetricContractError("unsupported item score schema")
        _require_identifier(self.item_id, field="item_id")
        _require_int(self.item_index, field="item_index", maximum=999_999)
        if not isinstance(self.sample_score_sha256, tuple) or not isinstance(
            self.sample_correctness, tuple
        ):
            raise MetricContractError("item score sample vectors must be immutable tuples")
        if not self.sample_score_sha256:
            raise MetricContractError("item score requires at least one sample")
        if len(self.sample_score_sha256) != len(self.sample_correctness):
            raise MetricContractError("sample score hashes and correctness lengths differ")
        for digest in self.sample_score_sha256:
            _require_sha256(digest, field="sample_score_sha256")
        if any(not isinstance(value, bool) for value in self.sample_correctness):
            raise MetricContractError("sample_correctness must contain Booleans")
        _require_int(
            self.correct_count,
            field="correct_count",
            maximum=len(self.sample_correctness),
        )
        if self.correct_count != sum(self.sample_correctness):
            raise MetricContractError("correct_count does not match sample_correctness")
        if not isinstance(self.pass_at_k_ppm, tuple) or not self.pass_at_k_ppm:
            raise MetricContractError("pass_at_k_ppm must be a non-empty immutable tuple")
        if tuple(sorted(self.pass_at_k_ppm)) != self.pass_at_k_ppm or len(
            {k for k, _ in self.pass_at_k_ppm}
        ) != len(self.pass_at_k_ppm):
            raise MetricContractError("pass_at_k_ppm must be unique and sorted")
        for k, value in self.pass_at_k_ppm:
            _require_int(k, field="pass k", minimum=1, maximum=len(self.sample_correctness))
            _require_int(value, field="pass ppm", maximum=SCORE_SCALE)
            expected = pass_at_k_ppm(
                total_samples=len(self.sample_correctness),
                correct_samples=self.correct_count,
                k=k,
            )
            if value != expected:
                raise MetricContractError(f"item pass@{k} is inconsistent")
        _require_sha256(self.item_score_sha256, field="item_score_sha256")
        expected_digest = hashlib.sha256(canonical_json_bytes(self.unsigned_record())).hexdigest()
        if self.item_score_sha256 != expected_digest:
            raise MetricContractError("item_score_sha256 mismatch")

    @classmethod
    def build(
        cls,
        scores: Sequence[SampleScore],
        *,
        pass_at_k: tuple[int, ...],
    ) -> ItemScore:
        frozen = tuple(scores)
        if not frozen:
            raise MetricContractError("item score requires non-empty scores")
        item_id = frozen[0].item_id
        item_index = frozen[0].item_index
        if any(
            score.item_id != item_id
            or score.item_index != item_index
            or score.sample_index != index
            for index, score in enumerate(frozen)
        ):
            raise MetricContractError("sample scores are not one canonical item sequence")
        correctness = tuple(score.correct for score in frozen)
        correct_count = sum(correctness)
        pass_values = tuple(
            (
                k,
                pass_at_k_ppm(
                    total_samples=len(frozen),
                    correct_samples=correct_count,
                    k=k,
                ),
            )
            for k in pass_at_k
        )
        unsigned = {
            "schema_version": ITEM_SCORE_SCHEMA_VERSION,
            "item_id": item_id,
            "item_index": item_index,
            "sample_score_sha256": [score.score_sha256 for score in frozen],
            "sample_correctness": list(correctness),
            "correct_count": correct_count,
            "pass_at_k_ppm": {str(k): value for k, value in pass_values},
        }
        return cls(
            item_id=item_id,
            item_index=item_index,
            sample_score_sha256=tuple(score.score_sha256 for score in frozen),
            sample_correctness=correctness,
            correct_count=correct_count,
            pass_at_k_ppm=pass_values,
            item_score_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ItemScore:
        required = {
            "schema_version",
            "item_id",
            "item_index",
            "sample_score_sha256",
            "sample_correctness",
            "correct_count",
            "pass_at_k_ppm",
            "item_score_sha256",
        }
        _require_exact_keys(raw, required=required, field="item score")
        digests = raw["sample_score_sha256"]
        correctness = raw["sample_correctness"]
        pass_values_raw = _require_mapping(raw["pass_at_k_ppm"], field="pass_at_k_ppm")
        if not isinstance(digests, list) or any(not isinstance(value, str) for value in digests):
            raise MetricContractError("sample_score_sha256 must be an array of strings")
        if not isinstance(correctness, list):
            raise MetricContractError("sample_correctness must be an array")
        pass_values: list[tuple[int, int]] = []
        for key, value in pass_values_raw.items():
            if not key.isdigit():
                raise MetricContractError("pass_at_k_ppm keys must be positive integer strings")
            pass_values.append((int(key), value))
        return cls(
            schema_version=raw["schema_version"],
            item_id=raw["item_id"],
            item_index=raw["item_index"],
            sample_score_sha256=tuple(digests),
            sample_correctness=tuple(correctness),
            correct_count=raw["correct_count"],
            pass_at_k_ppm=tuple(sorted(pass_values)),
            item_score_sha256=raw["item_score_sha256"],
        )

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "item_index": self.item_index,
            "sample_score_sha256": list(self.sample_score_sha256),
            "sample_correctness": list(self.sample_correctness),
            "correct_count": self.correct_count,
            "pass_at_k_ppm": {str(k): value for k, value in self.pass_at_k_ppm},
        }

    def to_record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "item_score_sha256": self.item_score_sha256}


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    evaluation_run_id: str
    generation_run_id: str
    benchmark_descriptor_sha256: str
    public_items_sha256: str
    sealed_references_sha256: str
    checkpoint: CheckpointIdentity
    protocol_sha256: str
    generation_record_set_sha256: str
    evaluator_contract: EvaluatorContract
    evaluator_version_sha256: str
    backend_versions: Mapping[str, str]
    sample_scores: tuple[SampleScore, ...]
    item_scores: tuple[ItemScore, ...]
    item_count: int
    sample_count: int
    correct_sample_count: int
    answer_accuracy_ppm: int
    extraction_rate_ppm: int
    parse_rate_ppm: int
    completion_token_count_total: int
    truncation_count: int
    truncation_rate_ppm: int
    pass_at_k_ppm: tuple[tuple[int, int], ...]
    verification_status_counts: tuple[tuple[str, int], ...]
    finish_reason_counts: tuple[tuple[str, int], ...]
    report_sha256: str
    schema_version: str = EVALUATION_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_REPORT_SCHEMA_VERSION:
            raise MetricContractError("unsupported evaluation report schema")
        _require_identifier(self.evaluation_run_id, field="evaluation_run_id")
        _require_identifier(self.generation_run_id, field="generation_run_id")
        for field in (
            "benchmark_descriptor_sha256",
            "public_items_sha256",
            "sealed_references_sha256",
            "protocol_sha256",
            "generation_record_set_sha256",
            "evaluator_version_sha256",
            "report_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if not self.backend_versions or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.backend_versions.items()
        ):
            raise MetricContractError("backend_versions must be a non-empty string mapping")
        object.__setattr__(
            self,
            "backend_versions",
            MappingProxyType(dict(sorted(self.backend_versions.items()))),
        )
        if self.evaluator_version_sha256 != evaluator_version_sha256(
            self.evaluator_contract,
            backend_versions=self.backend_versions,
        ):
            raise MetricContractError("evaluator_version_sha256 mismatch")
        if self.item_count != len(self.item_scores) or self.sample_count != len(self.sample_scores):
            raise MetricContractError("evaluation report counts do not match records")
        _require_int(self.item_count, field="item_count", minimum=1)
        _require_int(self.sample_count, field="sample_count", minimum=1)
        _require_int(
            self.correct_sample_count,
            field="correct_sample_count",
            maximum=self.sample_count,
        )
        for field in (
            "answer_accuracy_ppm",
            "extraction_rate_ppm",
            "parse_rate_ppm",
            "truncation_rate_ppm",
        ):
            _require_int(getattr(self, field), field=field, maximum=SCORE_SCALE)
        _require_int(
            self.completion_token_count_total,
            field="completion_token_count_total",
            maximum=self.sample_count * 65_536,
        )
        _require_int(
            self.truncation_count,
            field="truncation_count",
            maximum=self.sample_count,
        )
        if not isinstance(self.sample_scores, tuple) or not isinstance(self.item_scores, tuple):
            raise MetricContractError("evaluation records must be immutable tuples")
        if (
            not isinstance(self.pass_at_k_ppm, tuple)
            or not isinstance(self.verification_status_counts, tuple)
            or not isinstance(self.finish_reason_counts, tuple)
        ):
            raise MetricContractError("evaluation summaries must be immutable tuples")
        for index, pair in enumerate(self.pass_at_k_ppm):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise MetricContractError(f"pass_at_k_ppm[{index}] must be a pair")
            k, value = pair
            _require_int(k, field=f"pass_at_k_ppm[{index}].k", minimum=1, maximum=1_024)
            _require_int(value, field=f"pass_at_k_ppm[{index}].value", maximum=SCORE_SCALE)
        for field, pairs in (
            ("verification_status_counts", self.verification_status_counts),
            ("finish_reason_counts", self.finish_reason_counts),
        ):
            for index, pair in enumerate(pairs):
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise MetricContractError(f"{field}[{index}] must be a pair")
                key, value = pair
                _require_identifier(key, field=f"{field}[{index}].key")
                _require_int(value, field=f"{field}[{index}].value", minimum=1)
        if any(
            reason not in {candidate.value for candidate in FinishReason}
            for reason, _ in self.finish_reason_counts
        ):
            raise MetricContractError("finish_reason_counts contains an unsupported reason")
        if tuple(item.item_index for item in self.item_scores) != tuple(range(self.item_count)):
            raise MetricContractError("item scores are not in canonical item order")
        expected_sample_order = tuple(
            (item.item_index, sample_index)
            for item in self.item_scores
            for sample_index in range(len(item.sample_correctness))
        )
        actual_sample_order = tuple(
            (score.item_index, score.sample_index) for score in self.sample_scores
        )
        if expected_sample_order != actual_sample_order:
            raise MetricContractError("sample scores are not a canonical complete grid")
        sample_widths = {len(item.sample_correctness) for item in self.item_scores}
        if len(sample_widths) != 1 or self.sample_count != self.item_count * next(
            iter(sample_widths)
        ):
            raise MetricContractError("every item must have the same complete sample width")
        if any(
            tuple(k for k, _ in item.pass_at_k_ppm) != self.evaluator_contract.pass_at_k
            for item in self.item_scores
        ):
            raise MetricContractError("item pass@k keys do not match evaluator contract")
        if len({item.item_id for item in self.item_scores}) != self.item_count:
            raise MetricContractError("item scores contain duplicate item_id")
        offset = 0
        for item in self.item_scores:
            item_samples = self.sample_scores[offset : offset + len(item.sample_correctness)]
            if any(score.item_id != item.item_id for score in item_samples):
                raise MetricContractError("item score does not bind its sample item_id")
            if item.sample_score_sha256 != tuple(score.score_sha256 for score in item_samples):
                raise MetricContractError("item score does not bind its sample hashes")
            if item.sample_correctness != tuple(score.correct for score in item_samples):
                raise MetricContractError("item score does not bind sample correctness")
            offset += len(item.sample_correctness)
        if self.correct_sample_count != sum(score.correct for score in self.sample_scores):
            raise MetricContractError("correct_sample_count mismatch")
        expected_accuracy = fraction_to_ppm(Fraction(self.correct_sample_count, self.sample_count))
        if self.answer_accuracy_ppm != expected_accuracy:
            raise MetricContractError("answer_accuracy_ppm mismatch")
        extracted_count = sum(
            score.extraction_status in {"extracted", "strict_label_extracted"}
            for score in self.sample_scores
        )
        parsed_count = sum(
            score.parse_status in {"parsed", "strict_label_parsed"} for score in self.sample_scores
        )
        if self.extraction_rate_ppm != fraction_to_ppm(
            Fraction(extracted_count, self.sample_count)
        ):
            raise MetricContractError("extraction_rate_ppm mismatch")
        if self.parse_rate_ppm != fraction_to_ppm(Fraction(parsed_count, self.sample_count)):
            raise MetricContractError("parse_rate_ppm mismatch")
        expected_token_total = sum(score.completion_token_count for score in self.sample_scores)
        if self.completion_token_count_total != expected_token_total:
            raise MetricContractError("completion_token_count_total mismatch")
        expected_truncations = sum(score.finish_reason == "length" for score in self.sample_scores)
        if self.truncation_count != expected_truncations:
            raise MetricContractError("truncation_count mismatch")
        if self.truncation_rate_ppm != fraction_to_ppm(
            Fraction(expected_truncations, self.sample_count)
        ):
            raise MetricContractError("truncation_rate_ppm mismatch")
        expected_pass = tuple(
            (
                k,
                fraction_to_ppm(
                    sum(
                        pass_at_k_fraction(
                            total_samples=len(item.sample_correctness),
                            correct_samples=item.correct_count,
                            k=k,
                        )
                        for item in self.item_scores
                    )
                    / self.item_count
                ),
            )
            for k in self.evaluator_contract.pass_at_k
        )
        if self.pass_at_k_ppm != expected_pass:
            raise MetricContractError("aggregate pass_at_k_ppm mismatch")
        expected_status_counts = tuple(
            sorted(Counter(score.verification_status for score in self.sample_scores).items())
        )
        if self.verification_status_counts != expected_status_counts:
            raise MetricContractError("verification_status_counts mismatch")
        expected_finish_counts = tuple(
            sorted(Counter(score.finish_reason for score in self.sample_scores).items())
        )
        if self.finish_reason_counts != expected_finish_counts:
            raise MetricContractError("finish_reason_counts mismatch")
        expected_report = hashlib.sha256(canonical_json_bytes(self.unsigned_record())).hexdigest()
        if self.report_sha256 != expected_report:
            raise MetricContractError("evaluation report_sha256 mismatch")

    @classmethod
    def build(
        cls,
        *,
        evaluation_run_id: str,
        batch: GenerationBatch,
        sealed_references_sha256: str,
        evaluator_contract: EvaluatorContract,
        backend_versions: Mapping[str, str],
        sample_scores: Sequence[SampleScore],
    ) -> EvaluationReport:
        frozen_scores = tuple(sample_scores)
        expected_count = batch.descriptor.item_count * batch.protocol.samples_per_item
        if len(frozen_scores) != expected_count:
            raise MetricContractError("sample score count does not match generation grid")
        for score, generation in zip(frozen_scores, batch.records, strict=True):
            if (
                score.request_id != generation.request_id
                or score.generation_record_sha256 != generation.record_sha256
            ):
                raise MetricContractError("sample score does not bind its generation record")
        item_scores = tuple(
            ItemScore.build(
                frozen_scores[
                    item_index * batch.protocol.samples_per_item : (item_index + 1)
                    * batch.protocol.samples_per_item
                ],
                pass_at_k=evaluator_contract.pass_at_k,
            )
            for item_index in range(batch.descriptor.item_count)
        )
        correct_count = sum(score.correct for score in frozen_scores)
        extracted_count = sum(
            score.extraction_status in {"extracted", "strict_label_extracted"}
            for score in frozen_scores
        )
        parsed_count = sum(
            score.parse_status in {"parsed", "strict_label_parsed"} for score in frozen_scores
        )
        completion_token_total = sum(score.completion_token_count for score in frozen_scores)
        truncation_count = sum(score.finish_reason == "length" for score in frozen_scores)
        aggregate_pass = tuple(
            (
                k,
                fraction_to_ppm(
                    sum(
                        pass_at_k_fraction(
                            total_samples=len(item.sample_correctness),
                            correct_samples=item.correct_count,
                            k=k,
                        )
                        for item in item_scores
                    )
                    / len(item_scores)
                ),
            )
            for k in evaluator_contract.pass_at_k
        )
        status_counts = tuple(
            sorted(Counter(score.verification_status for score in frozen_scores).items())
        )
        finish_counts = tuple(
            sorted(Counter(score.finish_reason for score in frozen_scores).items())
        )
        version_sha256 = evaluator_version_sha256(
            evaluator_contract,
            backend_versions=backend_versions,
        )
        unsigned = {
            "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
            "evaluation_run_id": evaluation_run_id,
            "generation_run_id": batch.run_id,
            "benchmark_descriptor_sha256": batch.descriptor.digest,
            "public_items_sha256": batch.public_items_sha256,
            "sealed_references_sha256": sealed_references_sha256,
            "checkpoint": batch.checkpoint.to_record(),
            "protocol_sha256": batch.protocol.digest,
            "generation_record_set_sha256": batch.record_set_sha256,
            "evaluator_contract": evaluator_contract.to_record(),
            "evaluator_contract_sha256": evaluator_contract.digest,
            "evaluator_version_sha256": version_sha256,
            "backend_versions": dict(sorted(backend_versions.items())),
            "sample_scores": [score.to_record() for score in frozen_scores],
            "item_scores": [item.to_record() for item in item_scores],
            "item_count": len(item_scores),
            "sample_count": len(frozen_scores),
            "correct_sample_count": correct_count,
            "answer_accuracy_ppm": fraction_to_ppm(Fraction(correct_count, len(frozen_scores))),
            "extraction_rate_ppm": fraction_to_ppm(Fraction(extracted_count, len(frozen_scores))),
            "parse_rate_ppm": fraction_to_ppm(Fraction(parsed_count, len(frozen_scores))),
            "completion_token_count_total": completion_token_total,
            "truncation_count": truncation_count,
            "truncation_rate_ppm": fraction_to_ppm(Fraction(truncation_count, len(frozen_scores))),
            "pass_at_k_ppm": {str(k): value for k, value in aggregate_pass},
            "verification_status_counts": dict(status_counts),
            "finish_reason_counts": dict(finish_counts),
        }
        return cls(
            evaluation_run_id=evaluation_run_id,
            generation_run_id=batch.run_id,
            benchmark_descriptor_sha256=batch.descriptor.digest,
            public_items_sha256=batch.public_items_sha256,
            sealed_references_sha256=sealed_references_sha256,
            checkpoint=batch.checkpoint,
            protocol_sha256=batch.protocol.digest,
            generation_record_set_sha256=batch.record_set_sha256,
            evaluator_contract=evaluator_contract,
            evaluator_version_sha256=version_sha256,
            backend_versions=dict(sorted(backend_versions.items())),
            sample_scores=frozen_scores,
            item_scores=item_scores,
            item_count=len(item_scores),
            sample_count=len(frozen_scores),
            correct_sample_count=correct_count,
            answer_accuracy_ppm=fraction_to_ppm(Fraction(correct_count, len(frozen_scores))),
            extraction_rate_ppm=fraction_to_ppm(Fraction(extracted_count, len(frozen_scores))),
            parse_rate_ppm=fraction_to_ppm(Fraction(parsed_count, len(frozen_scores))),
            completion_token_count_total=completion_token_total,
            truncation_count=truncation_count,
            truncation_rate_ppm=fraction_to_ppm(Fraction(truncation_count, len(frozen_scores))),
            pass_at_k_ppm=aggregate_pass,
            verification_status_counts=status_counts,
            finish_reason_counts=finish_counts,
            report_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EvaluationReport:
        required = {
            "schema_version",
            "evaluation_run_id",
            "generation_run_id",
            "benchmark_descriptor_sha256",
            "public_items_sha256",
            "sealed_references_sha256",
            "checkpoint",
            "protocol_sha256",
            "generation_record_set_sha256",
            "evaluator_contract",
            "evaluator_contract_sha256",
            "evaluator_version_sha256",
            "backend_versions",
            "sample_scores",
            "item_scores",
            "item_count",
            "sample_count",
            "correct_sample_count",
            "answer_accuracy_ppm",
            "extraction_rate_ppm",
            "parse_rate_ppm",
            "completion_token_count_total",
            "truncation_count",
            "truncation_rate_ppm",
            "pass_at_k_ppm",
            "verification_status_counts",
            "finish_reason_counts",
            "report_sha256",
        }
        _require_exact_keys(raw, required=required, field="evaluation report")
        contract = EvaluatorContract.from_mapping(
            _require_mapping(raw["evaluator_contract"], field="evaluator_contract")
        )
        if raw["evaluator_contract_sha256"] != contract.digest:
            raise MetricContractError("evaluator_contract_sha256 mismatch")
        sample_rows = raw["sample_scores"]
        item_rows = raw["item_scores"]
        if not isinstance(sample_rows, list) or not isinstance(item_rows, list):
            raise MetricContractError("sample_scores and item_scores must be arrays")
        backend_versions = _require_mapping(raw["backend_versions"], field="backend_versions")
        pass_values_raw = _require_mapping(raw["pass_at_k_ppm"], field="pass_at_k_ppm")
        status_counts_raw = _require_mapping(
            raw["verification_status_counts"], field="verification_status_counts"
        )
        finish_counts_raw = _require_mapping(
            raw["finish_reason_counts"], field="finish_reason_counts"
        )
        pass_values: list[tuple[int, int]] = []
        for key, value in pass_values_raw.items():
            if not key.isdigit():
                raise MetricContractError("pass_at_k_ppm keys must be integer strings")
            pass_values.append((int(key), value))
        return cls(
            schema_version=raw["schema_version"],
            evaluation_run_id=raw["evaluation_run_id"],
            generation_run_id=raw["generation_run_id"],
            benchmark_descriptor_sha256=raw["benchmark_descriptor_sha256"],
            public_items_sha256=raw["public_items_sha256"],
            sealed_references_sha256=raw["sealed_references_sha256"],
            checkpoint=CheckpointIdentity.from_mapping(
                _require_mapping(raw["checkpoint"], field="checkpoint")
            ),
            protocol_sha256=raw["protocol_sha256"],
            generation_record_set_sha256=raw["generation_record_set_sha256"],
            evaluator_contract=contract,
            evaluator_version_sha256=raw["evaluator_version_sha256"],
            backend_versions=dict(backend_versions),
            sample_scores=tuple(
                SampleScore.from_mapping(_require_mapping(row, field="sample score"))
                for row in sample_rows
            ),
            item_scores=tuple(
                ItemScore.from_mapping(_require_mapping(row, field="item score"))
                for row in item_rows
            ),
            item_count=raw["item_count"],
            sample_count=raw["sample_count"],
            correct_sample_count=raw["correct_sample_count"],
            answer_accuracy_ppm=raw["answer_accuracy_ppm"],
            extraction_rate_ppm=raw["extraction_rate_ppm"],
            parse_rate_ppm=raw["parse_rate_ppm"],
            completion_token_count_total=raw["completion_token_count_total"],
            truncation_count=raw["truncation_count"],
            truncation_rate_ppm=raw["truncation_rate_ppm"],
            pass_at_k_ppm=tuple(sorted(pass_values)),
            verification_status_counts=tuple(sorted(status_counts_raw.items())),
            finish_reason_counts=tuple(sorted(finish_counts_raw.items())),
            report_sha256=raw["report_sha256"],
        )

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_run_id": self.evaluation_run_id,
            "generation_run_id": self.generation_run_id,
            "benchmark_descriptor_sha256": self.benchmark_descriptor_sha256,
            "public_items_sha256": self.public_items_sha256,
            "sealed_references_sha256": self.sealed_references_sha256,
            "checkpoint": self.checkpoint.to_record(),
            "protocol_sha256": self.protocol_sha256,
            "generation_record_set_sha256": self.generation_record_set_sha256,
            "evaluator_contract": self.evaluator_contract.to_record(),
            "evaluator_contract_sha256": self.evaluator_contract.digest,
            "evaluator_version_sha256": self.evaluator_version_sha256,
            "backend_versions": dict(sorted(self.backend_versions.items())),
            "sample_scores": [score.to_record() for score in self.sample_scores],
            "item_scores": [item.to_record() for item in self.item_scores],
            "item_count": self.item_count,
            "sample_count": self.sample_count,
            "correct_sample_count": self.correct_sample_count,
            "answer_accuracy_ppm": self.answer_accuracy_ppm,
            "extraction_rate_ppm": self.extraction_rate_ppm,
            "parse_rate_ppm": self.parse_rate_ppm,
            "completion_token_count_total": self.completion_token_count_total,
            "truncation_count": self.truncation_count,
            "truncation_rate_ppm": self.truncation_rate_ppm,
            "pass_at_k_ppm": {str(k): value for k, value in self.pass_at_k_ppm},
            "verification_status_counts": dict(self.verification_status_counts),
            "finish_reason_counts": dict(self.finish_reason_counts),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "report_sha256": self.report_sha256}


def write_evaluation_report(report: EvaluationReport, path: str | Path) -> None:
    write_json_atomic(report.to_record(), path)


def load_evaluation_report(path: str | Path) -> EvaluationReport:
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
        if size > _MAX_REPORT_BYTES:
            raise MetricContractError(f"evaluation report exceeds {_MAX_REPORT_BYTES} bytes")
        raw = resolved.read_bytes()
    except OSError as error:
        raise MetricContractError(f"cannot read evaluation report: {error}") from error
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise MetricContractError("evaluation report must be canonical UTF-8 without BOM/CR")
    try:
        parsed = strict_json_loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError) as error:
        raise MetricContractError(f"invalid strict evaluation report: {error}") from error
    return EvaluationReport.from_mapping(_require_mapping(parsed, field="evaluation report"))
