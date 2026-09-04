"""Versioned, text-free contracts for D08 paired statistical inference."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any

from posttrain_lab.data import canonical_json_bytes, strict_json_loads, write_json_atomic
from posttrain_lab.evaluation.contracts import (
    DecodingMode,
    GenerationProtocol,
    LoadedPublicBenchmark,
)
from posttrain_lab.evaluation.metrics import EvaluationReport

STATISTICS_PROTOCOL_SCHEMA_VERSION = "d08-statistics-protocol-v1"
PAIRED_PANEL_SCHEMA_VERSION = "d08-paired-panel-v1"
RATIONAL_VALUE_SCHEMA_VERSION = "d08-rational-value-v1"
CONFIDENCE_INTERVAL_SCHEMA_VERSION = "d08-confidence-interval-v1"

SCORE_SCALE = 1_000_000
PREREGISTERED_TRAINING_SEEDS = (101, 202, 303)
PREREGISTERED_BOOTSTRAP_REPETITIONS = 10_000
PREREGISTERED_RANDOMIZATION_REPETITIONS = 100_000
PREREGISTERED_BOOTSTRAP_SEED = 2_026_090_408
PREREGISTERED_RANDOMIZATION_SEED = 2_026_090_409
PREREGISTERED_CONFIDENCE_PPM = 950_000
PREREGISTERED_EQUIVALENCE_CONFIDENCE_PPM = 900_000
PREREGISTERED_ALPHA_PPM = 50_000
PREREGISTERED_EFFECT_PPM = 20_000
PREREGISTERED_EQUIVALENCE_MARGIN_PPM = 20_000

_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_CONTROL_FILE_BYTES = 2 * 1024**3
_MAX_ITEMS = 1_000_000
_MAX_REPETITIONS = 10_000_000
_MAX_SEED = 2**63 - 1
_ARM_ORDER = ("A0", "A1", "A2", "A3", "A4")


class StatisticsContractError(ValueError):
    """Raised when a D08 input or output violates its frozen contract."""


class ArmId(StrEnum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"


def _require_exact_keys(
    raw: Mapping[str, Any], *, required: set[str] | frozenset[str], field: str
) -> None:
    missing = required - set(raw)
    extra = set(raw) - required
    if missing or extra:
        raise StatisticsContractError(
            f"{field} has invalid keys; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StatisticsContractError(f"{field} must be a string-keyed mapping")
    return value


def _require_int(
    value: object,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StatisticsContractError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise StatisticsContractError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise StatisticsContractError(f"{field} must be <= {maximum}")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise StatisticsContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(
            not (
                character.isascii()
                and (character.isalnum() or character in "._+@:/-")
            )
            for character in value
        )
        or not value[0].isalnum()
        or ".." in value
        or value.endswith(("/", ":"))
    ):
        raise StatisticsContractError(f"{field} must be a bounded portable identifier")
    return value


def _signed_fraction_to_ppm(value: Fraction) -> int:
    if value < -1 or value > 1:
        raise StatisticsContractError("rational value must be in [-1, 1]")
    sign = -1 if value < 0 else 1
    absolute = abs(value)
    scaled_numerator = absolute.numerator * SCORE_SCALE
    rounded = (2 * scaled_numerator + absolute.denominator) // (2 * absolute.denominator)
    return sign * rounded


def _read_json_object(path: str | Path, *, field: str) -> tuple[str, Mapping[str, Any]]:
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
        if size > _MAX_CONTROL_FILE_BYTES:
            raise StatisticsContractError(f"{field} exceeds {_MAX_CONTROL_FILE_BYTES} bytes")
        raw = resolved.read_bytes()
    except OSError as error:
        raise StatisticsContractError(f"cannot read {field}: {error}") from error
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise StatisticsContractError(f"{field} must be canonical UTF-8 without BOM/CR")
    try:
        parsed = strict_json_loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError) as error:
        raise StatisticsContractError(f"invalid strict {field}: {error}") from error
    return hashlib.sha256(raw).hexdigest(), _require_mapping(parsed, field=field)


@dataclass(frozen=True, slots=True)
class RationalValue:
    """A reduced exact value in [-1, 1] with a checked display-scale projection."""

    numerator: int
    denominator: int
    ppm: int
    schema_version: str = RATIONAL_VALUE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RATIONAL_VALUE_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported rational-value schema")
        _require_int(self.numerator, field="numerator")
        _require_int(self.denominator, field="denominator", minimum=1)
        _require_int(self.ppm, field="ppm", minimum=-SCORE_SCALE, maximum=SCORE_SCALE)
        value = Fraction(self.numerator, self.denominator)
        if (value.numerator, value.denominator) != (self.numerator, self.denominator):
            raise StatisticsContractError(
                "rational value must be reduced with positive denominator"
            )
        if value < -1 or value > 1:
            raise StatisticsContractError("rational value must be in [-1, 1]")
        if self.ppm != _signed_fraction_to_ppm(value):
            raise StatisticsContractError("ppm does not match exact rational value")

    @classmethod
    def from_fraction(cls, value: Fraction | int) -> RationalValue:
        exact = Fraction(value)
        return cls(
            numerator=exact.numerator,
            denominator=exact.denominator,
            ppm=_signed_fraction_to_ppm(exact),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, field: str) -> RationalValue:
        _require_exact_keys(
            raw,
            required={"schema_version", "numerator", "denominator", "ppm"},
            field=field,
        )
        return cls(
            schema_version=raw["schema_version"],
            numerator=raw["numerator"],
            denominator=raw["denominator"],
            ppm=raw["ppm"],
        )

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "ppm": self.ppm,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    confidence_ppm: int
    lower: RationalValue
    upper: RationalValue
    schema_version: str = CONFIDENCE_INTERVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIDENCE_INTERVAL_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported confidence-interval schema")
        _require_int(
            self.confidence_ppm,
            field="confidence_ppm",
            minimum=1,
            maximum=SCORE_SCALE - 1,
        )
        if self.lower.fraction > self.upper.fraction:
            raise StatisticsContractError("confidence interval lower exceeds upper")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, field: str) -> ConfidenceInterval:
        _require_exact_keys(
            raw,
            required={"schema_version", "confidence_ppm", "lower", "upper"},
            field=field,
        )
        return cls(
            schema_version=raw["schema_version"],
            confidence_ppm=raw["confidence_ppm"],
            lower=RationalValue.from_mapping(
                _require_mapping(raw["lower"], field=f"{field}.lower"),
                field=f"{field}.lower",
            ),
            upper=RationalValue.from_mapping(
                _require_mapping(raw["upper"], field=f"{field}.upper"),
                field=f"{field}.upper",
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "confidence_ppm": self.confidence_ppm,
            "lower": self.lower.to_record(),
            "upper": self.upper.to_record(),
        }


@dataclass(frozen=True, slots=True)
class StatisticsProtocol:
    training_seeds: tuple[int, ...]
    stratum_key: str
    bootstrap_repetitions: int
    randomization_repetitions: int
    bootstrap_seed: int
    randomization_seed: int
    confidence_ppm: int
    equivalence_confidence_ppm: int
    family_alpha_ppm: int
    practical_effect_ppm: int
    equivalence_margin_ppm: int
    rng_algorithm: str = "numpy-pcg64-v1"
    quantile_method: str = "linear-type7-exact"
    randomization_correction: str = "plus-one"
    schema_version: str = STATISTICS_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATISTICS_PROTOCOL_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported statistics-protocol schema")
        if not isinstance(self.training_seeds, tuple) or not self.training_seeds:
            raise StatisticsContractError("training_seeds must be a non-empty immutable tuple")
        for index, seed in enumerate(self.training_seeds):
            _require_int(seed, field=f"training_seeds[{index}]", minimum=0, maximum=_MAX_SEED)
        if tuple(sorted(set(self.training_seeds))) != self.training_seeds:
            raise StatisticsContractError("training_seeds must be unique and sorted")
        _require_identifier(self.stratum_key, field="stratum_key")
        _require_int(
            self.bootstrap_repetitions,
            field="bootstrap_repetitions",
            minimum=1,
            maximum=_MAX_REPETITIONS,
        )
        _require_int(
            self.randomization_repetitions,
            field="randomization_repetitions",
            minimum=1,
            maximum=_MAX_REPETITIONS,
        )
        _require_int(self.bootstrap_seed, field="bootstrap_seed", minimum=0, maximum=_MAX_SEED)
        _require_int(
            self.randomization_seed,
            field="randomization_seed",
            minimum=0,
            maximum=_MAX_SEED,
        )
        for field in (
            "confidence_ppm",
            "equivalence_confidence_ppm",
            "family_alpha_ppm",
            "practical_effect_ppm",
            "equivalence_margin_ppm",
        ):
            _require_int(getattr(self, field), field=field, minimum=1, maximum=SCORE_SCALE - 1)
        if self.equivalence_confidence_ppm >= self.confidence_ppm:
            raise StatisticsContractError(
                "equivalence confidence must be lower than superiority confidence"
            )
        if self.rng_algorithm != "numpy-pcg64-v1":
            raise StatisticsContractError("unsupported RNG algorithm")
        if self.quantile_method != "linear-type7-exact":
            raise StatisticsContractError("unsupported quantile method")
        if self.randomization_correction != "plus-one":
            raise StatisticsContractError("unsupported randomization correction")

    @classmethod
    def preregistered(cls) -> StatisticsProtocol:
        return cls(
            training_seeds=PREREGISTERED_TRAINING_SEEDS,
            stratum_key="level",
            bootstrap_repetitions=PREREGISTERED_BOOTSTRAP_REPETITIONS,
            randomization_repetitions=PREREGISTERED_RANDOMIZATION_REPETITIONS,
            bootstrap_seed=PREREGISTERED_BOOTSTRAP_SEED,
            randomization_seed=PREREGISTERED_RANDOMIZATION_SEED,
            confidence_ppm=PREREGISTERED_CONFIDENCE_PPM,
            equivalence_confidence_ppm=PREREGISTERED_EQUIVALENCE_CONFIDENCE_PPM,
            family_alpha_ppm=PREREGISTERED_ALPHA_PPM,
            practical_effect_ppm=PREREGISTERED_EFFECT_PPM,
            equivalence_margin_ppm=PREREGISTERED_EQUIVALENCE_MARGIN_PPM,
        )

    def assert_preregistered(self) -> None:
        expected = StatisticsProtocol.preregistered()
        if self != expected:
            raise StatisticsContractError(
                "formal confirmatory analysis requires the exact preregistered D08 protocol"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StatisticsProtocol:
        required = {
            "schema_version",
            "training_seeds",
            "stratum_key",
            "bootstrap_repetitions",
            "randomization_repetitions",
            "bootstrap_seed",
            "randomization_seed",
            "confidence_ppm",
            "equivalence_confidence_ppm",
            "family_alpha_ppm",
            "practical_effect_ppm",
            "equivalence_margin_ppm",
            "rng_algorithm",
            "quantile_method",
            "randomization_correction",
        }
        _require_exact_keys(raw, required=required, field="statistics protocol")
        seeds = raw["training_seeds"]
        if not isinstance(seeds, list):
            raise StatisticsContractError("training_seeds must be an array")
        return cls(
            schema_version=raw["schema_version"],
            training_seeds=tuple(seeds),
            stratum_key=raw["stratum_key"],
            bootstrap_repetitions=raw["bootstrap_repetitions"],
            randomization_repetitions=raw["randomization_repetitions"],
            bootstrap_seed=raw["bootstrap_seed"],
            randomization_seed=raw["randomization_seed"],
            confidence_ppm=raw["confidence_ppm"],
            equivalence_confidence_ppm=raw["equivalence_confidence_ppm"],
            family_alpha_ppm=raw["family_alpha_ppm"],
            practical_effect_ppm=raw["practical_effect_ppm"],
            equivalence_margin_ppm=raw["equivalence_margin_ppm"],
            rng_algorithm=raw["rng_algorithm"],
            quantile_method=raw["quantile_method"],
            randomization_correction=raw["randomization_correction"],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "training_seeds": list(self.training_seeds),
            "stratum_key": self.stratum_key,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "randomization_repetitions": self.randomization_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "randomization_seed": self.randomization_seed,
            "confidence_ppm": self.confidence_ppm,
            "equivalence_confidence_ppm": self.equivalence_confidence_ppm,
            "family_alpha_ppm": self.family_alpha_ppm,
            "practical_effect_ppm": self.practical_effect_ppm,
            "equivalence_margin_ppm": self.equivalence_margin_ppm,
            "rng_algorithm": self.rng_algorithm,
            "quantile_method": self.quantile_method,
            "randomization_correction": self.randomization_correction,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_record())).hexdigest()


@dataclass(frozen=True, slots=True)
class PanelReportRef:
    arm: ArmId
    training_seed: int
    evaluation_run_id: str
    checkpoint_sha256: str
    report_sha256: str

    def __post_init__(self) -> None:
        _require_int(self.training_seed, field="training_seed", minimum=0, maximum=_MAX_SEED)
        _require_identifier(self.evaluation_run_id, field="evaluation_run_id")
        _require_sha256(self.checkpoint_sha256, field="checkpoint_sha256")
        _require_sha256(self.report_sha256, field="report_sha256")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, field: str) -> PanelReportRef:
        _require_exact_keys(
            raw,
            required={
                "arm",
                "training_seed",
                "evaluation_run_id",
                "checkpoint_sha256",
                "report_sha256",
            },
            field=field,
        )
        try:
            arm = ArmId(raw["arm"])
        except (TypeError, ValueError) as error:
            raise StatisticsContractError(f"{field}.arm is invalid") from error
        return cls(
            arm=arm,
            training_seed=raw["training_seed"],
            evaluation_run_id=raw["evaluation_run_id"],
            checkpoint_sha256=raw["checkpoint_sha256"],
            report_sha256=raw["report_sha256"],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "training_seed": self.training_seed,
            "evaluation_run_id": self.evaluation_run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "report_sha256": self.report_sha256,
        }


@dataclass(frozen=True, slots=True)
class PairedItem:
    item_id: str
    item_index: int
    stratum: str
    correctness: tuple[tuple[ArmId, tuple[bool, ...]], ...]

    def __post_init__(self) -> None:
        _require_identifier(self.item_id, field="item_id")
        _require_int(self.item_index, field="item_index", minimum=0, maximum=_MAX_ITEMS - 1)
        _require_identifier(self.stratum, field="stratum")
        if not isinstance(self.correctness, tuple):
            raise StatisticsContractError("correctness must be an immutable tuple")
        if tuple(arm.value for arm, _ in self.correctness) != _ARM_ORDER:
            raise StatisticsContractError("correctness must contain A0-A4 in canonical order")
        widths = set()
        for arm, values in self.correctness:
            if not isinstance(arm, ArmId) or not isinstance(values, tuple) or not values:
                raise StatisticsContractError(
                    "correctness entries must be non-empty immutable vectors"
                )
            if any(not isinstance(value, bool) for value in values):
                raise StatisticsContractError("correctness vectors must contain Booleans")
            widths.add(len(values))
        if len(widths) != 1:
            raise StatisticsContractError("all correctness vectors must have equal seed width")

    def for_arm(self, arm: ArmId) -> tuple[bool, ...]:
        return dict(self.correctness)[arm]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        field: str,
        seed_count: int,
    ) -> PairedItem:
        _require_exact_keys(
            raw,
            required={"item_id", "item_index", "stratum", "correctness"},
            field=field,
        )
        correctness_raw = _require_mapping(raw["correctness"], field=f"{field}.correctness")
        if tuple(correctness_raw) != _ARM_ORDER:
            raise StatisticsContractError(
                f"{field}.correctness keys must be A0-A4 in canonical JSON order"
            )
        correctness: list[tuple[ArmId, tuple[bool, ...]]] = []
        for arm_name in _ARM_ORDER:
            values = correctness_raw[arm_name]
            if not isinstance(values, list) or len(values) != seed_count:
                raise StatisticsContractError(
                    f"{field}.correctness.{arm_name} must have one value per training seed"
                )
            correctness.append((ArmId(arm_name), tuple(values)))
        return cls(
            item_id=raw["item_id"],
            item_index=raw["item_index"],
            stratum=raw["stratum"],
            correctness=tuple(correctness),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "item_index": self.item_index,
            "stratum": self.stratum,
            "correctness": {arm.value: list(values) for arm, values in self.correctness},
        }


@dataclass(frozen=True, slots=True)
class PairedPanel:
    benchmark_descriptor_sha256: str
    public_items_sha256: str
    public_item_set_sha256: str
    sealed_references_sha256: str
    generation_protocol_sha256: str
    evaluator_contract_sha256: str
    evaluator_version_sha256: str
    stratum_key: str
    training_seeds: tuple[int, ...]
    report_refs: tuple[PanelReportRef, ...]
    items: tuple[PairedItem, ...]
    panel_sha256: str
    schema_version: str = PAIRED_PANEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_PANEL_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported paired-panel schema")
        for field in (
            "benchmark_descriptor_sha256",
            "public_items_sha256",
            "public_item_set_sha256",
            "sealed_references_sha256",
            "generation_protocol_sha256",
            "evaluator_contract_sha256",
            "evaluator_version_sha256",
            "panel_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        _require_identifier(self.stratum_key, field="stratum_key")
        if self.training_seeds != PREREGISTERED_TRAINING_SEEDS:
            raise StatisticsContractError("paired panel requires seeds (101, 202, 303)")
        if not isinstance(self.report_refs, tuple) or not isinstance(self.items, tuple):
            raise StatisticsContractError("paired panel collections must be immutable tuples")
        expected_ref_keys = tuple(
            (ArmId(arm), seed) for arm in _ARM_ORDER for seed in self.training_seeds
        )
        actual_ref_keys = tuple((ref.arm, ref.training_seed) for ref in self.report_refs)
        if actual_ref_keys != expected_ref_keys:
            raise StatisticsContractError("report_refs must cover A0-A4 x three seeds canonically")
        if len({ref.report_sha256 for ref in self.report_refs}) != len(self.report_refs):
            raise StatisticsContractError("report_refs must contain unique D07 report hashes")
        if len({ref.evaluation_run_id for ref in self.report_refs}) != len(self.report_refs):
            raise StatisticsContractError(
                "report_refs must contain unique evaluation_run_id values"
            )
        if not self.items or len(self.items) > _MAX_ITEMS:
            raise StatisticsContractError("paired panel item count is outside supported bounds")
        if tuple(item.item_index for item in self.items) != tuple(range(len(self.items))):
            raise StatisticsContractError("paired items must use consecutive canonical indices")
        if len({item.item_id for item in self.items}) != len(self.items):
            raise StatisticsContractError("paired items contain duplicate item_id")
        if any(len(item.for_arm(ArmId.A0)) != len(self.training_seeds) for item in self.items):
            raise StatisticsContractError("item correctness width does not match training seeds")
        expected_digest = hashlib.sha256(canonical_json_bytes(self.unsigned_record())).hexdigest()
        if self.panel_sha256 != expected_digest:
            raise StatisticsContractError("panel_sha256 mismatch")

    @classmethod
    def build(
        cls,
        *,
        benchmark_descriptor_sha256: str,
        public_items_sha256: str,
        public_item_set_sha256: str,
        sealed_references_sha256: str,
        generation_protocol_sha256: str,
        evaluator_contract_sha256: str,
        evaluator_version_sha256: str,
        stratum_key: str,
        training_seeds: tuple[int, ...],
        report_refs: Sequence[PanelReportRef],
        items: Sequence[PairedItem],
    ) -> PairedPanel:
        unsigned = {
            "schema_version": PAIRED_PANEL_SCHEMA_VERSION,
            "benchmark_descriptor_sha256": benchmark_descriptor_sha256,
            "public_items_sha256": public_items_sha256,
            "public_item_set_sha256": public_item_set_sha256,
            "sealed_references_sha256": sealed_references_sha256,
            "generation_protocol_sha256": generation_protocol_sha256,
            "evaluator_contract_sha256": evaluator_contract_sha256,
            "evaluator_version_sha256": evaluator_version_sha256,
            "stratum_key": stratum_key,
            "training_seeds": list(training_seeds),
            "report_refs": [reference.to_record() for reference in report_refs],
            "items": [item.to_record() for item in items],
        }
        return cls(
            benchmark_descriptor_sha256=benchmark_descriptor_sha256,
            public_items_sha256=public_items_sha256,
            public_item_set_sha256=public_item_set_sha256,
            sealed_references_sha256=sealed_references_sha256,
            generation_protocol_sha256=generation_protocol_sha256,
            evaluator_contract_sha256=evaluator_contract_sha256,
            evaluator_version_sha256=evaluator_version_sha256,
            stratum_key=stratum_key,
            training_seeds=training_seeds,
            report_refs=tuple(report_refs),
            items=tuple(items),
            panel_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PairedPanel:
        required = {
            "schema_version",
            "benchmark_descriptor_sha256",
            "public_items_sha256",
            "public_item_set_sha256",
            "sealed_references_sha256",
            "generation_protocol_sha256",
            "evaluator_contract_sha256",
            "evaluator_version_sha256",
            "stratum_key",
            "training_seeds",
            "report_refs",
            "items",
            "panel_sha256",
        }
        _require_exact_keys(raw, required=required, field="paired panel")
        seeds = raw["training_seeds"]
        refs = raw["report_refs"]
        items = raw["items"]
        if not isinstance(seeds, list) or not isinstance(refs, list) or not isinstance(items, list):
            raise StatisticsContractError("panel seeds, refs, and items must be arrays")
        seed_tuple = tuple(seeds)
        return cls(
            schema_version=raw["schema_version"],
            benchmark_descriptor_sha256=raw["benchmark_descriptor_sha256"],
            public_items_sha256=raw["public_items_sha256"],
            public_item_set_sha256=raw["public_item_set_sha256"],
            sealed_references_sha256=raw["sealed_references_sha256"],
            generation_protocol_sha256=raw["generation_protocol_sha256"],
            evaluator_contract_sha256=raw["evaluator_contract_sha256"],
            evaluator_version_sha256=raw["evaluator_version_sha256"],
            stratum_key=raw["stratum_key"],
            training_seeds=seed_tuple,
            report_refs=tuple(
                PanelReportRef.from_mapping(
                    _require_mapping(value, field=f"report_refs[{index}]"),
                    field=f"report_refs[{index}]",
                )
                for index, value in enumerate(refs)
            ),
            items=tuple(
                PairedItem.from_mapping(
                    _require_mapping(value, field=f"items[{index}]"),
                    field=f"items[{index}]",
                    seed_count=len(seed_tuple),
                )
                for index, value in enumerate(items)
            ),
            panel_sha256=raw["panel_sha256"],
        )

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "benchmark_descriptor_sha256": self.benchmark_descriptor_sha256,
            "public_items_sha256": self.public_items_sha256,
            "public_item_set_sha256": self.public_item_set_sha256,
            "sealed_references_sha256": self.sealed_references_sha256,
            "generation_protocol_sha256": self.generation_protocol_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "evaluator_version_sha256": self.evaluator_version_sha256,
            "stratum_key": self.stratum_key,
            "training_seeds": list(self.training_seeds),
            "report_refs": [reference.to_record() for reference in self.report_refs],
            "items": [item.to_record() for item in self.items],
        }

    def to_record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "panel_sha256": self.panel_sha256}


@dataclass(frozen=True, slots=True)
class ArmSeedEvaluation:
    arm: ArmId
    training_seed: int
    report: EvaluationReport

    def __post_init__(self) -> None:
        _require_int(self.training_seed, field="training_seed", minimum=0, maximum=_MAX_SEED)
        if not isinstance(self.report, EvaluationReport):
            raise StatisticsContractError("report must be a validated D07 EvaluationReport")


def build_paired_panel(
    public: LoadedPublicBenchmark,
    generation_protocol: GenerationProtocol,
    evaluations: Sequence[ArmSeedEvaluation],
    *,
    stratum_key: str = "level",
) -> PairedPanel:
    """Project validated greedy D07 reports into the only D08 analysis surface."""

    if not isinstance(public, LoadedPublicBenchmark):
        raise StatisticsContractError("public must be a validated D07 public benchmark")
    if generation_protocol.mode is not DecodingMode.GREEDY:
        raise StatisticsContractError("D08 confirmatory panel requires greedy evaluation")
    _require_identifier(stratum_key, field="stratum_key")
    frozen = tuple(evaluations)
    expected_keys = tuple(
        (ArmId(arm), seed) for arm in _ARM_ORDER for seed in PREREGISTERED_TRAINING_SEEDS
    )
    actual_keys = tuple((entry.arm, entry.training_seed) for entry in frozen)
    if actual_keys != expected_keys:
        raise StatisticsContractError("evaluations must cover A0-A4 x seeds canonically")
    reports = tuple(entry.report for entry in frozen)
    first = reports[0]
    expected_item_ids = tuple(item.item_id for item in public.items)
    for entry in frozen:
        report = entry.report
        if report.benchmark_descriptor_sha256 != public.descriptor.digest:
            raise StatisticsContractError(
                "D07 report benchmark descriptor does not match public data"
            )
        if report.public_items_sha256 != public.raw_sha256:
            raise StatisticsContractError("D07 report public-items hash does not match public data")
        if report.sealed_references_sha256 != public.descriptor.sealed_references_sha256:
            raise StatisticsContractError(
                "D07 report sealed-reference hash does not match descriptor"
            )
        if report.protocol_sha256 != generation_protocol.digest:
            raise StatisticsContractError("D07 report does not use the supplied greedy protocol")
        if report.evaluator_contract.primary_metric != "answer_accuracy":
            raise StatisticsContractError("D08 requires D07 answer_accuracy")
        if report.evaluator_contract.pass_at_k != (1,):
            raise StatisticsContractError("D08 requires one greedy correctness value per item")
        if report.item_count != len(public.items):
            raise StatisticsContractError("D07 report item count does not match public data")
        if tuple(item.item_id for item in report.item_scores) != expected_item_ids:
            raise StatisticsContractError(
                "D07 report item identity/order does not match public data"
            )
        if any(len(item.sample_correctness) != 1 for item in report.item_scores):
            raise StatisticsContractError("D08 refuses non-greedy correctness vectors")
        if report.evaluator_contract.digest != first.evaluator_contract.digest:
            raise StatisticsContractError("D07 reports use different evaluator contracts")
        if report.evaluator_version_sha256 != first.evaluator_version_sha256:
            raise StatisticsContractError("D07 reports use different evaluator versions")
    strata: list[str] = []
    for item in public.items:
        by_key = dict(item.strata)
        if stratum_key not in by_key:
            raise StatisticsContractError(
                f"public item {item.item_id} is missing required stratum {stratum_key}"
            )
        strata.append(by_key[stratum_key])
    by_key = {(entry.arm, entry.training_seed): entry.report for entry in frozen}
    paired_items = tuple(
        PairedItem(
            item_id=public_item.item_id,
            item_index=public_item.item_index,
            stratum=strata[index],
            correctness=tuple(
                (
                    ArmId(arm),
                    tuple(
                        by_key[(ArmId(arm), seed)].item_scores[index].sample_correctness[0]
                        for seed in PREREGISTERED_TRAINING_SEEDS
                    ),
                )
                for arm in _ARM_ORDER
            ),
        )
        for index, public_item in enumerate(public.items)
    )
    references = tuple(
        PanelReportRef(
            arm=entry.arm,
            training_seed=entry.training_seed,
            evaluation_run_id=entry.report.evaluation_run_id,
            checkpoint_sha256=entry.report.checkpoint.checkpoint_sha256,
            report_sha256=entry.report.report_sha256,
        )
        for entry in frozen
    )
    return PairedPanel.build(
        benchmark_descriptor_sha256=public.descriptor.digest,
        public_items_sha256=public.raw_sha256,
        public_item_set_sha256=public.item_set_sha256,
        sealed_references_sha256=public.descriptor.sealed_references_sha256,
        generation_protocol_sha256=generation_protocol.digest,
        evaluator_contract_sha256=first.evaluator_contract.digest,
        evaluator_version_sha256=first.evaluator_version_sha256,
        stratum_key=stratum_key,
        training_seeds=PREREGISTERED_TRAINING_SEEDS,
        report_refs=references,
        items=paired_items,
    )


def _load_statistics_protocol_with_raw_sha256(
    path: str | Path,
) -> tuple[str, StatisticsProtocol]:
    raw_sha256, parsed = _read_json_object(path, field="statistics protocol")
    return raw_sha256, StatisticsProtocol.from_mapping(parsed)


def load_statistics_protocol(path: str | Path) -> StatisticsProtocol:
    _, protocol = _load_statistics_protocol_with_raw_sha256(path)
    return protocol


def write_statistics_protocol(protocol: StatisticsProtocol, path: str | Path) -> None:
    write_json_atomic(protocol.to_record(), path)


def _load_paired_panel_with_raw_sha256(path: str | Path) -> tuple[str, PairedPanel]:
    raw_sha256, parsed = _read_json_object(path, field="paired panel")
    return raw_sha256, PairedPanel.from_mapping(parsed)


def load_paired_panel(path: str | Path) -> PairedPanel:
    _, panel = _load_paired_panel_with_raw_sha256(path)
    return panel


def write_paired_panel(panel: PairedPanel, path: str | Path) -> None:
    write_json_atomic(panel.to_record(), path)


__all__ = [
    "CONFIDENCE_INTERVAL_SCHEMA_VERSION",
    "PAIRED_PANEL_SCHEMA_VERSION",
    "PREREGISTERED_ALPHA_PPM",
    "PREREGISTERED_BOOTSTRAP_REPETITIONS",
    "PREREGISTERED_BOOTSTRAP_SEED",
    "PREREGISTERED_CONFIDENCE_PPM",
    "PREREGISTERED_EFFECT_PPM",
    "PREREGISTERED_EQUIVALENCE_CONFIDENCE_PPM",
    "PREREGISTERED_EQUIVALENCE_MARGIN_PPM",
    "PREREGISTERED_RANDOMIZATION_REPETITIONS",
    "PREREGISTERED_RANDOMIZATION_SEED",
    "PREREGISTERED_TRAINING_SEEDS",
    "RATIONAL_VALUE_SCHEMA_VERSION",
    "SCORE_SCALE",
    "STATISTICS_PROTOCOL_SCHEMA_VERSION",
    "ArmId",
    "ArmSeedEvaluation",
    "ConfidenceInterval",
    "PairedItem",
    "PairedPanel",
    "PanelReportRef",
    "RationalValue",
    "StatisticsContractError",
    "StatisticsProtocol",
    "build_paired_panel",
    "load_paired_panel",
    "load_statistics_protocol",
    "write_paired_panel",
    "write_statistics_protocol",
]
