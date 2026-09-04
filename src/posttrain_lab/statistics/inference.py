"""Exact bookkeeping and deterministic item-level inference for D08."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np

from posttrain_lab.data import canonical_json_bytes, write_json_atomic

from .contracts import (
    SCORE_SCALE,
    ArmId,
    ConfidenceInterval,
    PairedPanel,
    RationalValue,
    StatisticsContractError,
    StatisticsProtocol,
    _read_json_object,
    _require_exact_keys,
    _require_identifier,
    _require_int,
    _require_mapping,
    _require_sha256,
)

SEED_EFFECT_SCHEMA_VERSION = "d08-seed-effect-v1"
RANDOMIZATION_EVIDENCE_SCHEMA_VERSION = "d08-randomization-evidence-v1"
C1_RESULT_SCHEMA_VERSION = "d08-c1-result-v1"
C2_RESULT_SCHEMA_VERSION = "d08-c2-result-v1"
CONFIRMATORY_ANALYSIS_SCHEMA_VERSION = "d08-confirmatory-analysis-v1"

_BOOTSTRAP_BATCH_SIZE = 1_024
_RANDOMIZATION_BATCH_SIZE = 4_096
_MAX_RESAMPLE_CELLS = 1_000_000
_MAX_EXACT_SIGN_FLIP_ITEMS = 20


class C2Classification(StrEnum):
    SUPERIOR_A3 = "superior_A3"
    SUPERIOR_A4 = "superior_A4"
    PRACTICAL_EQUIVALENCE = "practical_equivalence"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class SeedEffect:
    training_seed: int
    effect: RationalValue
    schema_version: str = SEED_EFFECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEED_EFFECT_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported seed-effect schema")
        _require_int(self.training_seed, field="training_seed", minimum=0, maximum=2**63 - 1)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "training_seed": self.training_seed,
            "effect": self.effect.to_record(),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, field: str) -> SeedEffect:
        _require_exact_keys(
            raw,
            required={"schema_version", "training_seed", "effect"},
            field=field,
        )
        return cls(
            schema_version=raw["schema_version"],
            training_seed=raw["training_seed"],
            effect=RationalValue.from_mapping(
                _require_mapping(raw["effect"], field=f"{field}.effect"),
                field=f"{field}.effect",
            ),
        )


@dataclass(frozen=True, slots=True)
class RandomizationEvidence:
    alternative: Literal["greater", "two-sided"]
    repetitions: int
    extreme_count: int
    p_value: RationalValue
    stream_sha256: str
    correction: str = "plus-one"
    schema_version: str = RANDOMIZATION_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RANDOMIZATION_EVIDENCE_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported randomization-evidence schema")
        if self.alternative not in {"greater", "two-sided"}:
            raise StatisticsContractError("unsupported randomization alternative")
        _require_int(self.repetitions, field="repetitions", minimum=1)
        _require_int(
            self.extreme_count,
            field="extreme_count",
            minimum=0,
            maximum=self.repetitions,
        )
        _require_sha256(self.stream_sha256, field="stream_sha256")
        if self.correction != "plus-one":
            raise StatisticsContractError("randomization evidence must use plus-one correction")
        expected = Fraction(self.extreme_count + 1, self.repetitions + 1)
        if self.p_value.fraction != expected:
            raise StatisticsContractError("randomization p-value does not match extreme count")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "alternative": self.alternative,
            "repetitions": self.repetitions,
            "extreme_count": self.extreme_count,
            "p_value": self.p_value.to_record(),
            "stream_sha256": self.stream_sha256,
            "correction": self.correction,
        }

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
        *,
        field: str,
    ) -> RandomizationEvidence:
        _require_exact_keys(
            raw,
            required={
                "schema_version",
                "alternative",
                "repetitions",
                "extreme_count",
                "p_value",
                "stream_sha256",
                "correction",
            },
            field=field,
        )
        return cls(
            schema_version=raw["schema_version"],
            alternative=raw["alternative"],
            repetitions=raw["repetitions"],
            extreme_count=raw["extreme_count"],
            p_value=RationalValue.from_mapping(
                _require_mapping(raw["p_value"], field=f"{field}.p_value"),
                field=f"{field}.p_value",
            ),
            stream_sha256=raw["stream_sha256"],
            correction=raw["correction"],
        )


@dataclass(frozen=True, slots=True)
class C1Result:
    hypothesis_id: str
    treatment_arm: ArmId
    control_arm: ArmId
    point_estimate: RationalValue
    seed_effects: tuple[SeedEffect, ...]
    seed_instability: bool
    bootstrap_interval: ConfidenceInterval
    bootstrap_stream_sha256: str
    randomization: RandomizationEvidence
    holm_adjusted_p_value: RationalValue
    statistical_superiority: bool
    practical_success: bool
    schema_version: str = C1_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != C1_RESULT_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported C1-result schema")
        _require_identifier(self.hypothesis_id, field="hypothesis_id")
        if self.control_arm is not ArmId.A0 or self.treatment_arm not in {ArmId.A1, ArmId.A2}:
            raise StatisticsContractError("C1 result must compare A1/A2 against A0")
        expected_treatment = {"C1a": ArmId.A1, "C1b": ArmId.A2}.get(self.hypothesis_id)
        if expected_treatment is None or self.treatment_arm is not expected_treatment:
            raise StatisticsContractError("C1 hypothesis ID does not match its frozen contrast")
        if not isinstance(self.seed_instability, bool):
            raise StatisticsContractError("seed_instability must be Boolean")
        _require_sha256(self.bootstrap_stream_sha256, field="bootstrap_stream_sha256")
        if not 0 <= self.holm_adjusted_p_value.fraction <= 1:
            raise StatisticsContractError("Holm-adjusted p-value must be in [0, 1]")
        expected_instability = _has_seed_instability(
            tuple(effect.effect.fraction for effect in self.seed_effects)
        )
        if tuple(effect.training_seed for effect in self.seed_effects) != (101, 202, 303):
            raise StatisticsContractError("C1 seed effects must cover 101, 202, 303")
        if self.seed_instability != expected_instability:
            raise StatisticsContractError("seed_instability disagrees with per-seed effects")
        expected_point = sum(
            (effect.effect.fraction for effect in self.seed_effects),
            start=Fraction(0),
        ) / len(self.seed_effects)
        if self.point_estimate.fraction != expected_point:
            raise StatisticsContractError("C1 point estimate disagrees with per-seed effects")
        if self.randomization.alternative != "greater":
            raise StatisticsContractError("C1 randomization must use greater alternative")
        for field in ("statistical_superiority", "practical_success"):
            if not isinstance(getattr(self, field), bool):
                raise StatisticsContractError(f"{field} must be Boolean")
        if self.practical_success and not self.statistical_superiority:
            raise StatisticsContractError("practical success requires statistical superiority")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "treatment_arm": self.treatment_arm.value,
            "control_arm": self.control_arm.value,
            "point_estimate": self.point_estimate.to_record(),
            "seed_effects": [effect.to_record() for effect in self.seed_effects],
            "seed_instability": self.seed_instability,
            "bootstrap_interval": self.bootstrap_interval.to_record(),
            "bootstrap_stream_sha256": self.bootstrap_stream_sha256,
            "randomization": self.randomization.to_record(),
            "holm_adjusted_p_value": self.holm_adjusted_p_value.to_record(),
            "statistical_superiority": self.statistical_superiority,
            "practical_success": self.practical_success,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, field: str) -> C1Result:
        _require_exact_keys(
            raw,
            required={
                "schema_version",
                "hypothesis_id",
                "treatment_arm",
                "control_arm",
                "point_estimate",
                "seed_effects",
                "seed_instability",
                "bootstrap_interval",
                "bootstrap_stream_sha256",
                "randomization",
                "holm_adjusted_p_value",
                "statistical_superiority",
                "practical_success",
            },
            field=field,
        )
        seeds = raw["seed_effects"]
        if not isinstance(seeds, list):
            raise StatisticsContractError(f"{field}.seed_effects must be an array")
        try:
            treatment = ArmId(raw["treatment_arm"])
            control = ArmId(raw["control_arm"])
        except (TypeError, ValueError) as error:
            raise StatisticsContractError(f"{field} has an invalid arm") from error
        return cls(
            schema_version=raw["schema_version"],
            hypothesis_id=raw["hypothesis_id"],
            treatment_arm=treatment,
            control_arm=control,
            point_estimate=RationalValue.from_mapping(
                _require_mapping(raw["point_estimate"], field=f"{field}.point_estimate"),
                field=f"{field}.point_estimate",
            ),
            seed_effects=tuple(
                SeedEffect.from_mapping(
                    _require_mapping(value, field=f"{field}.seed_effects[{index}]"),
                    field=f"{field}.seed_effects[{index}]",
                )
                for index, value in enumerate(seeds)
            ),
            seed_instability=raw["seed_instability"],
            bootstrap_interval=ConfidenceInterval.from_mapping(
                _require_mapping(
                    raw["bootstrap_interval"],
                    field=f"{field}.bootstrap_interval",
                ),
                field=f"{field}.bootstrap_interval",
            ),
            bootstrap_stream_sha256=raw["bootstrap_stream_sha256"],
            randomization=RandomizationEvidence.from_mapping(
                _require_mapping(raw["randomization"], field=f"{field}.randomization"),
                field=f"{field}.randomization",
            ),
            holm_adjusted_p_value=RationalValue.from_mapping(
                _require_mapping(
                    raw["holm_adjusted_p_value"],
                    field=f"{field}.holm_adjusted_p_value",
                ),
                field=f"{field}.holm_adjusted_p_value",
            ),
            statistical_superiority=raw["statistical_superiority"],
            practical_success=raw["practical_success"],
        )


@dataclass(frozen=True, slots=True)
class C2Result:
    hypothesis_id: str
    treatment_arm: ArmId
    control_arm: ArmId
    point_estimate: RationalValue
    seed_effects: tuple[SeedEffect, ...]
    seed_instability: bool
    superiority_interval: ConfidenceInterval
    equivalence_interval: ConfidenceInterval
    bootstrap_stream_sha256: str
    randomization: RandomizationEvidence
    equivalence_assessed: bool
    lower_margin_pass: bool | None
    upper_margin_pass: bool | None
    classification: C2Classification
    schema_version: str = C2_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != C2_RESULT_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported C2-result schema")
        if self.hypothesis_id != "C2":
            raise StatisticsContractError("C2 hypothesis_id must be C2")
        if self.treatment_arm is not ArmId.A3 or self.control_arm is not ArmId.A4:
            raise StatisticsContractError("C2 must compare A3-A4")
        if not isinstance(self.seed_instability, bool):
            raise StatisticsContractError("seed_instability must be Boolean")
        _require_sha256(self.bootstrap_stream_sha256, field="bootstrap_stream_sha256")
        expected_instability = _has_seed_instability(
            tuple(effect.effect.fraction for effect in self.seed_effects)
        )
        if tuple(effect.training_seed for effect in self.seed_effects) != (101, 202, 303):
            raise StatisticsContractError("C2 seed effects must cover 101, 202, 303")
        if self.seed_instability != expected_instability:
            raise StatisticsContractError("seed_instability disagrees with per-seed effects")
        expected_point = sum(
            (effect.effect.fraction for effect in self.seed_effects),
            start=Fraction(0),
        ) / len(self.seed_effects)
        if self.point_estimate.fraction != expected_point:
            raise StatisticsContractError("C2 point estimate disagrees with per-seed effects")
        if self.randomization.alternative != "two-sided":
            raise StatisticsContractError("C2 randomization must use two-sided alternative")
        if not isinstance(self.equivalence_assessed, bool):
            raise StatisticsContractError("equivalence_assessed must be Boolean")
        if self.equivalence_assessed:
            if not isinstance(self.lower_margin_pass, bool) or not isinstance(
                self.upper_margin_pass, bool
            ):
                raise StatisticsContractError("assessed TOST margins must be Boolean")
        elif self.lower_margin_pass is not None or self.upper_margin_pass is not None:
            raise StatisticsContractError("unassessed TOST margins must be null")
        if self.classification in {
            C2Classification.SUPERIOR_A3,
            C2Classification.SUPERIOR_A4,
        } and self.equivalence_assessed:
            raise StatisticsContractError("superiority result must not also assess equivalence")
        if self.classification in {
            C2Classification.PRACTICAL_EQUIVALENCE,
            C2Classification.INCONCLUSIVE,
        } and not self.equivalence_assessed:
            raise StatisticsContractError("non-superiority result must assess equivalence")
        if self.classification is C2Classification.PRACTICAL_EQUIVALENCE and not (
            self.lower_margin_pass and self.upper_margin_pass
        ):
            raise StatisticsContractError("practical equivalence requires both margin passes")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "treatment_arm": self.treatment_arm.value,
            "control_arm": self.control_arm.value,
            "point_estimate": self.point_estimate.to_record(),
            "seed_effects": [effect.to_record() for effect in self.seed_effects],
            "seed_instability": self.seed_instability,
            "superiority_interval": self.superiority_interval.to_record(),
            "equivalence_interval": self.equivalence_interval.to_record(),
            "bootstrap_stream_sha256": self.bootstrap_stream_sha256,
            "randomization": self.randomization.to_record(),
            "equivalence_assessed": self.equivalence_assessed,
            "lower_margin_pass": self.lower_margin_pass,
            "upper_margin_pass": self.upper_margin_pass,
            "classification": self.classification.value,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, field: str) -> C2Result:
        _require_exact_keys(
            raw,
            required={
                "schema_version",
                "hypothesis_id",
                "treatment_arm",
                "control_arm",
                "point_estimate",
                "seed_effects",
                "seed_instability",
                "superiority_interval",
                "equivalence_interval",
                "bootstrap_stream_sha256",
                "randomization",
                "equivalence_assessed",
                "lower_margin_pass",
                "upper_margin_pass",
                "classification",
            },
            field=field,
        )
        seeds = raw["seed_effects"]
        if not isinstance(seeds, list):
            raise StatisticsContractError(f"{field}.seed_effects must be an array")
        try:
            treatment = ArmId(raw["treatment_arm"])
            control = ArmId(raw["control_arm"])
            classification = C2Classification(raw["classification"])
        except (TypeError, ValueError) as error:
            raise StatisticsContractError(f"{field} has an invalid enum value") from error
        return cls(
            schema_version=raw["schema_version"],
            hypothesis_id=raw["hypothesis_id"],
            treatment_arm=treatment,
            control_arm=control,
            point_estimate=RationalValue.from_mapping(
                _require_mapping(raw["point_estimate"], field=f"{field}.point_estimate"),
                field=f"{field}.point_estimate",
            ),
            seed_effects=tuple(
                SeedEffect.from_mapping(
                    _require_mapping(value, field=f"{field}.seed_effects[{index}]"),
                    field=f"{field}.seed_effects[{index}]",
                )
                for index, value in enumerate(seeds)
            ),
            seed_instability=raw["seed_instability"],
            superiority_interval=ConfidenceInterval.from_mapping(
                _require_mapping(
                    raw["superiority_interval"],
                    field=f"{field}.superiority_interval",
                ),
                field=f"{field}.superiority_interval",
            ),
            equivalence_interval=ConfidenceInterval.from_mapping(
                _require_mapping(
                    raw["equivalence_interval"],
                    field=f"{field}.equivalence_interval",
                ),
                field=f"{field}.equivalence_interval",
            ),
            bootstrap_stream_sha256=raw["bootstrap_stream_sha256"],
            randomization=RandomizationEvidence.from_mapping(
                _require_mapping(raw["randomization"], field=f"{field}.randomization"),
                field=f"{field}.randomization",
            ),
            equivalence_assessed=raw["equivalence_assessed"],
            lower_margin_pass=raw["lower_margin_pass"],
            upper_margin_pass=raw["upper_margin_pass"],
            classification=classification,
        )


@dataclass(frozen=True, slots=True)
class ConfirmatoryAnalysisReport:
    panel_sha256: str
    statistics_protocol_sha256: str
    resampling_design_sha256: str
    item_count: int
    strata_counts: tuple[tuple[str, int], ...]
    c1_results: tuple[C1Result, C1Result]
    c2_result: C2Result
    analysis_report_sha256: str
    inference_population: str = "fixed-three-training-seeds_item-population"
    resampling_unit: str = "item-with-complete-three-seed-vector"
    schema_version: str = CONFIRMATORY_ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIRMATORY_ANALYSIS_SCHEMA_VERSION:
            raise StatisticsContractError("unsupported confirmatory-analysis schema")
        _require_sha256(self.panel_sha256, field="panel_sha256")
        _require_sha256(self.statistics_protocol_sha256, field="statistics_protocol_sha256")
        _require_sha256(self.resampling_design_sha256, field="resampling_design_sha256")
        _require_sha256(self.analysis_report_sha256, field="analysis_report_sha256")
        _require_int(self.item_count, field="item_count", minimum=1)
        if self.inference_population != "fixed-three-training-seeds_item-population":
            raise StatisticsContractError("unsupported inference population")
        if self.resampling_unit != "item-with-complete-three-seed-vector":
            raise StatisticsContractError("unsupported resampling unit")
        if tuple(identifier.hypothesis_id for identifier in self.c1_results) != ("C1a", "C1b"):
            raise StatisticsContractError("C1 family must contain C1a then C1b")
        if tuple(sorted(self.strata_counts)) != self.strata_counts:
            raise StatisticsContractError("strata_counts must be sorted")
        if sum(count for _, count in self.strata_counts) != self.item_count:
            raise StatisticsContractError("strata_counts do not sum to item_count")
        expected = hashlib.sha256(canonical_json_bytes(self.unsigned_record())).hexdigest()
        if self.analysis_report_sha256 != expected:
            raise StatisticsContractError("analysis_report_sha256 mismatch")

    @classmethod
    def build(
        cls,
        *,
        panel: PairedPanel,
        protocol: StatisticsProtocol,
        c1_results: tuple[C1Result, C1Result],
        c2_result: C2Result,
    ) -> ConfirmatoryAnalysisReport:
        strata_counts = tuple(sorted(Counter(item.stratum for item in panel.items).items()))
        unsigned = {
            "schema_version": CONFIRMATORY_ANALYSIS_SCHEMA_VERSION,
            "panel_sha256": panel.panel_sha256,
            "statistics_protocol_sha256": protocol.digest,
            "resampling_design_sha256": resampling_design_sha256(panel),
            "item_count": len(panel.items),
            "strata_counts": dict(strata_counts),
            "inference_population": "fixed-three-training-seeds_item-population",
            "resampling_unit": "item-with-complete-three-seed-vector",
            "c1_results": [result.to_record() for result in c1_results],
            "c2_result": c2_result.to_record(),
        }
        return cls(
            panel_sha256=panel.panel_sha256,
            statistics_protocol_sha256=protocol.digest,
            resampling_design_sha256=resampling_design_sha256(panel),
            item_count=len(panel.items),
            strata_counts=strata_counts,
            c1_results=c1_results,
            c2_result=c2_result,
            analysis_report_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        )

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "panel_sha256": self.panel_sha256,
            "statistics_protocol_sha256": self.statistics_protocol_sha256,
            "resampling_design_sha256": self.resampling_design_sha256,
            "item_count": self.item_count,
            "strata_counts": dict(self.strata_counts),
            "inference_population": self.inference_population,
            "resampling_unit": self.resampling_unit,
            "c1_results": [result.to_record() for result in self.c1_results],
            "c2_result": self.c2_result.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "analysis_report_sha256": self.analysis_report_sha256}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ConfirmatoryAnalysisReport:
        _require_exact_keys(
            raw,
            required={
                "schema_version",
                "panel_sha256",
                "statistics_protocol_sha256",
                "resampling_design_sha256",
                "item_count",
                "strata_counts",
                "inference_population",
                "resampling_unit",
                "c1_results",
                "c2_result",
                "analysis_report_sha256",
            },
            field="confirmatory analysis",
        )
        strata = _require_mapping(raw["strata_counts"], field="strata_counts")
        c1 = raw["c1_results"]
        if not isinstance(c1, list) or len(c1) != 2:
            raise StatisticsContractError("c1_results must contain exactly two entries")
        counts: list[tuple[str, int]] = []
        for key, value in strata.items():
            _require_identifier(key, field="strata_counts key")
            counts.append((key, _require_int(value, field=f"strata_counts.{key}", minimum=1)))
        return cls(
            schema_version=raw["schema_version"],
            panel_sha256=raw["panel_sha256"],
            statistics_protocol_sha256=raw["statistics_protocol_sha256"],
            resampling_design_sha256=raw["resampling_design_sha256"],
            item_count=raw["item_count"],
            strata_counts=tuple(counts),
            inference_population=raw["inference_population"],
            resampling_unit=raw["resampling_unit"],
            c1_results=(
                C1Result.from_mapping(
                    _require_mapping(c1[0], field="c1_results[0]"),
                    field="c1_results[0]",
                ),
                C1Result.from_mapping(
                    _require_mapping(c1[1], field="c1_results[1]"),
                    field="c1_results[1]",
                ),
            ),
            c2_result=C2Result.from_mapping(
                _require_mapping(raw["c2_result"], field="c2_result"),
                field="c2_result",
            ),
            analysis_report_sha256=raw["analysis_report_sha256"],
        )


def paired_difference_matrix(
    panel: PairedPanel,
    *,
    treatment: ArmId,
    control: ArmId,
) -> np.ndarray:
    """Return item x fixed-training-seed correctness differences in {-1,0,1}."""

    if treatment is control:
        raise StatisticsContractError("paired contrast requires different arms")
    return np.asarray(
        [
            [
                int(t) - int(c)
                for t, c in zip(
                    item.for_arm(treatment),
                    item.for_arm(control),
                    strict=True,
                )
            ]
            for item in panel.items
        ],
        dtype=np.int8,
    )


def paired_effect(differences: np.ndarray) -> Fraction:
    frozen = _validate_differences(differences)
    return Fraction(int(frozen.sum(dtype=np.int64)), frozen.size)


def per_seed_effects(differences: np.ndarray) -> tuple[Fraction, ...]:
    frozen = _validate_differences(differences)
    return tuple(
        Fraction(int(frozen[:, index].sum()), frozen.shape[0])
        for index in range(frozen.shape[1])
    )


def _validate_differences(differences: np.ndarray) -> np.ndarray:
    if not isinstance(differences, np.ndarray) or differences.ndim != 2 or not differences.size:
        raise StatisticsContractError("differences must be a non-empty item x seed ndarray")
    if differences.dtype.kind not in {"i", "u"}:
        raise StatisticsContractError("differences must use an integer dtype")
    if np.any((differences < -1) | (differences > 1)):
        raise StatisticsContractError("paired correctness differences must be in {-1,0,1}")
    return differences


def _derived_stream(
    *,
    base_seed: int,
    operation: str,
    hypothesis_id: str,
    resampling_design_sha256: str,
    protocol_sha256: str,
) -> tuple[str, int]:
    payload = {
        "schema_version": "d08-derived-rng-stream-v1",
        "base_seed": base_seed,
        "operation": operation,
        "hypothesis_id": hypothesis_id,
        "resampling_design_sha256": resampling_design_sha256,
        "statistics_protocol_sha256": protocol_sha256,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return digest, int(digest[:32], 16)


def resampling_design_sha256(panel: PairedPanel) -> str:
    """Hash only frozen, outcome-independent fields used to derive RNG streams."""

    payload = {
        "schema_version": "d08-resampling-design-v1",
        "benchmark_descriptor_sha256": panel.benchmark_descriptor_sha256,
        "public_items_sha256": panel.public_items_sha256,
        "public_item_set_sha256": panel.public_item_set_sha256,
        "sealed_references_sha256": panel.sealed_references_sha256,
        "generation_protocol_sha256": panel.generation_protocol_sha256,
        "evaluator_contract_sha256": panel.evaluator_contract_sha256,
        "evaluator_version_sha256": panel.evaluator_version_sha256,
        "stratum_key": panel.stratum_key,
        "training_seeds": list(panel.training_seeds),
        "items": [
            {
                "item_id": item.item_id,
                "item_index": item.item_index,
                "stratum": item.stratum,
            }
            for item in panel.items
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stratified_item_bootstrap_totals(
    differences: np.ndarray,
    strata: Sequence[str],
    *,
    repetitions: int,
    rng_seed: int,
) -> np.ndarray:
    """Bootstrap items within strata, carrying each item's full seed vector."""

    frozen = _validate_differences(differences)
    _require_int(repetitions, field="repetitions", minimum=1, maximum=10_000_000)
    _require_int(rng_seed, field="rng_seed", minimum=0, maximum=2**128 - 1)
    if len(strata) != frozen.shape[0] or any(
        not isinstance(value, str) or not value for value in strata
    ):
        raise StatisticsContractError("strata must contain one non-empty label per item")
    groups: dict[str, np.ndarray] = {}
    strata_array = np.asarray(strata, dtype=object)
    for stratum in sorted(set(strata)):
        groups[stratum] = np.flatnonzero(strata_array == stratum)
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    item_units = frozen.sum(axis=1, dtype=np.int64)
    totals = np.empty(repetitions, dtype=np.int64)
    largest_stratum = max(len(indices) for indices in groups.values())
    batch_size = min(
        _BOOTSTRAP_BATCH_SIZE,
        max(1, _MAX_RESAMPLE_CELLS // largest_stratum),
    )
    for start in range(0, repetitions, batch_size):
        stop = min(repetitions, start + batch_size)
        batch_total = np.zeros(stop - start, dtype=np.int64)
        for indices in groups.values():
            draws = rng.integers(
                0,
                len(indices),
                size=(stop - start, len(indices)),
                dtype=np.int64,
            )
            batch_total += item_units[indices[draws]].sum(axis=1, dtype=np.int64)
        totals[start:stop] = batch_total
    return totals


def exact_linear_quantile(values: Sequence[int] | np.ndarray, probability: Fraction) -> Fraction:
    """Hyndman-Fan type-7 quantile, evaluated exactly over integer samples."""

    if not 0 <= probability <= 1:
        raise StatisticsContractError("quantile probability must be in [0,1]")
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in {"i", "u"}:
        raise StatisticsContractError("quantile values must be a non-empty integer vector")
    ordered = np.sort(array.astype(np.int64, copy=False))
    position = Fraction(len(ordered) - 1) * probability
    lower_index = position.numerator // position.denominator
    remainder = position - lower_index
    upper_index = min(lower_index + 1, len(ordered) - 1)
    lower = int(ordered[lower_index])
    upper = int(ordered[upper_index])
    return Fraction(lower) + remainder * (upper - lower)


def percentile_interval(
    bootstrap_totals: Sequence[int] | np.ndarray,
    *,
    denominator: int,
    confidence_ppm: int,
) -> ConfidenceInterval:
    _require_int(denominator, field="denominator", minimum=1)
    _require_int(
        confidence_ppm,
        field="confidence_ppm",
        minimum=1,
        maximum=SCORE_SCALE - 1,
    )
    tail = Fraction(SCORE_SCALE - confidence_ppm, 2 * SCORE_SCALE)
    lower = exact_linear_quantile(bootstrap_totals, tail) / denominator
    upper = exact_linear_quantile(bootstrap_totals, 1 - tail) / denominator
    return ConfidenceInterval(
        confidence_ppm=confidence_ppm,
        lower=RationalValue.from_fraction(lower),
        upper=RationalValue.from_fraction(upper),
    )


def paired_sign_flip_randomization(
    differences: np.ndarray,
    *,
    repetitions: int,
    rng_seed: int,
    alternative: Literal["greater", "two-sided"],
) -> tuple[int, Fraction]:
    """Monte Carlo item-level sign flip with a conservative plus-one p-value."""

    frozen = _validate_differences(differences)
    _require_int(repetitions, field="repetitions", minimum=1, maximum=10_000_000)
    _require_int(rng_seed, field="rng_seed", minimum=0, maximum=2**128 - 1)
    if alternative not in {"greater", "two-sided"}:
        raise StatisticsContractError("alternative must be greater or two-sided")
    item_units = frozen.sum(axis=1, dtype=np.int64)
    observed = int(item_units.sum())
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    extreme = 0
    batch_size = min(
        _RANDOMIZATION_BATCH_SIZE,
        max(1, _MAX_RESAMPLE_CELLS // len(item_units)),
    )
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        bits = rng.integers(0, 2, size=(size, len(item_units)), dtype=np.int8)
        signed_totals = (bits.astype(np.int64) * 2 - 1) @ item_units
        if alternative == "greater":
            extreme += int(np.count_nonzero(signed_totals >= observed))
        else:
            extreme += int(np.count_nonzero(np.abs(signed_totals) >= abs(observed)))
    return extreme, Fraction(extreme + 1, repetitions + 1)


def exact_sign_flip_p_value(
    differences: np.ndarray,
    *,
    alternative: Literal["greater", "two-sided"],
) -> Fraction:
    """Exhaustive reference oracle for at most 20 item vectors."""

    frozen = _validate_differences(differences)
    if frozen.shape[0] > _MAX_EXACT_SIGN_FLIP_ITEMS:
        raise StatisticsContractError(
            f"exact sign flip is limited to {_MAX_EXACT_SIGN_FLIP_ITEMS} items"
        )
    if alternative not in {"greater", "two-sided"}:
        raise StatisticsContractError("alternative must be greater or two-sided")
    item_units = [int(value) for value in frozen.sum(axis=1, dtype=np.int64)]
    observed = sum(item_units)
    extreme = 0
    assignments = 1 << len(item_units)
    for mask in range(assignments):
        permuted = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(item_units)
        )
        if alternative == "greater":
            extreme += permuted >= observed
        else:
            extreme += abs(permuted) >= abs(observed)
    return Fraction(extreme, assignments)


def holm_adjust(p_values: Mapping[str, Fraction]) -> dict[str, Fraction]:
    """Return monotone Holm step-down adjusted p-values using exact arithmetic."""

    if not p_values:
        raise StatisticsContractError("Holm family must be non-empty")
    if any(not isinstance(key, str) or not key for key in p_values):
        raise StatisticsContractError("Holm hypothesis identifiers must be non-empty strings")
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise StatisticsContractError("Holm raw p-values must be in [0,1]")
    ordered = sorted(p_values.items(), key=lambda pair: (pair[1], pair[0]))
    adjusted: dict[str, Fraction] = {}
    running = Fraction(0)
    family_size = len(ordered)
    for rank, (identifier, p_value) in enumerate(ordered):
        candidate = min(Fraction(1), (family_size - rank) * p_value)
        running = max(running, candidate)
        adjusted[identifier] = running
    return adjusted


def _has_seed_instability(effects: Sequence[Fraction]) -> bool:
    return any(effect < 0 for effect in effects) and any(effect > 0 for effect in effects)


@dataclass(frozen=True, slots=True)
class _ContrastEvidence:
    point: Fraction
    seed_effects: tuple[Fraction, ...]
    confidence_interval: ConfidenceInterval
    equivalence_interval: ConfidenceInterval
    bootstrap_stream_sha256: str
    randomization: RandomizationEvidence


def _analyze_contrast(
    panel: PairedPanel,
    protocol: StatisticsProtocol,
    *,
    hypothesis_id: str,
    treatment: ArmId,
    control: ArmId,
    alternative: Literal["greater", "two-sided"],
) -> _ContrastEvidence:
    differences = paired_difference_matrix(panel, treatment=treatment, control=control)
    point = paired_effect(differences)
    seed_effect_values = per_seed_effects(differences)
    bootstrap_stream, bootstrap_seed = _derived_stream(
        base_seed=protocol.bootstrap_seed,
        operation="stratified-item-bootstrap",
        hypothesis_id=hypothesis_id,
        resampling_design_sha256=resampling_design_sha256(panel),
        protocol_sha256=protocol.digest,
    )
    totals = stratified_item_bootstrap_totals(
        differences,
        tuple(item.stratum for item in panel.items),
        repetitions=protocol.bootstrap_repetitions,
        rng_seed=bootstrap_seed,
    )
    denominator = differences.size
    interval = percentile_interval(
        totals,
        denominator=denominator,
        confidence_ppm=protocol.confidence_ppm,
    )
    equivalence_interval = percentile_interval(
        totals,
        denominator=denominator,
        confidence_ppm=protocol.equivalence_confidence_ppm,
    )
    randomization_stream, randomization_seed = _derived_stream(
        base_seed=protocol.randomization_seed,
        operation="paired-item-sign-flip",
        hypothesis_id=hypothesis_id,
        resampling_design_sha256=resampling_design_sha256(panel),
        protocol_sha256=protocol.digest,
    )
    extreme, p_value = paired_sign_flip_randomization(
        differences,
        repetitions=protocol.randomization_repetitions,
        rng_seed=randomization_seed,
        alternative=alternative,
    )
    return _ContrastEvidence(
        point=point,
        seed_effects=seed_effect_values,
        confidence_interval=interval,
        equivalence_interval=equivalence_interval,
        bootstrap_stream_sha256=bootstrap_stream,
        randomization=RandomizationEvidence(
            alternative=alternative,
            repetitions=protocol.randomization_repetitions,
            extreme_count=extreme,
            p_value=RationalValue.from_fraction(p_value),
            stream_sha256=randomization_stream,
        ),
    )


def _seed_effect_records(
    seeds: Sequence[int], effects: Sequence[Fraction]
) -> tuple[SeedEffect, ...]:
    return tuple(
        SeedEffect(training_seed=seed, effect=RationalValue.from_fraction(effect))
        for seed, effect in zip(seeds, effects, strict=True)
    )


def _classify_c2(
    evidence: _ContrastEvidence,
    *,
    practical_effect: Fraction,
    equivalence_margin: Fraction,
) -> tuple[bool, bool | None, bool | None, C2Classification]:
    point = evidence.point
    ci95 = evidence.confidence_interval
    ci90 = evidence.equivalence_interval
    if ci95.lower.fraction > 0 and point >= practical_effect:
        return False, None, None, C2Classification.SUPERIOR_A3
    if ci95.upper.fraction < 0 and point <= -practical_effect:
        return False, None, None, C2Classification.SUPERIOR_A4
    lower_pass = ci90.lower.fraction > -equivalence_margin
    upper_pass = ci90.upper.fraction < equivalence_margin
    if lower_pass and upper_pass:
        return True, True, True, C2Classification.PRACTICAL_EQUIVALENCE
    return True, lower_pass, upper_pass, C2Classification.INCONCLUSIVE


def run_confirmatory_analysis(
    panel: PairedPanel,
    protocol: StatisticsProtocol,
    *,
    require_preregistered: bool = True,
) -> ConfirmatoryAnalysisReport:
    """Run the C1 family and C2 order contrast under the frozen D08 rules."""

    if panel.training_seeds != protocol.training_seeds:
        raise StatisticsContractError("panel and statistics protocol training seeds differ")
    if panel.stratum_key != protocol.stratum_key:
        raise StatisticsContractError("panel and statistics protocol stratum key differ")
    if require_preregistered:
        protocol.assert_preregistered()
    c1_specs = (
        ("C1a", ArmId.A1, ArmId.A0),
        ("C1b", ArmId.A2, ArmId.A0),
    )
    c1_evidence = {
        identifier: _analyze_contrast(
            panel,
            protocol,
            hypothesis_id=identifier,
            treatment=treatment,
            control=control,
            alternative="greater",
        )
        for identifier, treatment, control in c1_specs
    }
    adjusted = holm_adjust(
        {
            identifier: evidence.randomization.p_value.fraction
            for identifier, evidence in c1_evidence.items()
        }
    )
    alpha = Fraction(protocol.family_alpha_ppm, SCORE_SCALE)
    practical = Fraction(protocol.practical_effect_ppm, SCORE_SCALE)
    c1_results: list[C1Result] = []
    for identifier, treatment, control in c1_specs:
        evidence = c1_evidence[identifier]
        statistically_superior = (
            adjusted[identifier] < alpha
            and evidence.confidence_interval.lower.fraction > 0
        )
        c1_results.append(
            C1Result(
                hypothesis_id=identifier,
                treatment_arm=treatment,
                control_arm=control,
                point_estimate=RationalValue.from_fraction(evidence.point),
                seed_effects=_seed_effect_records(
                    protocol.training_seeds,
                    evidence.seed_effects,
                ),
                seed_instability=_has_seed_instability(evidence.seed_effects),
                bootstrap_interval=evidence.confidence_interval,
                bootstrap_stream_sha256=evidence.bootstrap_stream_sha256,
                randomization=evidence.randomization,
                holm_adjusted_p_value=RationalValue.from_fraction(adjusted[identifier]),
                statistical_superiority=statistically_superior,
                practical_success=statistically_superior and evidence.point >= practical,
            )
        )
    c2_evidence = _analyze_contrast(
        panel,
        protocol,
        hypothesis_id="C2",
        treatment=ArmId.A3,
        control=ArmId.A4,
        alternative="two-sided",
    )
    equivalence_assessed, lower_pass, upper_pass, classification = _classify_c2(
        c2_evidence,
        practical_effect=practical,
        equivalence_margin=Fraction(protocol.equivalence_margin_ppm, SCORE_SCALE),
    )
    c2_result = C2Result(
        hypothesis_id="C2",
        treatment_arm=ArmId.A3,
        control_arm=ArmId.A4,
        point_estimate=RationalValue.from_fraction(c2_evidence.point),
        seed_effects=_seed_effect_records(protocol.training_seeds, c2_evidence.seed_effects),
        seed_instability=_has_seed_instability(c2_evidence.seed_effects),
        superiority_interval=c2_evidence.confidence_interval,
        equivalence_interval=c2_evidence.equivalence_interval,
        bootstrap_stream_sha256=c2_evidence.bootstrap_stream_sha256,
        randomization=c2_evidence.randomization,
        equivalence_assessed=equivalence_assessed,
        lower_margin_pass=lower_pass,
        upper_margin_pass=upper_pass,
        classification=classification,
    )
    report = ConfirmatoryAnalysisReport.build(
        panel=panel,
        protocol=protocol,
        c1_results=(c1_results[0], c1_results[1]),
        c2_result=c2_result,
    )
    _validate_decisions(report, protocol)
    return report


def _validate_decisions(
    report: ConfirmatoryAnalysisReport,
    protocol: StatisticsProtocol,
) -> None:
    alpha = Fraction(protocol.family_alpha_ppm, SCORE_SCALE)
    practical = Fraction(protocol.practical_effect_ppm, SCORE_SCALE)
    expected_holm = holm_adjust(
        {
            result.hypothesis_id: result.randomization.p_value.fraction
            for result in report.c1_results
        }
    )
    for result in report.c1_results:
        if result.holm_adjusted_p_value.fraction != expected_holm[result.hypothesis_id]:
            raise StatisticsContractError("C1 adjusted p-value disagrees with Holm family")
        statistical = (
            result.holm_adjusted_p_value.fraction < alpha
            and result.bootstrap_interval.lower.fraction > 0
        )
        if result.statistical_superiority != statistical:
            raise StatisticsContractError("C1 statistical decision disagrees with frozen rule")
        if result.practical_success != (
            statistical and result.point_estimate.fraction >= practical
        ):
            raise StatisticsContractError("C1 practical decision disagrees with frozen rule")
    expected = _classify_c2(
        _ContrastEvidence(
            point=report.c2_result.point_estimate.fraction,
            seed_effects=tuple(effect.effect.fraction for effect in report.c2_result.seed_effects),
            confidence_interval=report.c2_result.superiority_interval,
            equivalence_interval=report.c2_result.equivalence_interval,
            bootstrap_stream_sha256=report.c2_result.bootstrap_stream_sha256,
            randomization=report.c2_result.randomization,
        ),
        practical_effect=practical,
        equivalence_margin=Fraction(protocol.equivalence_margin_ppm, SCORE_SCALE),
    )
    actual = (
        report.c2_result.equivalence_assessed,
        report.c2_result.lower_margin_pass,
        report.c2_result.upper_margin_pass,
        report.c2_result.classification,
    )
    if actual != expected:
        raise StatisticsContractError("C2 decision disagrees with frozen rule")


def validate_analysis_bindings(
    report: ConfirmatoryAnalysisReport,
    *,
    panel: PairedPanel,
    protocol: StatisticsProtocol,
) -> None:
    """Revalidate an already loaded result against its panel and protocol."""

    expected = run_confirmatory_analysis(
        panel,
        protocol,
        require_preregistered=False,
    )
    if report != expected:
        raise StatisticsContractError(
            "analysis report differs from deterministic recomputation over panel and protocol"
        )


def load_confirmatory_analysis(path: str | Path) -> ConfirmatoryAnalysisReport:
    _, parsed = _read_json_object(path, field="confirmatory analysis")
    return ConfirmatoryAnalysisReport.from_mapping(parsed)


def write_confirmatory_analysis(
    report: ConfirmatoryAnalysisReport,
    path: str | Path,
) -> None:
    write_json_atomic(report.to_record(), path)


__all__ = [
    "C1_RESULT_SCHEMA_VERSION",
    "C2_RESULT_SCHEMA_VERSION",
    "CONFIRMATORY_ANALYSIS_SCHEMA_VERSION",
    "RANDOMIZATION_EVIDENCE_SCHEMA_VERSION",
    "SEED_EFFECT_SCHEMA_VERSION",
    "C1Result",
    "C2Classification",
    "C2Result",
    "ConfirmatoryAnalysisReport",
    "RandomizationEvidence",
    "SeedEffect",
    "exact_linear_quantile",
    "exact_sign_flip_p_value",
    "holm_adjust",
    "load_confirmatory_analysis",
    "paired_difference_matrix",
    "paired_effect",
    "paired_sign_flip_randomization",
    "per_seed_effects",
    "percentile_interval",
    "resampling_design_sha256",
    "run_confirmatory_analysis",
    "stratified_item_bootstrap_totals",
    "validate_analysis_bindings",
    "write_confirmatory_analysis",
]
