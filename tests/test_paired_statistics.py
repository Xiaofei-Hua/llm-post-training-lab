from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from posttrain_lab.data import canonical_json_bytes
from posttrain_lab.evaluation import (
    BenchmarkTask,
    CheckpointIdentity,
    EvaluatorContract,
    FinishReason,
    GenerationResponse,
    GenerationStatus,
    evaluate_generation_batch,
    load_benchmark_descriptor,
    load_generation_protocol,
    load_public_benchmark,
    load_sealed_answer_vault,
    run_generation,
)
from posttrain_lab.rewards import ExactMathVerifier
from posttrain_lab.statistics import (
    ArmId,
    ArmSeedEvaluation,
    C2Classification,
    PairedItem,
    PairedPanel,
    PanelReportRef,
    RationalValue,
    StatisticsContractError,
    StatisticsProtocol,
    build_paired_panel,
    exact_linear_quantile,
    exact_sign_flip_p_value,
    holm_adjust,
    load_confirmatory_analysis,
    load_paired_panel,
    paired_difference_matrix,
    paired_effect,
    paired_sign_flip_randomization,
    per_seed_effects,
    percentile_interval,
    resampling_design_sha256,
    run_confirmatory_analysis,
    stratified_item_bootstrap_totals,
    validate_analysis_bindings,
    write_confirmatory_analysis,
    write_paired_panel,
)

EVALUATION_FIXTURE = Path("tests/fixtures/evaluation_contract")
ARMS = (ArmId.A0, ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4)
SEEDS = (101, 202, 303)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def panel_from_correctness(
    correctness: dict[ArmId, list[tuple[bool, bool, bool]]],
    *,
    strata: tuple[str, ...] | None = None,
) -> PairedPanel:
    count = len(correctness[ArmId.A0])
    if strata is None:
        strata = tuple(f"L{index % 2 + 1}" for index in range(count))
    refs = tuple(
        PanelReportRef(
            arm=arm,
            training_seed=seed,
            evaluation_run_id=f"synthetic-{arm.value}-S{seed}",
            checkpoint_sha256=digest(f"checkpoint-{arm.value}-{seed}"),
            report_sha256=digest(f"report-{arm.value}-{seed}"),
        )
        for arm in ARMS
        for seed in SEEDS
    )
    items = tuple(
        PairedItem(
            item_id=f"item-{index:03d}",
            item_index=index,
            stratum=strata[index],
            correctness=tuple((arm, correctness[arm][index]) for arm in ARMS),
        )
        for index in range(count)
    )
    return PairedPanel.build(
        benchmark_descriptor_sha256=digest("descriptor"),
        public_items_sha256=digest("public-items"),
        public_item_set_sha256=digest("item-set"),
        sealed_references_sha256=digest("sealed"),
        generation_protocol_sha256=digest("generation-protocol"),
        evaluator_contract_sha256=digest("evaluator-contract"),
        evaluator_version_sha256=digest("evaluator-version"),
        stratum_key="level",
        training_seeds=SEEDS,
        report_refs=refs,
        items=items,
    )


def decision_fixture_panel() -> PairedPanel:
    count = 8
    false = [(False, False, False)] * count
    true = [(True, True, True)] * count
    alternating = [
        (index % 2 == 0, index % 3 == 0, index % 4 == 0) for index in range(count)
    ]
    return panel_from_correctness(
        {
            ArmId.A0: false,
            ArmId.A1: true,
            ArmId.A2: false,
            ArmId.A3: alternating,
            ArmId.A4: alternating,
        }
    )


def fast_protocol() -> StatisticsProtocol:
    return replace(
        StatisticsProtocol.preregistered(),
        bootstrap_repetitions=2_000,
        randomization_repetitions=5_000,
    )


def test_rational_value_is_exact_signed_and_reduced() -> None:
    assert RationalValue.from_fraction(Fraction(1, 3)).ppm == 333_333
    assert RationalValue.from_fraction(Fraction(-1, 3)).ppm == -333_333
    assert RationalValue.from_fraction(Fraction(1, 2_000_000)).ppm == 1
    assert RationalValue.from_fraction(Fraction(-1, 2_000_000)).ppm == -1
    with pytest.raises(StatisticsContractError, match="reduced"):
        RationalValue(numerator=2, denominator=4, ppm=500_000)
    with pytest.raises(StatisticsContractError, match="integer"):
        RationalValue(numerator=True, denominator=1, ppm=1_000_000)


def test_protocol_preregistration_rejects_quiet_analysis_changes() -> None:
    StatisticsProtocol.preregistered().assert_preregistered()
    with pytest.raises(StatisticsContractError, match="exact preregistered"):
        fast_protocol().assert_preregistered()
    with pytest.raises(StatisticsContractError, match="training_seeds"):
        replace(StatisticsProtocol.preregistered(), training_seeds=(101, 101, 303))
    with pytest.raises(StatisticsContractError, match="RNG"):
        replace(StatisticsProtocol.preregistered(), rng_algorithm="default_rng")


def test_panel_round_trip_and_hash_fail_closed(tmp_path: Path) -> None:
    panel = decision_fixture_panel()
    path = tmp_path / "panel.json"
    write_paired_panel(panel, path)
    assert load_paired_panel(path) == panel

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["correctness"]["A1"][0] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StatisticsContractError, match="panel_sha256 mismatch"):
        load_paired_panel(path)


def test_panel_loader_rejects_extra_keys_and_boolean_seed(tmp_path: Path) -> None:
    panel = decision_fixture_panel()
    payload = panel.to_record()
    payload["unexpected"] = "field"
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StatisticsContractError, match="invalid keys"):
        load_paired_panel(path)

    payload = panel.to_record()
    payload["training_seeds"][0] = True
    unsigned = {key: value for key, value in payload.items() if key != "panel_sha256"}
    payload["panel_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StatisticsContractError, match="seeds"):
        load_paired_panel(path)


def test_exact_paired_effect_and_per_seed_effects() -> None:
    differences = np.asarray([[1, 0, -1], [1, 1, 0]], dtype=np.int8)
    assert paired_effect(differences) == Fraction(1, 3)
    assert per_seed_effects(differences) == (Fraction(1), Fraction(1, 2), Fraction(-1, 2))
    with pytest.raises(StatisticsContractError, match=r"\{-1,0,1\}"):
        paired_effect(np.asarray([[2, 0, 0]], dtype=np.int8))


def test_bootstrap_carries_whole_seed_vector_and_respects_strata() -> None:
    cancelling_vectors = np.asarray([[1, -1, 0], [-1, 1, 0]], dtype=np.int8)
    totals = stratified_item_bootstrap_totals(
        cancelling_vectors,
        ("L1", "L1"),
        repetitions=500,
        rng_seed=11,
    )
    assert np.array_equal(totals, np.zeros(500, dtype=np.int64))

    fixed_by_stratum = np.asarray([[1, 1, 1], [-1, -1, -1]], dtype=np.int8)
    totals = stratified_item_bootstrap_totals(
        fixed_by_stratum,
        ("L1", "L2"),
        repetitions=500,
        rng_seed=12,
    )
    assert np.array_equal(totals, np.zeros(500, dtype=np.int64))


def test_bootstrap_is_reproducible_and_seed_sensitive() -> None:
    differences = np.asarray([[1, 1, 0], [0, -1, 0], [-1, 0, 1]], dtype=np.int8)
    first = stratified_item_bootstrap_totals(
        differences, ("L1", "L1", "L1"), repetitions=200, rng_seed=91
    )
    second = stratified_item_bootstrap_totals(
        differences, ("L1", "L1", "L1"), repetitions=200, rng_seed=91
    )
    third = stratified_item_bootstrap_totals(
        differences, ("L1", "L1", "L1"), repetitions=200, rng_seed=92
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_type7_quantile_and_percentile_interval_are_exact() -> None:
    assert exact_linear_quantile(np.asarray([0, 10]), Fraction(1, 4)) == Fraction(5, 2)
    interval = percentile_interval(
        np.asarray([-3, -1, 1, 3], dtype=np.int64),
        denominator=3,
        confidence_ppm=500_000,
    )
    assert interval.lower.fraction == Fraction(-1, 2)
    assert interval.upper.fraction == Fraction(1, 2)


def test_exact_sign_flip_has_known_one_and_two_sided_values() -> None:
    differences = np.ones((3, 3), dtype=np.int8)
    assert exact_sign_flip_p_value(differences, alternative="greater") == Fraction(1, 8)
    assert exact_sign_flip_p_value(differences, alternative="two-sided") == Fraction(1, 4)
    with pytest.raises(StatisticsContractError, match="limited"):
        exact_sign_flip_p_value(np.ones((21, 3), dtype=np.int8), alternative="greater")


def test_monte_carlo_sign_flip_is_deterministic_and_plus_one() -> None:
    differences = np.ones((8, 3), dtype=np.int8)
    first = paired_sign_flip_randomization(
        differences,
        repetitions=5_000,
        rng_seed=123,
        alternative="greater",
    )
    second = paired_sign_flip_randomization(
        differences,
        repetitions=5_000,
        rng_seed=123,
        alternative="greater",
    )
    assert first == second
    extreme, p_value = first
    assert p_value == Fraction(extreme + 1, 5_001)
    assert abs(float(p_value - Fraction(1, 256))) < 0.01


def test_holm_step_down_is_exact_monotone_and_tie_safe() -> None:
    adjusted = holm_adjust(
        {"h1": Fraction(1, 100), "h2": Fraction(3, 100), "h3": Fraction(4, 100)}
    )
    assert adjusted == {
        "h1": Fraction(3, 100),
        "h2": Fraction(3, 50),
        "h3": Fraction(3, 50),
    }
    tied = holm_adjust({"b": Fraction(1, 100), "a": Fraction(1, 100)})
    assert tied == {"a": Fraction(1, 50), "b": Fraction(1, 50)}


@settings(max_examples=100)
@given(st.lists(st.integers(min_value=0, max_value=1_000), min_size=1, max_size=12))
def test_holm_property_adjustment_is_bounded_conservative_and_monotone(
    numerators: list[int],
) -> None:
    raw = {f"h{index:02d}": Fraction(value, 1_000) for index, value in enumerate(numerators)}
    adjusted = holm_adjust(raw)
    assert set(adjusted) == set(raw)
    assert all(raw[key] <= adjusted[key] <= 1 for key in raw)
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    assert [adjusted[key] for key in ordered] == sorted(adjusted[key] for key in ordered)


@settings(max_examples=100)
@given(
    st.lists(
        st.lists(st.integers(min_value=-1, max_value=1), min_size=3, max_size=3),
        min_size=1,
        max_size=50,
    )
)
def test_paired_effect_property_is_sign_symmetric_and_item_order_invariant(
    values: list[list[int]],
) -> None:
    differences = np.asarray(values, dtype=np.int8)
    effect = paired_effect(differences)
    assert paired_effect(-differences) == -effect
    assert paired_effect(differences[::-1]) == effect


def test_confirmatory_analysis_closes_effect_null_holm_and_equivalence() -> None:
    report = run_confirmatory_analysis(
        decision_fixture_panel(),
        fast_protocol(),
        require_preregistered=False,
    )
    c1a, c1b = report.c1_results
    assert c1a.point_estimate.fraction == 1
    assert c1a.bootstrap_interval.lower.fraction == 1
    assert c1a.statistical_superiority
    assert c1a.practical_success
    assert c1b.point_estimate.fraction == 0
    assert c1b.randomization.p_value.fraction == 1
    assert c1b.holm_adjusted_p_value.fraction == 1
    assert not c1b.statistical_superiority
    assert report.c2_result.point_estimate.fraction == 0
    assert report.c2_result.classification is C2Classification.PRACTICAL_EQUIVALENCE
    assert report.c2_result.equivalence_assessed
    assert report.c2_result.lower_margin_pass
    assert report.c2_result.upper_margin_pass


def test_seed_direction_instability_is_reported() -> None:
    count = 8
    baseline = [(False, True, False)] * count
    unstable = [(True, False, False)] * count
    neutral = [(False, False, False)] * count
    panel = panel_from_correctness(
        {
            ArmId.A0: baseline,
            ArmId.A1: unstable,
            ArmId.A2: neutral,
            ArmId.A3: neutral,
            ArmId.A4: neutral,
        }
    )
    report = run_confirmatory_analysis(panel, fast_protocol(), require_preregistered=False)
    assert report.c1_results[0].seed_instability
    assert tuple(effect.effect.fraction for effect in report.c1_results[0].seed_effects) == (
        Fraction(1),
        Fraction(-1),
        Fraction(0),
    )


def test_c2_superiority_takes_precedence_over_equivalence() -> None:
    count = 8
    false = [(False, False, False)] * count
    true = [(True, True, True)] * count
    panel = panel_from_correctness(
        {
            ArmId.A0: false,
            ArmId.A1: true,
            ArmId.A2: false,
            ArmId.A3: true,
            ArmId.A4: false,
        }
    )
    report = run_confirmatory_analysis(panel, fast_protocol(), require_preregistered=False)
    assert report.c2_result.classification is C2Classification.SUPERIOR_A3
    assert not report.c2_result.equivalence_assessed
    assert report.c2_result.lower_margin_pass is None
    assert report.c2_result.upper_margin_pass is None


def test_c2_detects_negative_superiority_and_ambiguous_effects() -> None:
    count = 8
    false = [(False, False, False)] * count
    true = [(True, True, True)] * count
    negative = panel_from_correctness(
        {
            ArmId.A0: false,
            ArmId.A1: true,
            ArmId.A2: false,
            ArmId.A3: false,
            ArmId.A4: true,
        }
    )
    negative_report = run_confirmatory_analysis(
        negative, fast_protocol(), require_preregistered=False
    )
    assert negative_report.c2_result.classification is C2Classification.SUPERIOR_A4
    assert not negative_report.c2_result.equivalence_assessed

    a3 = list(false)
    a4 = list(false)
    a3[0] = (True, True, True)
    a4[1] = (True, True, True)
    ambiguous = panel_from_correctness(
        {
            ArmId.A0: false,
            ArmId.A1: true,
            ArmId.A2: false,
            ArmId.A3: a3,
            ArmId.A4: a4,
        }
    )
    ambiguous_report = run_confirmatory_analysis(
        ambiguous, fast_protocol(), require_preregistered=False
    )
    assert ambiguous_report.c2_result.classification is C2Classification.INCONCLUSIVE
    assert ambiguous_report.c2_result.equivalence_assessed
    assert not (
        ambiguous_report.c2_result.lower_margin_pass
        and ambiguous_report.c2_result.upper_margin_pass
    )


def test_analysis_output_has_no_text_or_raw_generation_surface() -> None:
    record = run_confirmatory_analysis(
        decision_fixture_panel(), fast_protocol(), require_preregistered=False
    ).to_record()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {nested for child in value.values() for nested in keys(child)}
        if isinstance(value, list):
            return {nested for child in value for nested in keys(child)}
        return set()

    assert keys(record).isdisjoint(
        {
            "prompt",
            "generated_text",
            "output_token_ids",
            "reference",
            "sealed_answer",
            "sample_scores",
        }
    )


def test_analysis_report_round_trip_hash_and_binding_validation(tmp_path: Path) -> None:
    panel = decision_fixture_panel()
    protocol = fast_protocol()
    report = run_confirmatory_analysis(panel, protocol, require_preregistered=False)
    path = tmp_path / "analysis.json"
    write_confirmatory_analysis(report, path)
    loaded = load_confirmatory_analysis(path)
    assert loaded == report
    validate_analysis_bindings(loaded, panel=panel, protocol=protocol)
    with pytest.raises(StatisticsContractError, match="deterministic recomputation"):
        validate_analysis_bindings(
            loaded,
            panel=panel,
            protocol=replace(protocol, bootstrap_seed=protocol.bootstrap_seed + 1),
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["c1_results"][0]["point_estimate"]["numerator"] = 0
    payload["c1_results"][0]["point_estimate"]["denominator"] = 1
    payload["c1_results"][0]["point_estimate"]["ppm"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StatisticsContractError, match=r"disagrees|analysis_report_sha256"):
        load_confirmatory_analysis(path)


def test_binding_validation_rejects_self_hashed_contrast_forgery() -> None:
    panel = decision_fixture_panel()
    protocol = fast_protocol()
    report = run_confirmatory_analysis(panel, protocol, require_preregistered=False)
    forged_c1b = replace(
        report.c1_results[0],
        hypothesis_id="C1b",
        treatment_arm=ArmId.A2,
    )
    forged = type(report).build(
        panel=panel,
        protocol=protocol,
        c1_results=(report.c1_results[0], forged_c1b),
        c2_result=report.c2_result,
    )
    with pytest.raises(StatisticsContractError, match="deterministic recomputation"):
        validate_analysis_bindings(forged, panel=panel, protocol=protocol)


def test_rng_stream_identity_is_independent_of_observed_correctness() -> None:
    first = decision_fixture_panel()
    changed = panel_from_correctness(
        {
            ArmId.A0: [(False, False, False)] * 8,
            ArmId.A1: [(False, True, False)] * 8,
            ArmId.A2: [(True, False, True)] * 8,
            ArmId.A3: [(True, True, True)] * 8,
            ArmId.A4: [(False, False, False)] * 8,
        }
    )
    assert first.panel_sha256 != changed.panel_sha256
    assert resampling_design_sha256(first) == resampling_design_sha256(changed)
    protocol = fast_protocol()
    first_report = run_confirmatory_analysis(first, protocol, require_preregistered=False)
    changed_report = run_confirmatory_analysis(changed, protocol, require_preregistered=False)
    assert [result.bootstrap_stream_sha256 for result in first_report.c1_results] == [
        result.bootstrap_stream_sha256 for result in changed_report.c1_results
    ]
    assert [result.randomization.stream_sha256 for result in first_report.c1_results] == [
        result.randomization.stream_sha256 for result in changed_report.c1_results
    ]
    assert (
        first_report.c2_result.randomization.stream_sha256
        == changed_report.c2_result.randomization.stream_sha256
    )


def _completed_response(request, prediction: str) -> GenerationResponse:
    return GenerationResponse(
        request_id=request.request_id,
        status=GenerationStatus.COMPLETED,
        generated_text=prediction,
        output_token_ids=(*prediction.encode(), request.protocol.eos_token_id),
        finish_reason=FinishReason.EOS,
        error_code=None,
    )


def test_panel_builder_consumes_validated_d07_greedy_correctness_only() -> None:
    descriptor = load_benchmark_descriptor(EVALUATION_FIXTURE / "benchmark_descriptor.json")
    public = load_public_benchmark(descriptor, EVALUATION_FIXTURE / "public_items.jsonl")
    vault = load_sealed_answer_vault(public, EVALUATION_FIXTURE / "sealed_references.jsonl")
    protocol = load_generation_protocol(EVALUATION_FIXTURE / "greedy_protocol.json")
    reference_rows = [
        json.loads(line)
        for line in (EVALUATION_FIXTURE / "sealed_references.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    references = {row["item_id"]: row["reference"] for row in reference_rows}
    verifier = ExactMathVerifier()
    evaluator = EvaluatorContract(
        task=BenchmarkTask.EXACT_MATH,
        primary_metric="answer_accuracy",
        pass_at_k=(1,),
        verifier_policy_sha256=verifier.policy_digest,
    )
    evaluations: list[ArmSeedEvaluation] = []
    for arm in ARMS:
        for seed in SEEDS:
            def backend(requests, arm=arm, seed=seed):
                responses = []
                for request in requests:
                    is_correct = (request.item_index + seed + ARMS.index(arm)) % 2 == 0
                    prediction = (
                        f"Final answer: {references[request.item_id]}"
                        if is_correct
                        else "Final answer: 123456789"
                    )
                    responses.append(_completed_response(request, prediction))
                return tuple(responses)

            checkpoint = CheckpointIdentity(
                model_id=f"synthetic/{arm.value}",
                model_revision="5" * 40,
                checkpoint_sha256=digest(f"integration-checkpoint-{arm.value}-{seed}"),
            )
            batch = run_generation(
                public,
                run_id=f"d08-{arm.value}-{seed}-generation",
                checkpoint=checkpoint,
                protocol=protocol,
                backend=backend,
            )
            report = evaluate_generation_batch(
                public,
                vault,
                batch,
                evaluation_run_id=f"d08-{arm.value}-{seed}-evaluation",
                contract=evaluator,
                verifier=verifier,
            )
            evaluations.append(
                ArmSeedEvaluation(arm=arm, training_seed=seed, report=report)
            )
    panel = build_paired_panel(public, protocol, evaluations)
    assert len(panel.report_refs) == 15
    assert len(panel.items) == descriptor.item_count
    assert panel.public_items_sha256 == public.raw_sha256
    assert panel.generation_protocol_sha256 == protocol.digest
    assert panel.evaluator_contract_sha256 == evaluator.digest
    assert panel.items[0].for_arm(ArmId.A0) == tuple(
        evaluations[index].report.item_scores[0].sample_correctness[0] for index in range(3)
    )
    serialized = canonical_json_bytes(panel.to_record()).lower()
    assert b"final answer" not in serialized
    assert b'"prompt"' not in serialized
    assert b'"reference"' not in serialized

    sampling = load_generation_protocol(EVALUATION_FIXTURE / "sampling_protocol.json")
    with pytest.raises(StatisticsContractError, match="greedy"):
        build_paired_panel(public, sampling, evaluations)
    with pytest.raises(StatisticsContractError, match="public-items hash"):
        build_paired_panel(replace(public, raw_sha256="0" * 64), protocol, evaluations)


def test_difference_matrix_uses_arm_order_and_all_three_seeds() -> None:
    panel = decision_fixture_panel()
    differences = paired_difference_matrix(panel, treatment=ArmId.A1, control=ArmId.A0)
    assert differences.shape == (8, 3)
    assert np.array_equal(differences, np.ones((8, 3), dtype=np.int8))
    with pytest.raises(StatisticsContractError, match="different arms"):
        paired_difference_matrix(panel, treatment=ArmId.A1, control=ArmId.A1)
