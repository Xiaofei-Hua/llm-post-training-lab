from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from posttrain_lab.data import canonical_json_bytes
from posttrain_lab.evaluation import (
    STRICT_LABEL_POLICY_SHA256,
    BenchmarkDescriptor,
    BenchmarkTask,
    CheckpointIdentity,
    EvaluationContractError,
    EvaluatorContract,
    EvaluatorInfrastructureError,
    FinishReason,
    GenerationResponse,
    GenerationStatus,
    MetricContractError,
    RevisionKind,
    evaluate_generation_batch,
    evaluator_version_sha256,
    load_benchmark_descriptor,
    load_evaluation_report,
    load_generation_protocol,
    load_public_benchmark,
    load_sealed_answer_vault,
    pass_at_k_fraction,
    pass_at_k_ppm,
    run_generation,
    write_evaluation_report,
)
from posttrain_lab.rewards import ExactMathVerifier

FIXTURE_ROOT = Path("tests/fixtures/evaluation_contract")


def checkpoint() -> CheckpointIdentity:
    return CheckpointIdentity(
        model_id="synthetic/student",
        model_revision="5" * 40,
        checkpoint_sha256="b" * 64,
    )


def fixture_inputs():
    descriptor = load_benchmark_descriptor(FIXTURE_ROOT / "benchmark_descriptor.json")
    public = load_public_benchmark(descriptor, FIXTURE_ROOT / "public_items.jsonl")
    vault = load_sealed_answer_vault(public, FIXTURE_ROOT / "sealed_references.jsonl")
    predictions = json.loads(
        (FIXTURE_ROOT / "fixture_predictions.json").read_text(encoding="utf-8")
    )["items"]
    return public, vault, {item["item_id"]: item for item in predictions}


def completed_response(request, text: str) -> GenerationResponse:
    return GenerationResponse(
        request_id=request.request_id,
        status=GenerationStatus.COMPLETED,
        generated_text=text,
        output_token_ids=(*text.encode("utf-8"), request.protocol.eos_token_id),
        finish_reason=FinishReason.EOS,
        error_code=None,
    )


def evaluate_fixture(mode: str):
    public, vault, predictions = fixture_inputs()
    protocol = load_generation_protocol(FIXTURE_ROOT / f"{mode}_protocol.json")

    def backend(requests):
        responses = []
        for request in requests:
            item = predictions[request.item_id]
            text = item["greedy"] if mode == "greedy" else item["sampling"][request.sample_index]
            responses.append(completed_response(request, text))
        return tuple(reversed(responses))

    batch = run_generation(
        public,
        run_id=f"d07-{mode}-fixture",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=backend,
    )
    verifier = ExactMathVerifier()
    contract = EvaluatorContract(
        task=BenchmarkTask.EXACT_MATH,
        primary_metric="answer_accuracy",
        pass_at_k=(1,) if mode == "greedy" else (1, 8),
        verifier_policy_sha256=verifier.policy_digest,
    )
    report = evaluate_generation_batch(
        public,
        vault,
        batch,
        evaluation_run_id=f"d07-{mode}-evaluation",
        contract=contract,
        verifier=verifier,
    )
    return public, vault, batch, report


def test_pass_at_k_matches_exact_definition() -> None:
    assert pass_at_k_fraction(total_samples=10, correct_samples=3, k=1) == Fraction(3, 10)
    assert pass_at_k_fraction(total_samples=10, correct_samples=3, k=10) == 1
    assert pass_at_k_ppm(total_samples=3, correct_samples=1, k=1) == 333_333
    assert pass_at_k_ppm(total_samples=3, correct_samples=1, k=2) == 666_667


@given(total=st.integers(min_value=1, max_value=64), data=st.data())
def test_pass_at_k_is_bounded_and_monotone(total: int, data: st.DataObject) -> None:
    correct = data.draw(st.integers(min_value=0, max_value=total))
    values = [
        pass_at_k_fraction(total_samples=total, correct_samples=correct, k=k)
        for k in range(1, total + 1)
    ]
    assert all(Fraction(0) <= value <= Fraction(1) for value in values)
    assert values == sorted(values)
    assert values[0] == Fraction(correct, total)
    assert values[-1] == (Fraction(0) if correct == 0 else Fraction(1))


@pytest.mark.parametrize(
    (
        "mode",
        "sample_count",
        "correct",
        "accuracy",
        "extraction",
        "parse",
        "completion_tokens",
        "passes",
    ),
    [
        ("greedy", 6, 3, 500_000, 833_333, 833_333, 116, ((1, 500_000),)),
        (
            "sampling",
            48,
            21,
            437_500,
            958_333,
            958_333,
            562,
            ((1, 437_500), (8, 833_333)),
        ),
    ],
)
def test_fixture_metrics_are_frozen_item_level_contracts(
    mode: str,
    sample_count: int,
    correct: int,
    accuracy: int,
    extraction: int,
    parse: int,
    completion_tokens: int,
    passes: tuple[tuple[int, int], ...],
) -> None:
    _, _, batch, report = evaluate_fixture(mode)
    assert report.sample_count == sample_count
    assert report.correct_sample_count == correct
    assert report.answer_accuracy_ppm == accuracy
    assert report.extraction_rate_ppm == extraction
    assert report.parse_rate_ppm == parse
    assert report.completion_token_count_total == completion_tokens
    assert report.truncation_count == 0
    assert report.truncation_rate_ppm == 0
    assert report.finish_reason_counts == (("eos", sample_count),)
    assert report.pass_at_k_ppm == passes
    assert len(report.sample_scores) == len(batch.records)
    assert all(
        item.sample_correctness
        == tuple(score.correct for score in report.sample_scores if score.item_id == item.item_id)
        for item in report.item_scores
    )
    if mode == "greedy":
        assert report.verification_status_counts == (
            ("match", 3),
            ("mismatch", 2),
            ("prediction_not_extracted", 1),
        )


def test_report_contains_no_reference_or_generated_text_surface() -> None:
    _, _, _, report = evaluate_fixture("sampling")
    record = report.to_record()

    def all_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {nested for child in value.values() for nested in all_keys(child)}
        if isinstance(value, list):
            return {nested for child in value for nested in all_keys(child)}
        return set()

    assert all_keys(record).isdisjoint(
        {
            "prompt",
            "generated_text",
            "output_token_ids",
            "reference",
            "reference_answer",
            "candidate",
        }
    )


def test_invalid_sealed_reference_is_infrastructure_failure(tmp_path: Path) -> None:
    public, _, predictions = fixture_inputs()
    rows = [
        json.loads(line)
        for line in (FIXTURE_ROOT / "sealed_references.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows[0]["reference"] = "0/0"
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    sealed_path = tmp_path / "sealed.jsonl"
    sealed_path.write_bytes(raw)
    changed_descriptor = replace(
        public.descriptor,
        sealed_references_sha256=hashlib.sha256(raw).hexdigest(),
    )
    changed_public = replace(public, descriptor=changed_descriptor)
    vault = load_sealed_answer_vault(changed_public, sealed_path)
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")

    def backend(requests):
        return tuple(
            completed_response(request, predictions[request.item_id]["greedy"])
            for request in requests
        )

    batch = run_generation(
        changed_public,
        run_id="d07-invalid-reference",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=backend,
    )
    verifier = ExactMathVerifier()
    contract = EvaluatorContract(
        task=BenchmarkTask.EXACT_MATH,
        primary_metric="answer_accuracy",
        pass_at_k=(1,),
        verifier_policy_sha256=verifier.policy_digest,
    )
    with pytest.raises(EvaluatorInfrastructureError, match="cannot score sealed reference"):
        evaluate_generation_batch(
            changed_public,
            vault,
            batch,
            evaluation_run_id="d07-invalid-reference-eval",
            contract=contract,
            verifier=verifier,
        )


def test_failed_generation_is_not_counted_as_wrong() -> None:
    public, vault, _ = fixture_inputs()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")

    def backend(requests):
        return tuple(
            GenerationResponse(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                generated_text=None,
                output_token_ids=(),
                finish_reason=None,
                error_code="backend.unavailable",
            )
            if index == 0
            else completed_response(request, "Final answer: 0")
            for index, request in enumerate(requests)
        )

    batch = run_generation(
        public,
        run_id="d07-infrastructure-failure",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=backend,
    )
    verifier = ExactMathVerifier()
    contract = EvaluatorContract(
        task=BenchmarkTask.EXACT_MATH,
        primary_metric="answer_accuracy",
        pass_at_k=(1,),
        verifier_policy_sha256=verifier.policy_digest,
    )
    with pytest.raises(EvaluatorInfrastructureError, match="contains 1 failures"):
        evaluate_generation_batch(
            public,
            vault,
            batch,
            evaluation_run_id="d07-failed-eval",
            contract=contract,
            verifier=verifier,
        )


def test_length_finish_is_preserved_in_sanitized_metrics() -> None:
    public, vault, _ = fixture_inputs()
    protocol = replace(
        load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json"),
        max_new_tokens=3,
    )

    def backend(requests):
        return tuple(
            GenerationResponse(
                request_id=request.request_id,
                status=GenerationStatus.COMPLETED,
                generated_text="x",
                output_token_ids=(7, 8, 9),
                finish_reason=FinishReason.LENGTH,
                error_code=None,
            )
            for request in requests
        )

    batch = run_generation(
        public,
        run_id="d07-length-finish",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=backend,
    )
    verifier = ExactMathVerifier()
    report = evaluate_generation_batch(
        public,
        vault,
        batch,
        evaluation_run_id="d07-length-finish-eval",
        contract=EvaluatorContract(
            task=BenchmarkTask.EXACT_MATH,
            primary_metric="answer_accuracy",
            pass_at_k=(1,),
            verifier_policy_sha256=verifier.policy_digest,
        ),
        verifier=verifier,
    )
    assert report.completion_token_count_total == 18
    assert report.truncation_count == 6
    assert report.truncation_rate_ppm == 1_000_000
    assert report.finish_reason_counts == (("length", 6),)
    assert all(score.completion_token_count == 3 for score in report.sample_scores)


def test_evaluator_policy_and_backend_versions_are_hash_bound() -> None:
    _, _, _, report = evaluate_fixture("greedy")
    assert report.evaluator_version_sha256 == evaluator_version_sha256(
        report.evaluator_contract,
        backend_versions=report.backend_versions,
    )
    assert report.evaluator_version_sha256 != evaluator_version_sha256(
        report.evaluator_contract,
        backend_versions={**report.backend_versions, "extra": "changed"},
    )
    with pytest.raises(TypeError):
        report.backend_versions["mutable"] = "forbidden"  # type: ignore[index]
    incompatible = replace(report.evaluator_contract, verifier_policy_sha256="f" * 64)
    public, vault, batch, _ = evaluate_fixture("greedy")
    with pytest.raises(MetricContractError, match="verifier policy"):
        evaluate_generation_batch(
            public,
            vault,
            batch,
            evaluation_run_id="d07-policy-mismatch",
            contract=incompatible,
        )


def test_evaluation_report_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    _, _, _, report = evaluate_fixture("sampling")
    path = tmp_path / "report.json"
    write_evaluation_report(report, path)
    assert load_evaluation_report(path) == report

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["answer_accuracy_ppm"] += 1
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(MetricContractError, match="answer_accuracy_ppm"):
        load_evaluation_report(path)


def test_report_revalidates_item_to_sample_binding() -> None:
    _, _, _, report = evaluate_fixture("greedy")
    original = report.item_scores[0]
    unsigned = {**original.unsigned_record(), "item_id": "math:forged"}
    forged = replace(
        original,
        item_id="math:forged",
        item_score_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    )
    with pytest.raises(MetricContractError, match="sample item_id"):
        replace(report, item_scores=(forged, *report.item_scores[1:]))

    with pytest.raises(MetricContractError, match="verification_status=match"):
        replace(report.sample_scores[0], correct=False)


def test_boolean_summary_counts_are_rejected_even_when_equal_to_one() -> None:
    _, _, _, report = evaluate_fixture("greedy")
    item = report.item_scores[0]
    assert item.correct_count == 1
    forged_item_unsigned = {**item.unsigned_record(), "correct_count": True}
    with pytest.raises(EvaluationContractError, match="integer"):
        replace(
            item,
            correct_count=True,
            item_score_sha256=hashlib.sha256(
                canonical_json_bytes(forged_item_unsigned)
            ).hexdigest(),
        )

    forged_counts = (
        ("match", 3),
        ("mismatch", 2),
        ("prediction_not_extracted", True),
    )
    forged_report_unsigned = {
        **report.unsigned_record(),
        "verification_status_counts": dict(forged_counts),
    }
    with pytest.raises(EvaluationContractError, match="integer"):
        replace(
            report,
            verification_status_counts=forged_counts,
            report_sha256=hashlib.sha256(
                canonical_json_bytes(forged_report_unsigned)
            ).hexdigest(),
        )


def test_prediction_change_changes_generation_and_report_hashes() -> None:
    public, vault, original_batch, original_report = evaluate_fixture("greedy")
    protocol = original_batch.protocol

    def backend(requests):
        return tuple(
            completed_response(
                request,
                "Final answer: 0" if index == 0 else original_batch.records[index].generated_text,
            )
            for index, request in enumerate(requests)
        )

    changed_batch = run_generation(
        public,
        run_id=original_batch.run_id,
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=backend,
    )
    changed_report = evaluate_generation_batch(
        public,
        vault,
        changed_batch,
        evaluation_run_id=original_report.evaluation_run_id,
        contract=original_report.evaluator_contract,
    )
    assert changed_batch.record_set_sha256 != original_batch.record_set_sha256
    assert changed_report.report_sha256 != original_report.report_sha256
    assert changed_report.answer_accuracy_ppm == 333_333


def test_strict_label_adapter_obeys_frozen_exact_policy(tmp_path: Path) -> None:
    revision = "6" * 40
    public_rows = [
        {
            "schema_version": "d07-public-benchmark-item-v1",
            "benchmark_id": "synthetic/labels",
            "benchmark_revision": revision,
            "item_id": f"label:{index}",
            "item_index": index,
            "prompt": f"Return label {label}.",
            "strata": {"kind": "label"},
        }
        for index, label in enumerate(("A", "B"))
    ]
    sealed_rows = [
        {
            "schema_version": "d07-sealed-reference-v1",
            "benchmark_id": "synthetic/labels",
            "benchmark_revision": revision,
            "item_id": f"label:{index}",
            "reference": label,
        }
        for index, label in enumerate(("A", "B"))
    ]
    public_raw = b"".join(canonical_json_bytes(row) + b"\n" for row in public_rows)
    sealed_raw = b"".join(canonical_json_bytes(row) + b"\n" for row in sealed_rows)
    public_path = tmp_path / "public.jsonl"
    sealed_path = tmp_path / "sealed.jsonl"
    public_path.write_bytes(public_raw)
    sealed_path.write_bytes(sealed_raw)
    descriptor = BenchmarkDescriptor(
        benchmark_id="synthetic/labels",
        revision_kind=RevisionKind.GIT_COMMIT,
        benchmark_revision=revision,
        split_name="test",
        task=BenchmarkTask.STRICT_LABEL,
        item_count=2,
        public_items_sha256=hashlib.sha256(public_raw).hexdigest(),
        sealed_references_sha256=hashlib.sha256(sealed_raw).hexdigest(),
        source_registry_sha256="c" * 64,
        data_manifest_sha256="d" * 64,
    )
    public = load_public_benchmark(descriptor, public_path)
    vault = load_sealed_answer_vault(public, sealed_path)
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")

    def backend(requests):
        return (
            completed_response(requests[0], " A\n"),
            completed_response(requests[1], "b"),
        )

    batch = run_generation(
        public,
        run_id="d07-strict-label",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=backend,
    )
    contract = EvaluatorContract(
        task=BenchmarkTask.STRICT_LABEL,
        primary_metric="answer_accuracy",
        pass_at_k=(1,),
        verifier_policy_sha256=STRICT_LABEL_POLICY_SHA256,
    )
    report = evaluate_generation_batch(
        public,
        vault,
        batch,
        evaluation_run_id="d07-strict-label-eval",
        contract=contract,
    )
    assert report.answer_accuracy_ppm == 500_000
    assert tuple(score.correct for score in report.sample_scores) == (True, False)


def test_metric_contract_rejects_invalid_domains() -> None:
    with pytest.raises(EvaluationContractError, match="correct_samples"):
        pass_at_k_fraction(total_samples=2, correct_samples=3, k=1)
    with pytest.raises(EvaluationContractError, match="k"):
        pass_at_k_fraction(total_samples=2, correct_samples=1, k=3)
