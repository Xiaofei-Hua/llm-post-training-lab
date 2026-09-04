"""Generator/evaluator separation and deterministic D07 execution paths."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from posttrain_lab.data import canonical_json_bytes
from posttrain_lab.rewards import ExactMathVerifier, VerificationResult

from .contracts import (
    BenchmarkTask,
    CheckpointIdentity,
    GenerationBatch,
    GenerationContractError,
    GenerationProtocol,
    GenerationRecord,
    GenerationRequest,
    GenerationResponse,
    GenerationStatus,
    LoadedPublicBenchmark,
    SealedAnswerVault,
    SealedReferenceError,
    build_generation_request,
)
from .metrics import (
    EvaluationReport,
    EvaluatorContract,
    EvaluatorInfrastructureError,
    MetricContractError,
    SampleScore,
)

STRICT_LABEL_POLICY = "strip_nfc_case_sensitive_exact"
STRICT_LABEL_POLICY_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "schema_version": "d07-strict-label-policy-v1",
            "normalization": STRICT_LABEL_POLICY,
            "reference_pattern": "[A-Za-z0-9][A-Za-z0-9._+-]{0,63}",
            "prediction_requires_entire_trimmed_surface": True,
        }
    )
).hexdigest()
_STRICT_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")


class GenerationBackend(Protocol):
    """A backend receives only public requests and returns keyed responses."""

    def __call__(
        self,
        requests: tuple[GenerationRequest, ...],
    ) -> Sequence[GenerationResponse]: ...


@dataclass(frozen=True, slots=True)
class _ScoreDecision:
    correct: bool
    verification_status: str
    extraction_status: str
    parse_status: str
    answer_source: str | None
    marker_count: int


def prepare_generation_requests(
    public: LoadedPublicBenchmark,
    *,
    checkpoint: CheckpointIdentity,
    protocol: GenerationProtocol,
) -> tuple[GenerationRequest, ...]:
    """Build a canonical request grid without receiving a sealed vault."""

    requests = tuple(
        build_generation_request(
            item,
            checkpoint=checkpoint,
            protocol=protocol,
            sample_index=sample_index,
        )
        for item in public.items
        for sample_index in range(protocol.samples_per_item)
    )
    if len({request.request_id for request in requests}) != len(requests):
        raise GenerationContractError("request-id collision in generation grid")
    return requests


def run_generation(
    public: LoadedPublicBenchmark,
    *,
    run_id: str,
    checkpoint: CheckpointIdentity,
    protocol: GenerationProtocol,
    backend: GenerationBackend,
) -> GenerationBatch:
    """Execute one complete grid and canonicalize out-of-order backend responses."""

    requests = prepare_generation_requests(
        public,
        checkpoint=checkpoint,
        protocol=protocol,
    )
    responses_raw = backend(requests)
    if isinstance(responses_raw, (str, bytes)):
        raise GenerationContractError("generation backend must return response objects")
    responses = tuple(responses_raw)
    if any(not isinstance(response, GenerationResponse) for response in responses):
        raise GenerationContractError("generation backend returned a non-response object")
    expected_ids = {request.request_id for request in requests}
    response_ids = [response.request_id for response in responses]
    if len(response_ids) != len(set(response_ids)):
        raise GenerationContractError("generation backend returned duplicate request_id")
    actual_ids = set(response_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:3]
        extra = sorted(actual_ids - expected_ids)[:3]
        raise GenerationContractError(
            f"generation response set differs from requests; missing={missing}, extra={extra}"
        )
    by_id = {response.request_id: response for response in responses}
    records = tuple(
        # Backend order is never trusted; output order follows the frozen request grid.
        # Each response is consumed exactly once because duplicate IDs were rejected.
        GenerationRecord.from_request_response(request, by_id[request.request_id])
        for request in requests
    )
    return GenerationBatch(
        run_id=run_id,
        descriptor=public.descriptor,
        public_items_sha256=public.raw_sha256,
        public_item_set_sha256=public.item_set_sha256,
        checkpoint=checkpoint,
        protocol=protocol,
        records=records,
    )


def _math_decision(
    reference: str,
    prediction: str,
    *,
    verifier: ExactMathVerifier,
) -> _ScoreDecision:
    result: VerificationResult = verifier.verify(reference, prediction)
    if result.reward is None:
        raise EvaluatorInfrastructureError(
            f"cannot score sealed reference: {result.status.value}: {result.reason}"
        )
    extraction = result.prediction.extraction
    return _ScoreDecision(
        correct=result.reward == 1.0,
        verification_status=result.status.value,
        extraction_status=extraction.status.value,
        parse_status=result.prediction.status.value,
        answer_source=extraction.source.value if extraction.source is not None else None,
        marker_count=extraction.marker_count,
    )


def _strict_label_decision(reference: str, prediction: str) -> _ScoreDecision:
    if _STRICT_LABEL_RE.fullmatch(reference) is None:
        raise EvaluatorInfrastructureError("sealed strict-label reference violates policy")
    candidate = unicodedata.normalize("NFC", prediction.strip())
    extracted = _STRICT_LABEL_RE.fullmatch(candidate) is not None
    return _ScoreDecision(
        correct=extracted and candidate == reference,
        verification_status=("match" if extracted and candidate == reference else "mismatch"),
        extraction_status=("strict_label_extracted" if extracted else "strict_label_not_extracted"),
        parse_status=("strict_label_parsed" if extracted else "strict_label_unparseable"),
        answer_source="strict_label" if extracted else None,
        marker_count=1 if extracted else 0,
    )


def evaluate_generation_batch(
    public: LoadedPublicBenchmark,
    vault: SealedAnswerVault,
    batch: GenerationBatch,
    *,
    evaluation_run_id: str,
    contract: EvaluatorContract,
    verifier: ExactMathVerifier | None = None,
) -> EvaluationReport:
    """Score against sealed ground truth and return reference-free item records."""

    if (
        batch.descriptor != public.descriptor
        or batch.public_items_sha256 != public.raw_sha256
        or batch.public_item_set_sha256 != public.item_set_sha256
    ):
        raise MetricContractError("generation batch does not match public benchmark")
    if (
        vault.benchmark_id != public.descriptor.benchmark_id
        or vault.benchmark_revision != public.descriptor.benchmark_revision
        or vault.task is not public.descriptor.task
        or vault.raw_sha256 != public.descriptor.sealed_references_sha256
        or vault.item_count != public.descriptor.item_count
    ):
        raise SealedReferenceError("sealed vault does not match benchmark descriptor")
    expected_id_set_sha256 = hashlib.sha256(
        canonical_json_bytes(sorted(item.item_id for item in public.items))
    ).hexdigest()
    if vault.item_set_sha256 != expected_id_set_sha256:
        raise SealedReferenceError("sealed vault item set does not match public benchmark")
    contract.assert_compatible(public.descriptor, batch.protocol)
    if batch.failed_count:
        failed_ids = [
            record.request_id
            for record in batch.records
            if record.status is GenerationStatus.FAILED
        ][:3]
        raise EvaluatorInfrastructureError(
            f"generation batch contains {batch.failed_count} failures: {failed_ids}"
        )

    resolved_verifier = verifier or ExactMathVerifier()
    if contract.task is BenchmarkTask.EXACT_MATH:
        if contract.verifier_policy_sha256 != resolved_verifier.policy_digest:
            raise MetricContractError("evaluator contract does not match verifier policy")
        backend_versions = resolved_verifier.backend_versions

        def score(task: BenchmarkTask, reference: str, prediction: str) -> _ScoreDecision:
            if task is not BenchmarkTask.EXACT_MATH:
                raise MetricContractError("sealed task changed during evaluation")
            return _math_decision(reference, prediction, verifier=resolved_verifier)

    else:
        if contract.verifier_policy_sha256 != STRICT_LABEL_POLICY_SHA256:
            raise MetricContractError("evaluator contract does not match strict-label policy")
        backend_versions = {"strict-label": "d07-strict-label-policy-v1"}

        def score(task: BenchmarkTask, reference: str, prediction: str) -> _ScoreDecision:
            if task is not BenchmarkTask.STRICT_LABEL:
                raise MetricContractError("sealed task changed during evaluation")
            return _strict_label_decision(reference, prediction)

    sample_scores: list[SampleScore] = []
    for record, item in zip(
        batch.records,
        (item for item in public.items for _ in range(batch.protocol.samples_per_item)),
        strict=True,
    ):
        if (
            record.item_id != item.item_id
            or record.item_index != item.item_index
            or record.prompt_sha256 != item.prompt_sha256
        ):
            raise MetricContractError("generation record is not bound to its public item")
        assert record.generated_text is not None
        decision = vault._score_with(item.item_id, record.generated_text, score)
        if not isinstance(decision, _ScoreDecision):
            raise EvaluatorInfrastructureError("sealed scorer returned an invalid decision")
        sample_scores.append(
            SampleScore.build(
                record,
                correct=decision.correct,
                verification_status=decision.verification_status,
                extraction_status=decision.extraction_status,
                parse_status=decision.parse_status,
                answer_source=decision.answer_source,
                marker_count=decision.marker_count,
            )
        )
    return EvaluationReport.build(
        evaluation_run_id=evaluation_run_id,
        batch=batch,
        sealed_references_sha256=vault.raw_sha256,
        evaluator_contract=contract,
        backend_versions=backend_versions,
        sample_scores=sample_scores,
    )
