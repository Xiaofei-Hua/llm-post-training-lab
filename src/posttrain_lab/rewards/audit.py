"""Deterministic adversarial audit runner for the D05 math verifier."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .verifier import ExactMathVerifier, VerificationResult, VerificationStatus

_CASE_SCHEMA_VERSION = "d05-verifier-audit-case-v1"
_REPORT_SCHEMA_VERSION = "d05-verifier-audit-report-v1"
_IMPLEMENTATION_SOURCE_PATHS = (
    "src/posttrain_lab/rewards/verifier.py",
    "src/posttrain_lab/rewards/audit.py",
    "scripts/audit_verifier.py",
)
_DEPENDENCY_LOCK_PATH = "uv.lock"
_ALLOWED_CASE_FIELDS = frozenset(
    {
        "case_id",
        "category",
        "reference",
        "prediction",
        "expected_reward",
        "expected_status",
        "notes",
    }
)
_REQUIRED_CASE_FIELDS = frozenset(
    {"case_id", "category", "reference", "prediction", "expected_reward"}
)


class AuditCorpusError(ValueError):
    """Raised when an audit corpus violates its immutable schema."""


class AuditProvenanceError(RuntimeError):
    """Raised when an audit cannot be tied to clean versioned inputs."""


@dataclass(frozen=True, slots=True)
class VerifierAuditCase:
    """One explicit prediction attack or correctness check."""

    case_id: str
    category: str
    reference: str
    prediction: str
    expected_reward: float | None
    expected_status: VerificationStatus | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise AuditCorpusError("case_id must be a non-empty string")
        if not isinstance(self.category, str) or not self.category.strip():
            raise AuditCorpusError(f"{self.case_id}: category must be a non-empty string")
        if not isinstance(self.reference, str) or not isinstance(self.prediction, str):
            raise AuditCorpusError(f"{self.case_id}: reference and prediction must be strings")
        if isinstance(self.expected_reward, bool) or self.expected_reward not in {
            None,
            0.0,
            1.0,
        }:
            raise AuditCorpusError(
                f"{self.case_id}: expected_reward must be exactly 0, 1, or null"
            )
        infrastructure_statuses = {
            VerificationStatus.REFERENCE_INVALID,
            VerificationStatus.BACKEND_ERROR,
        }
        if self.expected_reward is None and self.expected_status not in infrastructure_statuses:
            raise AuditCorpusError(
                f"{self.case_id}: null reward requires reference_invalid or backend_error status"
            )
        if self.expected_reward is not None and self.expected_status in infrastructure_statuses:
            raise AuditCorpusError(
                f"{self.case_id}: infrastructure status requires a null reward"
            )
        if self.notes is not None and not isinstance(self.notes, str):
            raise AuditCorpusError(f"{self.case_id}: notes must be a string when provided")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, line_number: int) -> VerifierAuditCase:
        extra = set(raw) - _ALLOWED_CASE_FIELDS
        missing = _REQUIRED_CASE_FIELDS - set(raw)
        if extra:
            raise AuditCorpusError(
                f"line {line_number}: unknown fields: {', '.join(sorted(extra))}"
            )
        if missing:
            raise AuditCorpusError(
                f"line {line_number}: missing fields: {', '.join(sorted(missing))}"
            )
        expected_status_raw = raw.get("expected_status")
        try:
            expected_status = (
                VerificationStatus(expected_status_raw)
                if expected_status_raw is not None
                else None
            )
        except (TypeError, ValueError) as error:
            raise AuditCorpusError(
                f"line {line_number}: invalid expected_status {expected_status_raw!r}"
            ) from error
        expected_reward_raw = raw["expected_reward"]
        if expected_reward_raw is not None and (
            isinstance(expected_reward_raw, bool)
            or not isinstance(expected_reward_raw, (int, float))
        ):
            raise AuditCorpusError(
                f"line {line_number}: expected_reward must be numeric or null"
            )
        return cls(
            case_id=raw["case_id"],
            category=raw["category"],
            reference=raw["reference"],
            prediction=raw["prediction"],
            expected_reward=(
                float(expected_reward_raw) if expected_reward_raw is not None else None
            ),
            expected_status=expected_status,
            notes=raw.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class LoadedAuditCorpus:
    """Validated cases plus the raw-file digest used for provenance."""

    schema_version: str
    path: str
    sha256: str
    cases: tuple[VerifierAuditCase, ...]


@dataclass(frozen=True, slots=True)
class AuditProvenance:
    """Git, source, and lock fingerprints for one verifier audit."""

    git_revision: str
    tracked_inputs_match_git: bool
    tracked_input_paths: tuple[str, ...]
    source_files_sha256: dict[str, str]
    implementation_source_sha256: str
    dependency_lock_path: str
    dependency_lock_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "git_revision": self.git_revision,
            "tracked_inputs_match_git": self.tracked_inputs_match_git,
            "tracked_input_paths": list(self.tracked_input_paths),
            "source_files_sha256": dict(sorted(self.source_files_sha256.items())),
            "implementation_source_sha256": self.implementation_source_sha256,
            "dependency_lock_path": self.dependency_lock_path,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }


@dataclass(frozen=True, slots=True)
class AuditFailure:
    """One expectation mismatch retained in the machine-readable report."""

    case_id: str
    category: str
    expected_reward: float | None
    actual_reward: float | None
    expected_status: str | None
    actual_status: str
    result: VerificationResult

    def to_record(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "expected_reward": self.expected_reward,
            "actual_reward": self.actual_reward,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "result": self.result.to_record(),
        }


@dataclass(frozen=True, slots=True)
class VerifierAuditReport:
    """Deterministic audit summary; no training or benchmark result is implied."""

    schema_version: str
    case_schema_version: str
    corpus_path: str
    corpus_sha256: str
    policy_sha256: str
    backend_versions: dict[str, str]
    provenance: AuditProvenance
    total_cases: int
    passed_cases: int
    failed_cases: int
    passed: bool
    category_counts: dict[str, dict[str, int]]
    failures: tuple[AuditFailure, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_schema_version": self.case_schema_version,
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "policy_sha256": self.policy_sha256,
            "backend_versions": dict(self.backend_versions),
            "provenance": self.provenance.to_record(),
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "passed": self.passed,
            "category_counts": {
                category: dict(counts) for category, counts in sorted(self.category_counts.items())
            },
            "failures": [failure.to_record() for failure in self.failures],
        }


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise AuditProvenanceError(f"cannot fingerprint {path}: {error}") from error


def _run_git(repository_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise AuditProvenanceError(f"cannot execute git: {error}") from error


def collect_audit_provenance(
    repository_root: str | Path,
    *,
    require_clean: bool = True,
    additional_input_paths: Sequence[str | Path] = (),
) -> AuditProvenance:
    """Fingerprint implementation inputs and bind them to one Git revision.

    ``additional_input_paths`` is intended for the exact corpus used by a
    formal audit.  Every path must remain inside ``repository_root`` so Git can
    prove that the bytes belong to the recorded revision.
    """

    root = Path(repository_root).resolve()
    normalized_additional_inputs: list[str] = []
    for raw_path in additional_input_paths:
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise AuditProvenanceError(
                f"audit input must be inside repository root: {candidate}"
            ) from error
        if relative not in normalized_additional_inputs:
            normalized_additional_inputs.append(relative)

    revision_result = _run_git(root, "rev-parse", "--verify", "HEAD")
    if revision_result.returncode != 0:
        raise AuditProvenanceError("repository HEAD cannot be resolved")
    git_revision = revision_result.stdout.strip()
    if len(git_revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in git_revision
    ):
        raise AuditProvenanceError("repository HEAD is not a full Git object revision")

    all_inputs = tuple(
        dict.fromkeys(
            (
                *_IMPLEMENTATION_SOURCE_PATHS,
                _DEPENDENCY_LOCK_PATH,
                *normalized_additional_inputs,
            )
        )
    )
    source_digests = {
        relative: _sha256_file(root / relative) for relative in _IMPLEMENTATION_SOURCE_PATHS
    }
    dependency_lock_sha256 = _sha256_file(root / _DEPENDENCY_LOCK_PATH)
    aggregate_payload = json.dumps(
        source_digests,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    implementation_source_sha256 = hashlib.sha256(aggregate_payload).hexdigest()

    tracked_inputs_match_git = True
    for relative in all_inputs:
        tracked = _run_git(root, "ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            tracked_inputs_match_git = False
            break
    if tracked_inputs_match_git:
        diff_result = _run_git(root, "diff", "--quiet", "HEAD", "--", *all_inputs)
        if diff_result.returncode not in {0, 1}:
            raise AuditProvenanceError("git could not compare audit inputs with HEAD")
        tracked_inputs_match_git = diff_result.returncode == 0
    if require_clean and not tracked_inputs_match_git:
        raise AuditProvenanceError(
            "audit implementation sources, uv.lock, and additional inputs "
            "must be tracked and match HEAD"
        )

    return AuditProvenance(
        git_revision=git_revision,
        tracked_inputs_match_git=tracked_inputs_match_git,
        tracked_input_paths=all_inputs,
        source_files_sha256=source_digests,
        implementation_source_sha256=implementation_source_sha256,
        dependency_lock_path=_DEPENDENCY_LOCK_PATH,
        dependency_lock_sha256=dependency_lock_sha256,
    )


def load_audit_corpus(path: str | Path) -> LoadedAuditCorpus:
    """Load strict UTF-8 JSONL and reject duplicate or unknown case fields."""

    resolved = Path(path)
    try:
        raw_bytes = resolved.read_bytes()
    except OSError as error:
        raise AuditCorpusError(f"cannot read audit corpus {resolved}: {error}") from error
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditCorpusError(f"audit corpus {resolved} is not valid UTF-8") from error

    cases: list[VerifierAuditCase] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuditCorpusError(
                f"line {line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(raw, dict):
            raise AuditCorpusError(f"line {line_number}: each case must be a JSON object")
        case = VerifierAuditCase.from_mapping(raw, line_number=line_number)
        if case.case_id in case_ids:
            raise AuditCorpusError(f"line {line_number}: duplicate case_id {case.case_id!r}")
        case_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise AuditCorpusError("audit corpus must contain at least one case")
    return LoadedAuditCorpus(
        schema_version=_CASE_SCHEMA_VERSION,
        path=resolved.as_posix(),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        cases=tuple(cases),
    )


def run_verifier_audit(
    corpus: LoadedAuditCorpus,
    *,
    verifier: ExactMathVerifier | None = None,
    provenance: AuditProvenance | None = None,
    minimum_cases: int = 100,
    maximum_cases: int = 300,
) -> VerifierAuditReport:
    """Evaluate every case and retain all expectation mismatches."""

    if isinstance(minimum_cases, bool) or not isinstance(minimum_cases, int):
        raise ValueError("minimum_cases must be an integer")
    if isinstance(maximum_cases, bool) or not isinstance(maximum_cases, int):
        raise ValueError("maximum_cases must be an integer")
    if minimum_cases <= 0 or maximum_cases < minimum_cases:
        raise ValueError("case-count bounds must satisfy 0 < minimum_cases <= maximum_cases")
    total = len(corpus.cases)
    if not minimum_cases <= total <= maximum_cases:
        raise AuditCorpusError(
            f"audit corpus must contain {minimum_cases}-{maximum_cases} cases; got {total}"
        )

    resolved_verifier = verifier or ExactMathVerifier()
    resolved_provenance = provenance or collect_audit_provenance(
        Path.cwd(),
        require_clean=False,
    )
    failures: list[AuditFailure] = []
    category_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0}
    )
    for case in corpus.cases:
        result = resolved_verifier.verify(case.reference, case.prediction)
        status_matches = case.expected_status is None or result.status is case.expected_status
        reward_matches = result.reward == case.expected_reward
        passed = status_matches and reward_matches
        counts = category_counts[case.category]
        counts["total"] += 1
        counts["passed" if passed else "failed"] += 1
        if not passed:
            failures.append(
                AuditFailure(
                    case_id=case.case_id,
                    category=case.category,
                    expected_reward=case.expected_reward,
                    actual_reward=result.reward,
                    expected_status=(
                        case.expected_status.value if case.expected_status is not None else None
                    ),
                    actual_status=result.status.value,
                    result=result,
                )
            )

    failed_count = len(failures)
    return VerifierAuditReport(
        schema_version=_REPORT_SCHEMA_VERSION,
        case_schema_version=corpus.schema_version,
        corpus_path=corpus.path,
        corpus_sha256=corpus.sha256,
        policy_sha256=resolved_verifier.policy_digest,
        backend_versions=resolved_verifier.backend_versions,
        provenance=resolved_provenance,
        total_cases=total,
        passed_cases=total - failed_count,
        failed_cases=failed_count,
        passed=failed_count == 0,
        category_counts={category: dict(counts) for category, counts in category_counts.items()},
        failures=tuple(failures),
    )


def write_audit_report(report: VerifierAuditReport, path: str | Path) -> None:
    """Atomically write the deterministic JSON report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report.to_record(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary_path: Path | None = None
    try:
        descriptor, raw_temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
        temporary_path = Path(raw_temporary_path)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
