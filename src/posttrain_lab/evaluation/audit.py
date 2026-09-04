"""Git-bound synthetic audit for the D07 sealed evaluation pipeline."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain_lab.data import canonical_json_bytes, strict_json_loads, write_json_atomic
from posttrain_lab.rewards import ExactMathVerifier

from .contracts import (
    BenchmarkTask,
    CheckpointIdentity,
    FinishReason,
    GenerationProtocol,
    GenerationRequest,
    GenerationResponse,
    GenerationStatus,
    LoadedPublicBenchmark,
    SealedAnswerVault,
    _load_benchmark_descriptor_with_raw_sha256,
    _load_generation_protocol_with_raw_sha256,
    load_public_benchmark,
    load_sealed_answer_vault,
)
from .metrics import EvaluationReport, EvaluatorContract
from .runner import evaluate_generation_batch, run_generation

EVALUATOR_AUDIT_EXPECTATION_SCHEMA_VERSION = "d07-evaluator-audit-expectation-v1"
EVALUATOR_AUDIT_REPORT_SCHEMA_VERSION = "d07-evaluator-audit-report-v1"
EVALUATOR_FIXTURE_PREDICTIONS_SCHEMA_VERSION = "d07-evaluator-fixture-predictions-v2"

_MAX_CONTROL_FILE_BYTES = 16 * 1024**2
_MAX_PREDICTION_CHARS = 4_000_000
_IMPLEMENTATION_SOURCE_PATHS = (
    "src/posttrain_lab/__init__.py",
    "src/posttrain_lab/data/__init__.py",
    "src/posttrain_lab/data/audit.py",
    "src/posttrain_lab/data/contamination.py",
    "src/posttrain_lab/data/registry.py",
    "src/posttrain_lab/rewards/__init__.py",
    "src/posttrain_lab/rewards/audit.py",
    "src/posttrain_lab/rewards/verifier.py",
    "src/posttrain_lab/evaluation/__init__.py",
    "src/posttrain_lab/evaluation/contracts.py",
    "src/posttrain_lab/evaluation/metrics.py",
    "src/posttrain_lab/evaluation/runner.py",
    "src/posttrain_lab/evaluation/audit.py",
    "scripts/audit_evaluator.py",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
)
_RUNTIME_MODULE_PATHS = (
    ("posttrain_lab", "src/posttrain_lab/__init__.py"),
    ("posttrain_lab.data", "src/posttrain_lab/data/__init__.py"),
    ("posttrain_lab.data.audit", "src/posttrain_lab/data/audit.py"),
    ("posttrain_lab.data.contamination", "src/posttrain_lab/data/contamination.py"),
    ("posttrain_lab.data.registry", "src/posttrain_lab/data/registry.py"),
    ("posttrain_lab.rewards", "src/posttrain_lab/rewards/__init__.py"),
    ("posttrain_lab.rewards.audit", "src/posttrain_lab/rewards/audit.py"),
    ("posttrain_lab.rewards.verifier", "src/posttrain_lab/rewards/verifier.py"),
    ("posttrain_lab.evaluation", "src/posttrain_lab/evaluation/__init__.py"),
    ("posttrain_lab.evaluation.audit", "src/posttrain_lab/evaluation/audit.py"),
    ("posttrain_lab.evaluation.contracts", "src/posttrain_lab/evaluation/contracts.py"),
    ("posttrain_lab.evaluation.metrics", "src/posttrain_lab/evaluation/metrics.py"),
    ("posttrain_lab.evaluation.runner", "src/posttrain_lab/evaluation/runner.py"),
    ("__main__", "scripts/audit_evaluator.py"),
)


class EvaluatorAuditError(ValueError):
    """Raised when an audit fixture or expectation violates its contract."""


class EvaluatorAuditProvenanceError(RuntimeError):
    """Raised when audit bytes cannot be tied to one clean Git revision."""


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluatorAuditError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluatorAuditError(f"{field} must be a non-negative integer")
    return value


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EvaluatorAuditError(f"{field} must be a string-keyed mapping")
    return value


def _require_exact_keys(raw: Mapping[str, Any], *, required: set[str], field: str) -> None:
    missing = required - set(raw)
    extra = set(raw) - required
    if missing or extra:
        raise EvaluatorAuditError(
            f"{field} has invalid keys; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _strict_counts(value: object, *, field: str) -> dict[str, int]:
    raw = _require_mapping(value, field=field)
    counts: dict[str, int] = {}
    for key, count in raw.items():
        if not key:
            raise EvaluatorAuditError(f"{field} keys must be non-empty")
        counts[key] = _require_nonnegative_int(count, field=f"{field}.{key}")
    return dict(sorted(counts.items()))


def _load_json_object(path: Path, *, field: str) -> tuple[str, Mapping[str, Any]]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_CONTROL_FILE_BYTES + 1)
    except OSError as error:
        raise EvaluatorAuditError(f"cannot read {field}: {error}") from error
    if len(raw) > _MAX_CONTROL_FILE_BYTES:
        raise EvaluatorAuditError(f"{field} exceeds {_MAX_CONTROL_FILE_BYTES} bytes")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise EvaluatorAuditError(f"{field} must be canonical UTF-8 without BOM/CR")
    try:
        parsed = strict_json_loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError) as error:
        raise EvaluatorAuditError(f"invalid strict {field}: {error}") from error
    return hashlib.sha256(raw).hexdigest(), _require_mapping(parsed, field=field)


@dataclass(frozen=True, slots=True)
class FixturePredictions:
    raw_sha256: str
    by_item_id: dict[str, tuple[str, tuple[str, ...]]]
    truncations: frozenset[tuple[str, str, int]]

    def is_truncated(self, *, mode: str, request: GenerationRequest) -> bool:
        return (mode, request.item_id, request.sample_index) in self.truncations


def load_fixture_predictions(
    path: str | Path,
    *,
    public: LoadedPublicBenchmark,
    greedy_protocol: GenerationProtocol,
    sampling_protocol: GenerationProtocol,
) -> FixturePredictions:
    """Load the bounded synthetic backend outputs used only by the D07 audit."""

    raw_sha256, payload = _load_json_object(Path(path), field="fixture predictions")
    _require_exact_keys(
        payload,
        required={"schema_version", "items", "truncations"},
        field="fixture predictions",
    )
    if payload["schema_version"] != EVALUATOR_FIXTURE_PREDICTIONS_SCHEMA_VERSION:
        raise EvaluatorAuditError("unsupported fixture predictions schema")
    items = payload["items"]
    if not isinstance(items, list) or len(items) != public.descriptor.item_count:
        raise EvaluatorAuditError("fixture predictions must cover every public item exactly")
    loaded: dict[str, tuple[str, tuple[str, ...]]] = {}
    observed_order: list[str] = []
    for index, value in enumerate(items):
        item = _require_mapping(value, field=f"fixture predictions.items[{index}]")
        _require_exact_keys(
            item,
            required={"item_id", "greedy", "sampling"},
            field=f"fixture predictions.items[{index}]",
        )
        item_id = item["item_id"]
        greedy = item["greedy"]
        sampling = item["sampling"]
        if not isinstance(item_id, str) or not item_id:
            raise EvaluatorAuditError(f"fixture item {index} has invalid item_id")
        if item_id in loaded:
            raise EvaluatorAuditError(f"duplicate fixture item_id: {item_id}")
        if not isinstance(greedy, str) or len(greedy) > _MAX_PREDICTION_CHARS:
            raise EvaluatorAuditError(f"fixture item {item_id} has invalid greedy output")
        if (
            not isinstance(sampling, list)
            or len(sampling) != sampling_protocol.samples_per_item
            or any(
                not isinstance(prediction, str) or len(prediction) > _MAX_PREDICTION_CHARS
                for prediction in sampling
            )
        ):
            raise EvaluatorAuditError(
                f"fixture item {item_id} must have exactly "
                f"{sampling_protocol.samples_per_item} bounded sampling outputs"
            )
        loaded[item_id] = (greedy, tuple(sampling))
        observed_order.append(item_id)
    expected_order = [item.item_id for item in public.items]
    if observed_order != expected_order:
        raise EvaluatorAuditError("fixture predictions are not in canonical public-item order")
    if greedy_protocol.samples_per_item != 1:
        raise EvaluatorAuditError("audit greedy protocol must have one sample per item")
    truncations = payload["truncations"]
    if not isinstance(truncations, list):
        raise EvaluatorAuditError("fixture predictions.truncations must be a list")
    public_indices = {item.item_id: item.item_index for item in public.items}
    loaded_truncations: list[tuple[str, str, int]] = []
    for index, value in enumerate(truncations):
        entry = _require_mapping(value, field=f"fixture predictions.truncations[{index}]")
        _require_exact_keys(
            entry,
            required={"mode", "item_id", "sample_index"},
            field=f"fixture predictions.truncations[{index}]",
        )
        mode = entry["mode"]
        item_id = entry["item_id"]
        sample_index = _require_nonnegative_int(
            entry["sample_index"],
            field=f"fixture predictions.truncations[{index}].sample_index",
        )
        if not isinstance(mode, str) or mode not in {"greedy", "sampling"}:
            raise EvaluatorAuditError("fixture truncation mode must be greedy or sampling")
        if not isinstance(item_id, str) or item_id not in public_indices:
            raise EvaluatorAuditError("fixture truncation item_id must name a public item")
        protocol = greedy_protocol if mode == "greedy" else sampling_protocol
        if sample_index >= protocol.samples_per_item:
            raise EvaluatorAuditError("fixture truncation sample_index exceeds protocol")
        key = (mode, item_id, sample_index)
        if key in loaded_truncations:
            raise EvaluatorAuditError("duplicate fixture truncation key")
        loaded_truncations.append(key)
    mode_order = {"greedy": 0, "sampling": 1}
    expected_truncation_order = sorted(
        loaded_truncations,
        key=lambda key: (mode_order[key[0]], public_indices[key[1]], key[2]),
    )
    if loaded_truncations != expected_truncation_order:
        raise EvaluatorAuditError("fixture truncations are not in canonical order")
    return FixturePredictions(
        raw_sha256=raw_sha256,
        by_item_id=loaded,
        truncations=frozenset(loaded_truncations),
    )


@dataclass(frozen=True, slots=True)
class ExpectedRunSummary:
    protocol_sha256: str
    generation_record_set_sha256: str
    evaluator_contract_sha256: str
    evaluator_version_sha256: str
    report_sha256: str
    sample_count: int
    correct_sample_count: int
    answer_accuracy_ppm: int
    extraction_rate_ppm: int
    parse_rate_ppm: int
    completion_token_count_total: int
    truncation_count: int
    truncation_rate_ppm: int
    pass_at_k_ppm: dict[str, int]
    verification_status_counts: dict[str, int]
    finish_reason_counts: dict[str, int]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, field: str) -> ExpectedRunSummary:
        required = {
            "protocol_sha256",
            "generation_record_set_sha256",
            "evaluator_contract_sha256",
            "evaluator_version_sha256",
            "report_sha256",
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
        }
        _require_exact_keys(raw, required=required, field=field)
        for name in (
            "protocol_sha256",
            "generation_record_set_sha256",
            "evaluator_contract_sha256",
            "evaluator_version_sha256",
            "report_sha256",
        ):
            _require_sha256(raw[name], field=f"{field}.{name}")
        integer_fields = (
            "sample_count",
            "correct_sample_count",
            "answer_accuracy_ppm",
            "extraction_rate_ppm",
            "parse_rate_ppm",
            "completion_token_count_total",
            "truncation_count",
            "truncation_rate_ppm",
        )
        values = {
            name: _require_nonnegative_int(raw[name], field=f"{field}.{name}")
            for name in integer_fields
        }
        pass_counts = _strict_counts(raw["pass_at_k_ppm"], field=f"{field}.pass_at_k_ppm")
        if not pass_counts or any(not key.isdigit() or int(key) < 1 for key in pass_counts):
            raise EvaluatorAuditError(f"{field}.pass_at_k_ppm keys must be positive integers")
        if values["sample_count"] < 1:
            raise EvaluatorAuditError(f"{field}.sample_count must be positive")
        if values["correct_sample_count"] > values["sample_count"]:
            raise EvaluatorAuditError(f"{field}.correct_sample_count exceeds sample_count")
        for name in (
            "answer_accuracy_ppm",
            "extraction_rate_ppm",
            "parse_rate_ppm",
            "truncation_rate_ppm",
        ):
            if values[name] > 1_000_000:
                raise EvaluatorAuditError(f"{field}.{name} exceeds one million ppm")
        if values["truncation_count"] > values["sample_count"]:
            raise EvaluatorAuditError(f"{field}.truncation_count exceeds sample_count")
        if any(value > 1_000_000 for value in pass_counts.values()):
            raise EvaluatorAuditError(f"{field}.pass_at_k_ppm exceeds one million ppm")
        verification_counts = _strict_counts(
            raw["verification_status_counts"],
            field=f"{field}.verification_status_counts",
        )
        finish_counts = _strict_counts(
            raw["finish_reason_counts"],
            field=f"{field}.finish_reason_counts",
        )
        if sum(verification_counts.values()) != values["sample_count"]:
            raise EvaluatorAuditError(
                f"{field}.verification_status_counts do not sum to sample_count"
            )
        if sum(finish_counts.values()) != values["sample_count"]:
            raise EvaluatorAuditError(f"{field}.finish_reason_counts do not sum to sample_count")
        return cls(
            protocol_sha256=raw["protocol_sha256"],
            generation_record_set_sha256=raw["generation_record_set_sha256"],
            evaluator_contract_sha256=raw["evaluator_contract_sha256"],
            evaluator_version_sha256=raw["evaluator_version_sha256"],
            report_sha256=raw["report_sha256"],
            **values,
            pass_at_k_ppm=pass_counts,
            verification_status_counts=verification_counts,
            finish_reason_counts=finish_counts,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "generation_record_set_sha256": self.generation_record_set_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "evaluator_version_sha256": self.evaluator_version_sha256,
            "report_sha256": self.report_sha256,
            "sample_count": self.sample_count,
            "correct_sample_count": self.correct_sample_count,
            "answer_accuracy_ppm": self.answer_accuracy_ppm,
            "extraction_rate_ppm": self.extraction_rate_ppm,
            "parse_rate_ppm": self.parse_rate_ppm,
            "completion_token_count_total": self.completion_token_count_total,
            "truncation_count": self.truncation_count,
            "truncation_rate_ppm": self.truncation_rate_ppm,
            "pass_at_k_ppm": dict(sorted(self.pass_at_k_ppm.items())),
            "verification_status_counts": dict(sorted(self.verification_status_counts.items())),
            "finish_reason_counts": dict(sorted(self.finish_reason_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class EvaluatorAuditExpectation:
    benchmark_descriptor_sha256: str
    public_items_sha256: str
    sealed_references_sha256: str
    fixture_predictions_sha256: str
    verifier_policy_sha256: str
    runs: dict[str, ExpectedRunSummary]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EvaluatorAuditExpectation:
        required = {
            "schema_version",
            "benchmark_descriptor_sha256",
            "public_items_sha256",
            "sealed_references_sha256",
            "fixture_predictions_sha256",
            "verifier_policy_sha256",
            "runs",
        }
        _require_exact_keys(raw, required=required, field="audit expectation")
        if raw["schema_version"] != EVALUATOR_AUDIT_EXPECTATION_SCHEMA_VERSION:
            raise EvaluatorAuditError("unsupported evaluator audit expectation schema")
        for name in required - {"schema_version", "runs"}:
            _require_sha256(raw[name], field=f"audit expectation.{name}")
        raw_runs = _require_mapping(raw["runs"], field="audit expectation.runs")
        if set(raw_runs) != {"greedy", "sampling"}:
            raise EvaluatorAuditError("audit expectation must define greedy and sampling runs")
        return cls(
            benchmark_descriptor_sha256=raw["benchmark_descriptor_sha256"],
            public_items_sha256=raw["public_items_sha256"],
            sealed_references_sha256=raw["sealed_references_sha256"],
            fixture_predictions_sha256=raw["fixture_predictions_sha256"],
            verifier_policy_sha256=raw["verifier_policy_sha256"],
            runs={
                mode: ExpectedRunSummary.from_mapping(
                    _require_mapping(raw_runs[mode], field=f"runs.{mode}"),
                    field=f"runs.{mode}",
                )
                for mode in ("greedy", "sampling")
            },
        )


def load_evaluator_audit_expectation(
    path: str | Path,
) -> tuple[str, EvaluatorAuditExpectation]:
    raw_sha256, payload = _load_json_object(Path(path), field="audit expectation")
    return raw_sha256, EvaluatorAuditExpectation.from_mapping(payload)


@dataclass(frozen=True, slots=True)
class EvaluatorAuditProvenance:
    git_revision: str
    tracked_inputs_match_git: bool
    tracked_input_paths: tuple[str, ...]
    tracked_files_sha256: dict[str, str]
    implementation_source_sha256: str
    runtime_versions: dict[str, str]

    def formal_validation_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if len(self.git_revision) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in self.git_revision
        ):
            failures.append("git_revision is not a full lowercase commit digest")
        if self.tracked_input_paths[: len(_IMPLEMENTATION_SOURCE_PATHS)] != (
            _IMPLEMENTATION_SOURCE_PATHS
        ):
            failures.append("tracked inputs do not start with the frozen implementation set")
        if len(self.tracked_input_paths) != len(set(self.tracked_input_paths)):
            failures.append("tracked input paths contain duplicates")
        if set(self.tracked_files_sha256) != set(self.tracked_input_paths):
            failures.append("tracked hashes do not exactly cover tracked input paths")
        for path in self.tracked_input_paths:
            candidate = Path(path)
            if (
                not path
                or candidate.is_absolute()
                or "\\" in path
                or any(part in {"", ".", ".."} for part in candidate.parts)
            ):
                failures.append(f"tracked input path is not canonical: {path!r}")
                break
        for path, digest in self.tracked_files_sha256.items():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                failures.append(f"invalid tracked SHA-256 for {path}")
                break
        implementation_hash = hashlib.sha256(
            canonical_json_bytes(
                {
                    path: self.tracked_files_sha256[path]
                    for path in _IMPLEMENTATION_SOURCE_PATHS
                    if path in self.tracked_files_sha256
                }
            )
        ).hexdigest()
        if self.implementation_source_sha256 != implementation_hash:
            failures.append("implementation_source_sha256 is inconsistent")
        if not self.runtime_versions or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.runtime_versions.items()
        ):
            failures.append("runtime_versions must be a non-empty string mapping")
        if self.tracked_inputs_match_git is not True:
            failures.append("tracked inputs do not match Git")
        return tuple(failures)

    def assert_formal(self) -> None:
        failures = self.formal_validation_failures()
        if failures:
            raise EvaluatorAuditProvenanceError(
                f"invalid formal evaluator provenance: {failures[0]}"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "git_revision": self.git_revision,
            "tracked_inputs_match_git": self.tracked_inputs_match_git,
            "tracked_input_paths": list(self.tracked_input_paths),
            "tracked_files_sha256": dict(sorted(self.tracked_files_sha256.items())),
            "implementation_source_paths": list(_IMPLEMENTATION_SOURCE_PATHS),
            "implementation_source_sha256": self.implementation_source_sha256,
            "runtime_versions": dict(sorted(self.runtime_versions.items())),
        }


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise EvaluatorAuditProvenanceError(f"cannot execute Git: {error}") from error


def _repository_relative_path(root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise EvaluatorAuditProvenanceError(
            f"audit input must exist inside repository: {path}"
        ) from error
    return relative.as_posix()


def _assert_runtime_source_origin(root: Path) -> None:
    """Require every executed project module to originate in the audited checkout."""

    for module_name, relative in _RUNTIME_MODULE_PATHS:
        module = sys.modules.get(module_name)
        source = getattr(module, "__file__", None) if module is not None else None
        if not isinstance(source, str):
            raise EvaluatorAuditProvenanceError(
                f"runtime module source is unavailable: {module_name}"
            )
        try:
            actual = Path(source).resolve(strict=True)
            expected = (root / relative).resolve(strict=True)
        except OSError as error:
            raise EvaluatorAuditProvenanceError(
                f"cannot resolve runtime module source: {module_name}: {error}"
            ) from error
        if actual != expected:
            raise EvaluatorAuditProvenanceError(
                f"runtime module source is outside audited repository: {module_name}"
            )


def collect_evaluator_audit_provenance(
    repository_root: str | Path,
    *,
    input_paths: Sequence[str | Path],
    require_clean: bool = True,
) -> EvaluatorAuditProvenance:
    """Fingerprint working bytes and compare them with one captured Git revision."""

    root = Path(repository_root).resolve(strict=True)
    top_level = _run_git(root, "rev-parse", "--show-toplevel")
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root:
        raise EvaluatorAuditProvenanceError("repository_root must be the Git worktree root")
    revision_result = _run_git(root, "rev-parse", "HEAD")
    revision = revision_result.stdout.strip()
    if (
        revision_result.returncode != 0
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise EvaluatorAuditProvenanceError("cannot resolve a full Git HEAD revision")
    all_paths: list[str] = []
    for path in (*_IMPLEMENTATION_SOURCE_PATHS, *input_paths):
        relative = _repository_relative_path(root, path)
        if relative not in all_paths:
            all_paths.append(relative)
    tracked_hashes: dict[str, str] = {}
    tracked_match = True
    for relative in all_paths:
        tracked = _run_git(root, "ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            raise EvaluatorAuditProvenanceError(f"audit input is not Git tracked: {relative}")
        head_bytes = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if head_bytes.returncode != 0:
            raise EvaluatorAuditProvenanceError(
                f"cannot read {relative} from recorded revision {revision}"
            )
        try:
            working_bytes = (root / relative).read_bytes()
        except OSError as error:
            raise EvaluatorAuditProvenanceError(
                f"cannot read audit input {relative}: {error}"
            ) from error
        if working_bytes != head_bytes.stdout:
            tracked_match = False
            if require_clean:
                raise EvaluatorAuditProvenanceError(
                    f"audit input differs from recorded Git revision: {relative}"
                )
        tracked_hashes[relative] = hashlib.sha256(working_bytes).hexdigest()
    final_revision = _run_git(root, "rev-parse", "HEAD")
    if final_revision.returncode != 0 or final_revision.stdout.strip() != revision:
        raise EvaluatorAuditProvenanceError("Git HEAD changed while provenance was collected")
    implementation_hash = hashlib.sha256(
        canonical_json_bytes({path: tracked_hashes[path] for path in _IMPLEMENTATION_SOURCE_PATHS})
    ).hexdigest()
    return EvaluatorAuditProvenance(
        git_revision=revision,
        tracked_inputs_match_git=tracked_match,
        tracked_input_paths=tuple(all_paths),
        tracked_files_sha256=tracked_hashes,
        implementation_source_sha256=implementation_hash,
        runtime_versions={
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "unicode_database": unicodedata.unidata_version,
        },
    )


def _assert_inputs_unchanged(root: Path, provenance: EvaluatorAuditProvenance) -> None:
    for relative, expected_sha256 in provenance.tracked_files_sha256.items():
        try:
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as error:
            raise EvaluatorAuditProvenanceError(
                f"cannot reread audit input {relative}: {error}"
            ) from error
        if actual != expected_sha256:
            raise EvaluatorAuditProvenanceError(f"audit input changed during execution: {relative}")


def _assert_head_unchanged(root: Path, provenance: EvaluatorAuditProvenance) -> None:
    current_revision = _run_git(root, "rev-parse", "HEAD")
    if (
        current_revision.returncode != 0
        or current_revision.stdout.strip() != provenance.git_revision
    ):
        raise EvaluatorAuditProvenanceError("Git HEAD changed during evaluator audit")


def _assert_consumed_digest(
    provenance: EvaluatorAuditProvenance,
    *,
    relative: str,
    actual_sha256: str,
) -> None:
    expected_sha256 = provenance.tracked_files_sha256.get(relative)
    if expected_sha256 is None or actual_sha256 != expected_sha256:
        raise EvaluatorAuditProvenanceError(
            f"audit loader consumed bytes outside captured provenance: {relative}"
        )


@dataclass(frozen=True, slots=True)
class RunAuditSummary:
    protocol_sha256: str
    generation_record_set_sha256: str
    evaluator_contract_sha256: str
    evaluator_version_sha256: str
    report_sha256: str
    sample_count: int
    correct_sample_count: int
    answer_accuracy_ppm: int
    extraction_rate_ppm: int
    parse_rate_ppm: int
    completion_token_count_total: int
    truncation_count: int
    truncation_rate_ppm: int
    pass_at_k_ppm: dict[str, int]
    verification_status_counts: dict[str, int]
    finish_reason_counts: dict[str, int]

    @classmethod
    def from_report(cls, report: EvaluationReport) -> RunAuditSummary:
        return cls(
            protocol_sha256=report.protocol_sha256,
            generation_record_set_sha256=report.generation_record_set_sha256,
            evaluator_contract_sha256=report.evaluator_contract.digest,
            evaluator_version_sha256=report.evaluator_version_sha256,
            report_sha256=report.report_sha256,
            sample_count=report.sample_count,
            correct_sample_count=report.correct_sample_count,
            answer_accuracy_ppm=report.answer_accuracy_ppm,
            extraction_rate_ppm=report.extraction_rate_ppm,
            parse_rate_ppm=report.parse_rate_ppm,
            completion_token_count_total=report.completion_token_count_total,
            truncation_count=report.truncation_count,
            truncation_rate_ppm=report.truncation_rate_ppm,
            pass_at_k_ppm={str(k): value for k, value in report.pass_at_k_ppm},
            verification_status_counts=dict(report.verification_status_counts),
            finish_reason_counts=dict(report.finish_reason_counts),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "generation_record_set_sha256": self.generation_record_set_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "evaluator_version_sha256": self.evaluator_version_sha256,
            "report_sha256": self.report_sha256,
            "sample_count": self.sample_count,
            "correct_sample_count": self.correct_sample_count,
            "answer_accuracy_ppm": self.answer_accuracy_ppm,
            "extraction_rate_ppm": self.extraction_rate_ppm,
            "parse_rate_ppm": self.parse_rate_ppm,
            "completion_token_count_total": self.completion_token_count_total,
            "truncation_count": self.truncation_count,
            "truncation_rate_ppm": self.truncation_rate_ppm,
            "pass_at_k_ppm": dict(sorted(self.pass_at_k_ppm.items())),
            "verification_status_counts": dict(sorted(self.verification_status_counts.items())),
            "finish_reason_counts": dict(sorted(self.finish_reason_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class EvaluatorAuditReport:
    provenance: EvaluatorAuditProvenance
    expectation_sha256: str
    benchmark_descriptor_sha256: str
    public_items_sha256: str
    sealed_references_sha256: str
    fixture_predictions_sha256: str
    verifier_policy_sha256: str
    runs: dict[str, RunAuditSummary]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures and not self.provenance.formal_validation_failures()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": EVALUATOR_AUDIT_REPORT_SCHEMA_VERSION,
            "passed": self.passed,
            "provenance": self.provenance.to_record(),
            "expectation_sha256": self.expectation_sha256,
            "benchmark_descriptor_sha256": self.benchmark_descriptor_sha256,
            "public_items_sha256": self.public_items_sha256,
            "sealed_references_sha256": self.sealed_references_sha256,
            "fixture_predictions_sha256": self.fixture_predictions_sha256,
            "verifier_policy_sha256": self.verifier_policy_sha256,
            "runs": {mode: self.runs[mode].to_record() for mode in ("greedy", "sampling")},
            "failures": list(self.failures),
        }


def _fixture_response(
    request: GenerationRequest,
    prediction: str,
    *,
    truncated: bool,
) -> GenerationResponse:
    if truncated:
        output_token_ids = (1,) * request.protocol.max_new_tokens
        finish_reason = FinishReason.LENGTH
    else:
        output_token_ids = (*prediction.encode("utf-8"), request.protocol.eos_token_id)
        finish_reason = FinishReason.EOS
    return GenerationResponse(
        request_id=request.request_id,
        status=GenerationStatus.COMPLETED,
        generated_text=prediction,
        output_token_ids=output_token_ids,
        finish_reason=finish_reason,
        error_code=None,
    )


def _run_fixture_mode(
    mode: str,
    *,
    public: LoadedPublicBenchmark,
    vault: SealedAnswerVault,
    predictions: FixturePredictions,
    protocol: GenerationProtocol,
    verifier: ExactMathVerifier,
) -> RunAuditSummary:
    if mode not in {"greedy", "sampling"}:
        raise EvaluatorAuditError(f"unsupported audit mode: {mode}")

    def backend(requests: tuple[GenerationRequest, ...]) -> tuple[GenerationResponse, ...]:
        responses: list[GenerationResponse] = []
        for request in requests:
            greedy, samples = predictions.by_item_id[request.item_id]
            prediction = greedy if mode == "greedy" else samples[request.sample_index]
            responses.append(
                _fixture_response(
                    request,
                    prediction,
                    truncated=predictions.is_truncated(mode=mode, request=request),
                )
            )
        return tuple(reversed(responses))

    checkpoint = CheckpointIdentity(
        model_id="synthetic/d07-audit-student",
        model_revision="5" * 40,
        checkpoint_sha256="b" * 64,
    )
    batch = run_generation(
        public,
        run_id=f"d07-audit-{mode}-generation",
        checkpoint=checkpoint,
        protocol=protocol,
        backend=backend,
    )
    pass_values = (1,) if mode == "greedy" else (1, protocol.samples_per_item)
    contract = EvaluatorContract(
        task=BenchmarkTask.EXACT_MATH,
        primary_metric="answer_accuracy",
        pass_at_k=pass_values,
        verifier_policy_sha256=verifier.policy_digest,
    )
    report = evaluate_generation_batch(
        public,
        vault,
        batch,
        evaluation_run_id=f"d07-audit-{mode}-evaluation",
        contract=contract,
        verifier=verifier,
    )
    return RunAuditSummary.from_report(report)


def _comparison_failures(
    expected: EvaluatorAuditExpectation,
    *,
    descriptor_sha256: str,
    public_sha256: str,
    sealed_sha256: str,
    predictions_sha256: str,
    verifier_policy_sha256: str,
    runs: Mapping[str, RunAuditSummary],
) -> tuple[str, ...]:
    failures: list[str] = []
    observed_top = {
        "benchmark_descriptor_sha256": descriptor_sha256,
        "public_items_sha256": public_sha256,
        "sealed_references_sha256": sealed_sha256,
        "fixture_predictions_sha256": predictions_sha256,
        "verifier_policy_sha256": verifier_policy_sha256,
    }
    for field, observed in observed_top.items():
        if observed != getattr(expected, field):
            failures.append(f"{field} differs from frozen expectation")
    for mode in ("greedy", "sampling"):
        expected_record = expected.runs[mode].to_record()
        observed_record = runs[mode].to_record()
        for field in expected_record:
            if observed_record[field] != expected_record[field]:
                failures.append(f"{mode}.{field} differs from frozen expectation")
    return tuple(failures)


def run_evaluator_audit(
    repository_root: str | Path,
    *,
    descriptor_path: str | Path,
    public_items_path: str | Path,
    sealed_references_path: str | Path,
    greedy_protocol_path: str | Path,
    sampling_protocol_path: str | Path,
    fixture_predictions_path: str | Path,
    expectation_path: str | Path,
) -> EvaluatorAuditReport:
    """Run the D07 audit from Git-tracked paths without caller-supplied hashes."""

    root = Path(repository_root).resolve(strict=True)
    _assert_runtime_source_origin(root)
    requested = (
        descriptor_path,
        public_items_path,
        sealed_references_path,
        greedy_protocol_path,
        sampling_protocol_path,
        fixture_predictions_path,
        expectation_path,
    )
    relative = tuple(_repository_relative_path(root, path) for path in requested)
    provenance = collect_evaluator_audit_provenance(
        root,
        input_paths=relative,
        require_clean=True,
    )
    provenance.assert_formal()
    _assert_inputs_unchanged(root, provenance)

    descriptor_raw_sha256, descriptor = _load_benchmark_descriptor_with_raw_sha256(
        root / relative[0]
    )
    _assert_consumed_digest(
        provenance,
        relative=relative[0],
        actual_sha256=descriptor_raw_sha256,
    )
    public = load_public_benchmark(descriptor, root / relative[1])
    _assert_consumed_digest(
        provenance,
        relative=relative[1],
        actual_sha256=public.raw_sha256,
    )
    vault = load_sealed_answer_vault(public, root / relative[2])
    _assert_consumed_digest(
        provenance,
        relative=relative[2],
        actual_sha256=vault.raw_sha256,
    )
    greedy_protocol_raw_sha256, greedy_protocol = (
        _load_generation_protocol_with_raw_sha256(root / relative[3])
    )
    _assert_consumed_digest(
        provenance,
        relative=relative[3],
        actual_sha256=greedy_protocol_raw_sha256,
    )
    sampling_protocol_raw_sha256, sampling_protocol = (
        _load_generation_protocol_with_raw_sha256(root / relative[4])
    )
    _assert_consumed_digest(
        provenance,
        relative=relative[4],
        actual_sha256=sampling_protocol_raw_sha256,
    )
    predictions = load_fixture_predictions(
        root / relative[5],
        public=public,
        greedy_protocol=greedy_protocol,
        sampling_protocol=sampling_protocol,
    )
    _assert_consumed_digest(
        provenance,
        relative=relative[5],
        actual_sha256=predictions.raw_sha256,
    )
    expectation_sha256, expectation = load_evaluator_audit_expectation(root / relative[6])
    _assert_consumed_digest(
        provenance,
        relative=relative[6],
        actual_sha256=expectation_sha256,
    )
    if descriptor.task is not BenchmarkTask.EXACT_MATH:
        raise EvaluatorAuditError("D07 formal audit requires an exact-math descriptor")
    verifier = ExactMathVerifier()
    runs = {
        mode: _run_fixture_mode(
            mode,
            public=public,
            vault=vault,
            predictions=predictions,
            protocol=protocol,
            verifier=verifier,
        )
        for mode, protocol in (
            ("greedy", greedy_protocol),
            ("sampling", sampling_protocol),
        )
    }
    failures = _comparison_failures(
        expectation,
        descriptor_sha256=descriptor.digest,
        public_sha256=public.raw_sha256,
        sealed_sha256=vault.raw_sha256,
        predictions_sha256=predictions.raw_sha256,
        verifier_policy_sha256=verifier.policy_digest,
        runs=runs,
    )
    report = EvaluatorAuditReport(
        provenance=provenance,
        expectation_sha256=expectation_sha256,
        benchmark_descriptor_sha256=descriptor.digest,
        public_items_sha256=public.raw_sha256,
        sealed_references_sha256=vault.raw_sha256,
        fixture_predictions_sha256=predictions.raw_sha256,
        verifier_policy_sha256=verifier.policy_digest,
        runs=runs,
        failures=failures,
    )
    _assert_inputs_unchanged(root, provenance)
    _assert_head_unchanged(root, provenance)
    return report


def write_evaluator_audit(report: EvaluatorAuditReport, path: str | Path) -> None:
    write_json_atomic(report.to_record(), path)


def evaluator_audit_report_sha256(report: EvaluatorAuditReport) -> str:
    return hashlib.sha256(canonical_json_bytes(report.to_record())).hexdigest()


__all__ = [
    "EVALUATOR_AUDIT_EXPECTATION_SCHEMA_VERSION",
    "EVALUATOR_AUDIT_REPORT_SCHEMA_VERSION",
    "EVALUATOR_FIXTURE_PREDICTIONS_SCHEMA_VERSION",
    "EvaluatorAuditError",
    "EvaluatorAuditExpectation",
    "EvaluatorAuditProvenance",
    "EvaluatorAuditProvenanceError",
    "EvaluatorAuditReport",
    "ExpectedRunSummary",
    "FixturePredictions",
    "RunAuditSummary",
    "collect_evaluator_audit_provenance",
    "evaluator_audit_report_sha256",
    "load_evaluator_audit_expectation",
    "load_fixture_predictions",
    "run_evaluator_audit",
    "write_evaluator_audit",
]
