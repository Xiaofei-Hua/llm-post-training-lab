"""Reproducible D06 synthetic trust-stack audit and Git provenance binding."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .contamination import (
    DEFAULT_CONTAMINATION_POLICY,
    ContaminationMatch,
    MatchKind,
    build_data_manifest,
    quarantine_contaminated_families,
    scan_contamination,
)
from .registry import (
    EMPTY_PARENT_PAYLOAD_LEDGER,
    DataContractError,
    DataRecord,
    FamilySplitPolicy,
    LoadedDataRecords,
    LoadedFamilySplitPolicy,
    LoadedParentPayloadLedger,
    LoadedSourceRegistry,
    LoadedTransformRegistry,
    ParentPayloadLedger,
    SourceRegistry,
    SplitAssignment,
    assign_family_disjoint_splits,
    canonical_json_bytes,
    load_data_records,
    load_family_split_policy,
    load_parent_payload_ledger,
    load_source_registry,
    load_transform_registry,
    strict_json_loads,
    validate_record_set,
    write_json_atomic,
)

DATA_TRUST_EXPECTATION_SCHEMA_VERSION = "d06-data-trust-expectation-v1"
DATA_TRUST_AUDIT_SCHEMA_VERSION = "d06-data-trust-audit-v1"
_MAX_EXPECTATION_BYTES = 16 * 1024**2
_MAX_EXPECTED_MATCHES = 100_000
_MAX_QUARANTINED_RECORDS = 1_000_000

_IMPLEMENTATION_SOURCE_PATHS = (
    "src/posttrain_lab/__init__.py",
    "src/posttrain_lab/data/__init__.py",
    "src/posttrain_lab/data/registry.py",
    "src/posttrain_lab/data/contamination.py",
    "src/posttrain_lab/data/audit.py",
    "scripts/audit_data_trust.py",
)
_DEPENDENCY_LOCK_PATH = "uv.lock"


class DataTrustAuditError(DataContractError):
    """Raised when a formal D06 audit cannot be reproduced safely."""


class DataTrustProvenanceError(RuntimeError):
    """Raised when formal audit bytes are not tied to one Git revision."""


@dataclass(frozen=True, slots=True)
class ExpectedContaminationMatch:
    kind: MatchKind
    train_record_id: str
    train_field: str
    eval_record_id: str
    eval_field: str

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, index: int
    ) -> ExpectedContaminationMatch:
        required = {
            "kind",
            "train_record_id",
            "train_field",
            "eval_record_id",
            "eval_field",
        }
        missing = required - set(raw)
        extra = set(raw) - required
        if missing or extra:
            raise DataTrustAuditError(
                f"expected_matches[{index}] invalid keys; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        try:
            kind = MatchKind(raw["kind"])
        except (TypeError, ValueError) as error:
            raise DataTrustAuditError(
                f"expected_matches[{index}] has invalid kind"
            ) from error
        string_values = {
            field: raw[field]
            for field in required
            if field != "kind"
        }
        if any(not isinstance(value, str) or not value for value in string_values.values()):
            raise DataTrustAuditError(
                f"expected_matches[{index}] identifiers and fields must be non-empty strings"
            )
        return cls(kind=kind, **string_values)

    def to_record(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "train_record_id": self.train_record_id,
            "train_field": self.train_field,
            "eval_record_id": self.eval_record_id,
            "eval_field": self.eval_field,
        }

    @classmethod
    def from_match(cls, match: ContaminationMatch) -> ExpectedContaminationMatch:
        return cls(
            kind=match.kind,
            train_record_id=match.train_record_id,
            train_field=match.train_field,
            eval_record_id=match.eval_record_id,
            eval_field=match.eval_field,
        )


@dataclass(frozen=True, slots=True)
class DataTrustAuditExpectation:
    source_registry_sha256: str
    transform_registry_sha256: str
    split_policy_sha256: str
    parent_ledger_sha256: str
    contamination_policy_sha256: str
    assignment_sha256: str
    dirty_report_sha256: str
    clean_report_sha256: str
    manifest_sha256: str
    manifest_record_set_sha256: str
    split_counts: dict[str, int]
    expected_matches: tuple[ExpectedContaminationMatch, ...]
    quarantined_record_ids: tuple[str, ...]
    clean_training_record_count: int
    evaluation_record_count: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DataTrustAuditExpectation:
        required = {
            "schema_version",
            "source_registry_sha256",
            "transform_registry_sha256",
            "split_policy_sha256",
            "parent_ledger_sha256",
            "contamination_policy_sha256",
            "assignment_sha256",
            "dirty_report_sha256",
            "clean_report_sha256",
            "manifest_sha256",
            "manifest_record_set_sha256",
            "split_counts",
            "expected_matches",
            "quarantined_record_ids",
            "clean_training_record_count",
            "evaluation_record_count",
        }
        missing = required - set(raw)
        extra = set(raw) - required
        if missing or extra:
            raise DataTrustAuditError(
                f"expectation invalid keys; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if raw["schema_version"] != DATA_TRUST_EXPECTATION_SCHEMA_VERSION:
            raise DataTrustAuditError(
                f"unsupported expectation schema: {raw['schema_version']!r}"
            )
        digest_fields = (
            "source_registry_sha256",
            "transform_registry_sha256",
            "split_policy_sha256",
            "parent_ledger_sha256",
            "contamination_policy_sha256",
            "assignment_sha256",
            "dirty_report_sha256",
            "clean_report_sha256",
            "manifest_sha256",
            "manifest_record_set_sha256",
        )
        for field in digest_fields:
            value = raw[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise DataTrustAuditError(f"{field} must be a lowercase SHA-256 digest")
        split_counts = _strict_counts(raw["split_counts"], field="split_counts")
        matches_raw = raw["expected_matches"]
        quarantine_raw = raw["quarantined_record_ids"]
        if not isinstance(matches_raw, list):
            raise DataTrustAuditError("expected_matches must be an array")
        if len(matches_raw) > _MAX_EXPECTED_MATCHES:
            raise DataTrustAuditError(
                f"expected_matches exceeds {_MAX_EXPECTED_MATCHES} entries"
            )
        if not isinstance(quarantine_raw, list) or any(
            not isinstance(item, str) or not item for item in quarantine_raw
        ):
            raise DataTrustAuditError("quarantined_record_ids must be a string array")
        if len(quarantine_raw) > _MAX_QUARANTINED_RECORDS:
            raise DataTrustAuditError(
                f"quarantined_record_ids exceeds {_MAX_QUARANTINED_RECORDS} entries"
            )
        matches = tuple(
            ExpectedContaminationMatch.from_mapping(
                _strict_mapping(item, field=f"expected_matches[{index}]"), index=index
            )
            for index, item in enumerate(matches_raw)
        )
        if tuple(sorted(matches, key=_expected_match_key)) != matches:
            raise DataTrustAuditError("expected_matches must be canonically sorted")
        quarantined = tuple(quarantine_raw)
        if quarantined != tuple(sorted(set(quarantined))):
            raise DataTrustAuditError("quarantined_record_ids must be unique and sorted")
        clean_count = _positive_integer(
            raw["clean_training_record_count"], field="clean_training_record_count"
        )
        eval_count = _positive_integer(
            raw["evaluation_record_count"], field="evaluation_record_count"
        )
        return cls(
            source_registry_sha256=raw["source_registry_sha256"],
            transform_registry_sha256=raw["transform_registry_sha256"],
            split_policy_sha256=raw["split_policy_sha256"],
            parent_ledger_sha256=raw["parent_ledger_sha256"],
            contamination_policy_sha256=raw["contamination_policy_sha256"],
            assignment_sha256=raw["assignment_sha256"],
            dirty_report_sha256=raw["dirty_report_sha256"],
            clean_report_sha256=raw["clean_report_sha256"],
            manifest_sha256=raw["manifest_sha256"],
            manifest_record_set_sha256=raw["manifest_record_set_sha256"],
            split_counts=split_counts,
            expected_matches=matches,
            quarantined_record_ids=quarantined,
            clean_training_record_count=clean_count,
            evaluation_record_count=eval_count,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": DATA_TRUST_EXPECTATION_SCHEMA_VERSION,
            "source_registry_sha256": self.source_registry_sha256,
            "transform_registry_sha256": self.transform_registry_sha256,
            "split_policy_sha256": self.split_policy_sha256,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "contamination_policy_sha256": self.contamination_policy_sha256,
            "assignment_sha256": self.assignment_sha256,
            "dirty_report_sha256": self.dirty_report_sha256,
            "clean_report_sha256": self.clean_report_sha256,
            "manifest_sha256": self.manifest_sha256,
            "manifest_record_set_sha256": self.manifest_record_set_sha256,
            "split_counts": dict(sorted(self.split_counts.items())),
            "expected_matches": [match.to_record() for match in self.expected_matches],
            "quarantined_record_ids": list(self.quarantined_record_ids),
            "clean_training_record_count": self.clean_training_record_count,
            "evaluation_record_count": self.evaluation_record_count,
        }


@dataclass(frozen=True, slots=True)
class LoadedDataTrustExpectation:
    path: str
    raw_sha256: str
    expectation: DataTrustAuditExpectation


def _strict_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DataTrustAuditError(f"{field} must be a string-keyed mapping")
    return value


def _strict_counts(value: object, *, field: str) -> dict[str, int]:
    mapping = _strict_mapping(value, field=field)
    result: dict[str, int] = {}
    for key, count in mapping.items():
        if not key or not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise DataTrustAuditError(
                f"{field} must map non-empty strings to non-negative integers"
            )
        result[key] = count
    return dict(sorted(result.items()))


def _positive_integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DataTrustAuditError(f"{field} must be a positive integer")
    return value


def _expected_match_key(match: ExpectedContaminationMatch) -> tuple[str, ...]:
    return (
        match.train_record_id,
        match.eval_record_id,
        match.train_field,
        match.eval_field,
        match.kind.value,
    )


def load_data_trust_expectation(path: str | Path) -> LoadedDataTrustExpectation:
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
        if size > _MAX_EXPECTATION_BYTES:
            raise DataTrustAuditError(
                f"expectation exceeds {_MAX_EXPECTATION_BYTES} bytes"
            )
        raw = resolved.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise DataTrustAuditError("expectation must not contain a UTF-8 BOM")
        parsed = strict_json_loads(raw.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as error:
        raise DataTrustAuditError("expectation is not strict UTF-8") from error
    except DataContractError as error:
        raise DataTrustAuditError(f"invalid expectation JSON: {error}") from error
    except OSError as error:
        raise DataTrustAuditError(f"cannot read expectation: {error}") from error
    expectation = DataTrustAuditExpectation.from_mapping(
        _strict_mapping(parsed, field="expectation")
    )
    return LoadedDataTrustExpectation(
        path=resolved.as_posix(),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        expectation=expectation,
    )


@dataclass(frozen=True, slots=True)
class DataTrustProvenance:
    git_revision: str
    tracked_inputs_match_git: bool
    tracked_input_paths: tuple[str, ...]
    tracked_files_sha256: dict[str, str]
    source_files_sha256: dict[str, str]
    implementation_source_sha256: str
    dependency_lock_sha256: str

    def formal_validation_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if (
            len(self.git_revision) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in self.git_revision)
        ):
            failures.append("git_revision is not a full lowercase commit digest")
        required_prefix = (*_IMPLEMENTATION_SOURCE_PATHS, _DEPENDENCY_LOCK_PATH)
        if self.tracked_input_paths[: len(required_prefix)] != required_prefix:
            failures.append("tracked_input_paths does not start with the frozen code/lock set")
        if len(self.tracked_input_paths) != len(set(self.tracked_input_paths)):
            failures.append("tracked_input_paths contains duplicates")
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
        if set(self.tracked_files_sha256) != set(self.tracked_input_paths):
            failures.append("tracked file hashes do not cover exactly tracked_input_paths")
        if set(self.source_files_sha256) != set(_IMPLEMENTATION_SOURCE_PATHS):
            failures.append("source file hashes do not cover the frozen implementation set")
        for path, digest in {
            **self.tracked_files_sha256,
            **self.source_files_sha256,
            "implementation": self.implementation_source_sha256,
            "dependency_lock": self.dependency_lock_sha256,
        }.items():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                failures.append(f"invalid provenance SHA-256 for {path}")
                break
        if any(
            self.tracked_files_sha256.get(path) != digest
            for path, digest in self.source_files_sha256.items()
        ):
            failures.append("source hashes disagree with tracked file hashes")
        expected_implementation_hash = hashlib.sha256(
            canonical_json_bytes(dict(sorted(self.source_files_sha256.items())))
        ).hexdigest()
        if self.implementation_source_sha256 != expected_implementation_hash:
            failures.append("implementation_source_sha256 is inconsistent")
        if self.tracked_files_sha256.get(_DEPENDENCY_LOCK_PATH) != (
            self.dependency_lock_sha256
        ):
            failures.append("dependency lock hash disagrees with tracked file hash")
        if self.tracked_inputs_match_git is not True:
            failures.append("tracked inputs do not match Git")
        return tuple(failures)

    def assert_formal(self) -> None:
        failures = self.formal_validation_failures()
        if failures:
            raise DataTrustProvenanceError(f"invalid formal provenance: {failures[0]}")

    def to_record(self) -> dict[str, object]:
        return {
            "git_revision": self.git_revision,
            "tracked_inputs_match_git": self.tracked_inputs_match_git,
            "tracked_input_paths": list(self.tracked_input_paths),
            "tracked_files_sha256": dict(sorted(self.tracked_files_sha256.items())),
            "source_files_sha256": dict(sorted(self.source_files_sha256.items())),
            "implementation_source_sha256": self.implementation_source_sha256,
            "dependency_lock_path": _DEPENDENCY_LOCK_PATH,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }


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
        raise DataTrustProvenanceError(f"cannot execute git: {error}") from error


def _relative_repository_path(repository_root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(repository_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise DataTrustProvenanceError(
            f"audit input must exist inside repository: {path}"
        ) from error
    return relative.as_posix()


def collect_data_trust_provenance(
    repository_root: str | Path,
    *,
    input_paths: Sequence[str | Path],
    require_clean: bool = True,
) -> DataTrustProvenance:
    root = Path(repository_root).resolve(strict=True)
    top_level = _run_git(root, "rev-parse", "--show-toplevel")
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root:
        raise DataTrustProvenanceError("repository_root must be the Git worktree root")
    revision_result = _run_git(root, "rev-parse", "HEAD")
    revision = revision_result.stdout.strip()
    if (
        revision_result.returncode != 0
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise DataTrustProvenanceError("cannot resolve a full Git HEAD revision")

    all_paths: list[str] = []
    for path in (*_IMPLEMENTATION_SOURCE_PATHS, _DEPENDENCY_LOCK_PATH, *input_paths):
        relative = _relative_repository_path(root, path)
        if relative not in all_paths:
            all_paths.append(relative)
    source_paths = tuple(_IMPLEMENTATION_SOURCE_PATHS)
    source_hashes: dict[str, str] = {}
    tracked_hashes: dict[str, str] = {}
    tracked_inputs_match_git = True
    for relative in all_paths:
        tracked = _run_git(root, "ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            raise DataTrustProvenanceError(f"audit input is not Git tracked: {relative}")
        head_bytes = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if head_bytes.returncode != 0:
            raise DataTrustProvenanceError(
                f"cannot read {relative} from recorded Git revision {revision}"
            )
        try:
            working_bytes = (root / relative).read_bytes()
        except OSError as error:
            raise DataTrustProvenanceError(f"cannot read audit input {relative}") from error
        if working_bytes != head_bytes.stdout:
            tracked_inputs_match_git = False
            if require_clean:
                raise DataTrustProvenanceError(
                    f"audit input differs from recorded Git revision: {relative}"
                )
        tracked_hashes[relative] = hashlib.sha256(working_bytes).hexdigest()
        if relative in source_paths:
            source_hashes[relative] = tracked_hashes[relative]
    implementation_hash = hashlib.sha256(
        canonical_json_bytes(dict(sorted(source_hashes.items())))
    ).hexdigest()
    lock_hash = tracked_hashes[_DEPENDENCY_LOCK_PATH]
    return DataTrustProvenance(
        git_revision=revision,
        tracked_inputs_match_git=tracked_inputs_match_git,
        tracked_input_paths=tuple(all_paths),
        tracked_files_sha256=tracked_hashes,
        source_files_sha256=source_hashes,
        implementation_source_sha256=implementation_hash,
        dependency_lock_sha256=lock_hash,
    )


@dataclass(frozen=True, slots=True)
class DataTrustAuditReport:
    provenance: DataTrustProvenance
    input_files_sha256: dict[str, str]
    source_registry_sha256: str
    transform_registry_sha256: str
    split_policy_sha256: str
    parent_ledger_sha256: str
    contamination_policy_sha256: str
    split_assignment: SplitAssignment
    dirty_report_sha256: str
    dirty_match_counts: dict[str, int]
    observed_matches: tuple[ExpectedContaminationMatch, ...]
    observed_match_details: tuple[ContaminationMatch, ...]
    quarantined_record_ids: tuple[str, ...]
    clean_report_sha256: str
    clean_training_record_count: int
    evaluation_record_count: int
    manifest_sha256: str
    manifest_record_set_sha256: str
    manifest_split_counts: dict[str, int]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures and not self.provenance.formal_validation_failures()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": DATA_TRUST_AUDIT_SCHEMA_VERSION,
            "passed": self.passed,
            "provenance": self.provenance.to_record(),
            "input_files_sha256": dict(sorted(self.input_files_sha256.items())),
            "source_registry_sha256": self.source_registry_sha256,
            "transform_registry_sha256": self.transform_registry_sha256,
            "split_policy_sha256": self.split_policy_sha256,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "contamination_policy_sha256": self.contamination_policy_sha256,
            "split_assignment": self.split_assignment.to_record(),
            "dirty_report_sha256": self.dirty_report_sha256,
            "dirty_match_counts": dict(sorted(self.dirty_match_counts.items())),
            "observed_matches": [match.to_record() for match in self.observed_matches],
            "observed_match_details": [
                match.to_record() for match in self.observed_match_details
            ],
            "quarantined_record_ids": list(self.quarantined_record_ids),
            "clean_report_sha256": self.clean_report_sha256,
            "clean_training_record_count": self.clean_training_record_count,
            "evaluation_record_count": self.evaluation_record_count,
            "manifest_sha256": self.manifest_sha256,
            "manifest_record_set_sha256": self.manifest_record_set_sha256,
            "manifest_split_counts": dict(sorted(self.manifest_split_counts.items())),
            "failures": list(self.failures),
        }


def _compare_expectation(
    expectation: DataTrustAuditExpectation,
    *,
    source_registry_sha256: str,
    transform_registry_sha256: str,
    parent_ledger_sha256: str,
    contamination_policy_sha256: str,
    dirty_report_sha256: str,
    clean_report_sha256: str,
    manifest_sha256: str,
    manifest_record_set_sha256: str,
    split_assignment: SplitAssignment,
    observed_matches: tuple[ExpectedContaminationMatch, ...],
    quarantined_record_ids: tuple[str, ...],
    clean_training_record_count: int,
    evaluation_record_count: int,
    split_policy: FamilySplitPolicy,
) -> tuple[str, ...]:
    failures: list[str] = []
    checks = {
        "source_registry_sha256": (
            expectation.source_registry_sha256,
            source_registry_sha256,
        ),
        "transform_registry_sha256": (
            expectation.transform_registry_sha256,
            transform_registry_sha256,
        ),
        "split_policy_sha256": (expectation.split_policy_sha256, split_policy.sha256),
        "parent_ledger_sha256": (
            expectation.parent_ledger_sha256,
            parent_ledger_sha256,
        ),
        "contamination_policy_sha256": (
            expectation.contamination_policy_sha256,
            contamination_policy_sha256,
        ),
        "assignment_sha256": (
            expectation.assignment_sha256,
            split_assignment.assignment_sha256,
        ),
        "split_counts": (expectation.split_counts, split_assignment.split_counts),
        "expected_matches": (expectation.expected_matches, observed_matches),
        "quarantined_record_ids": (
            expectation.quarantined_record_ids,
            quarantined_record_ids,
        ),
        "clean_training_record_count": (
            expectation.clean_training_record_count,
            clean_training_record_count,
        ),
        "evaluation_record_count": (
            expectation.evaluation_record_count,
            evaluation_record_count,
        ),
        "dirty_report_sha256": (
            expectation.dirty_report_sha256,
            dirty_report_sha256,
        ),
        "clean_report_sha256": (
            expectation.clean_report_sha256,
            clean_report_sha256,
        ),
        "manifest_sha256": (expectation.manifest_sha256, manifest_sha256),
        "manifest_record_set_sha256": (
            expectation.manifest_record_set_sha256,
            manifest_record_set_sha256,
        ),
    }
    for field, (expected, observed) in checks.items():
        if expected != observed:
            failures.append(f"{field} mismatch: expected={expected!r}, observed={observed!r}")
    return tuple(failures)


def _run_data_trust_audit_core(
    *,
    loaded_registry: LoadedSourceRegistry,
    loaded_transform_registry: LoadedTransformRegistry,
    loaded_candidates: LoadedDataRecords,
    loaded_evaluation: LoadedDataRecords,
    loaded_split_policy: LoadedFamilySplitPolicy,
    loaded_parent_ledger: LoadedParentPayloadLedger | None,
    loaded_expectation: LoadedDataTrustExpectation,
    provenance: DataTrustProvenance,
) -> DataTrustAuditReport:
    registry: SourceRegistry = loaded_registry.registry
    transform_registry = loaded_transform_registry.registry
    split_policy = loaded_split_policy.policy
    parent_ledger: ParentPayloadLedger = (
        EMPTY_PARENT_PAYLOAD_LEDGER
        if loaded_parent_ledger is None
        else loaded_parent_ledger.ledger
    )
    assigned, assignment = assign_family_disjoint_splits(
        loaded_candidates.records, split_policy
    )
    validate_record_set(
        assigned,
        registry,
        transform_registry=transform_registry,
        parent_ledger=parent_ledger,
    )
    validate_record_set(
        loaded_evaluation.records,
        registry,
        transform_registry=transform_registry,
        parent_ledger=parent_ledger,
    )

    dirty_report = scan_contamination(
        assigned,
        loaded_evaluation.records,
        policy=DEFAULT_CONTAMINATION_POLICY,
    )
    observed_matches = tuple(
        sorted(
            (ExpectedContaminationMatch.from_match(match) for match in dirty_report.matches),
            key=_expected_match_key,
        )
    )
    clean_training, quarantined = quarantine_contaminated_families(
        assigned, dirty_report
    )
    quarantined_ids = tuple(record.sample_id for record in quarantined)
    clean_report = scan_contamination(
        clean_training,
        loaded_evaluation.records,
        policy=DEFAULT_CONTAMINATION_POLICY,
    )
    clean_report.assert_clean()
    combined_records: tuple[DataRecord, ...] = (
        *clean_training,
        *loaded_evaluation.records,
    )
    validate_record_set(
        combined_records,
        registry,
        transform_registry=transform_registry,
        parent_ledger=parent_ledger,
    )
    manifest = build_data_manifest(
        combined_records,
        registry,
        transform_registry=transform_registry,
        split_policy_sha256=split_policy.sha256,
        policy=DEFAULT_CONTAMINATION_POLICY,
        parent_ledger=parent_ledger,
    )
    failures = _compare_expectation(
        loaded_expectation.expectation,
        source_registry_sha256=registry.sha256,
        transform_registry_sha256=transform_registry.sha256,
        parent_ledger_sha256=parent_ledger.sha256,
        contamination_policy_sha256=DEFAULT_CONTAMINATION_POLICY.sha256,
        dirty_report_sha256=dirty_report.report_sha256,
        clean_report_sha256=clean_report.report_sha256,
        manifest_sha256=manifest.manifest_sha256,
        manifest_record_set_sha256=manifest.record_set_sha256,
        split_assignment=assignment,
        observed_matches=observed_matches,
        quarantined_record_ids=quarantined_ids,
        clean_training_record_count=len(clean_training),
        evaluation_record_count=len(loaded_evaluation.records),
        split_policy=split_policy,
    )
    input_files_sha256 = {
        loaded_registry.path: loaded_registry.raw_sha256,
        loaded_transform_registry.path: loaded_transform_registry.raw_sha256,
        loaded_candidates.path: loaded_candidates.raw_sha256,
        loaded_evaluation.path: loaded_evaluation.raw_sha256,
        loaded_split_policy.path: loaded_split_policy.raw_sha256,
        loaded_expectation.path: loaded_expectation.raw_sha256,
    }
    if loaded_parent_ledger is not None:
        input_files_sha256[loaded_parent_ledger.path] = loaded_parent_ledger.raw_sha256
    return DataTrustAuditReport(
        provenance=provenance,
        input_files_sha256=dict(sorted(input_files_sha256.items())),
        source_registry_sha256=registry.sha256,
        transform_registry_sha256=transform_registry.sha256,
        split_policy_sha256=split_policy.sha256,
        parent_ledger_sha256=parent_ledger.sha256,
        contamination_policy_sha256=DEFAULT_CONTAMINATION_POLICY.sha256,
        split_assignment=assignment,
        dirty_report_sha256=dirty_report.report_sha256,
        dirty_match_counts=dirty_report.match_counts,
        observed_matches=observed_matches,
        observed_match_details=dirty_report.matches,
        quarantined_record_ids=quarantined_ids,
        clean_report_sha256=clean_report.report_sha256,
        clean_training_record_count=len(clean_training),
        evaluation_record_count=len(loaded_evaluation.records),
        manifest_sha256=manifest.manifest_sha256,
        manifest_record_set_sha256=manifest.record_set_sha256,
        manifest_split_counts=manifest.split_counts,
        failures=failures,
    )


def run_data_trust_audit(
    repository_root: str | Path,
    *,
    source_registry_path: str | Path,
    transform_registry_path: str | Path,
    candidate_records_path: str | Path,
    evaluation_records_path: str | Path,
    split_policy_path: str | Path,
    expectation_path: str | Path,
    parent_ledger_path: str | Path | None = None,
) -> DataTrustAuditReport:
    """Load and audit one Git-tracked snapshot without caller-supplied provenance."""

    root = Path(repository_root).resolve(strict=True)
    required_paths = (
        source_registry_path,
        transform_registry_path,
        candidate_records_path,
        evaluation_records_path,
        split_policy_path,
        expectation_path,
    )
    requested_paths = (
        required_paths
        if parent_ledger_path is None
        else (*required_paths, parent_ledger_path)
    )
    relative_paths = tuple(
        _relative_repository_path(root, path) for path in requested_paths
    )
    loaded_transform_registry = replace(
        load_transform_registry(root / relative_paths[1]), path=relative_paths[1]
    )
    transform_artifact_paths = tuple(
        path
        for artifact in loaded_transform_registry.registry.artifacts
        for path in (artifact.code_path, artifact.config_path)
    )
    provenance = collect_data_trust_provenance(
        root,
        input_paths=(*relative_paths, *transform_artifact_paths),
        require_clean=True,
    )
    provenance.assert_formal()

    loaded_registry = replace(
        load_source_registry(root / relative_paths[0]), path=relative_paths[0]
    )
    loaded_candidates = replace(
        load_data_records(root / relative_paths[2]), path=relative_paths[2]
    )
    loaded_evaluation = replace(
        load_data_records(root / relative_paths[3]), path=relative_paths[3]
    )
    loaded_split_policy = replace(
        load_family_split_policy(root / relative_paths[4]), path=relative_paths[4]
    )
    loaded_expectation = replace(
        load_data_trust_expectation(root / relative_paths[5]), path=relative_paths[5]
    )
    loaded_parent_ledger = (
        None
        if parent_ledger_path is None
        else replace(
            load_parent_payload_ledger(root / relative_paths[6]), path=relative_paths[6]
        )
    )
    loaded_hashes = {
        loaded_registry.path: loaded_registry.raw_sha256,
        loaded_transform_registry.path: loaded_transform_registry.raw_sha256,
        loaded_candidates.path: loaded_candidates.raw_sha256,
        loaded_evaluation.path: loaded_evaluation.raw_sha256,
        loaded_split_policy.path: loaded_split_policy.raw_sha256,
        loaded_expectation.path: loaded_expectation.raw_sha256,
    }
    if loaded_parent_ledger is not None:
        loaded_hashes[loaded_parent_ledger.path] = loaded_parent_ledger.raw_sha256
    for artifact in loaded_transform_registry.registry.artifacts:
        for path, declared_sha256 in (
            (artifact.code_path, artifact.code_sha256),
            (artifact.config_path, artifact.config_sha256),
        ):
            if provenance.tracked_files_sha256.get(path) != declared_sha256:
                raise DataTrustProvenanceError(
                    f"transform artifact hash does not match Git bytes: {path}"
                )
    for relative, raw_sha256 in loaded_hashes.items():
        if provenance.tracked_files_sha256.get(relative) != raw_sha256:
            raise DataTrustProvenanceError(
                f"audit input changed while it was being loaded: {relative}"
            )
    return _run_data_trust_audit_core(
        loaded_registry=loaded_registry,
        loaded_transform_registry=loaded_transform_registry,
        loaded_candidates=loaded_candidates,
        loaded_evaluation=loaded_evaluation,
        loaded_split_policy=loaded_split_policy,
        loaded_parent_ledger=loaded_parent_ledger,
        loaded_expectation=loaded_expectation,
        provenance=provenance,
    )


def write_data_trust_audit(
    report: DataTrustAuditReport,
    output_path: str | Path,
) -> None:
    write_json_atomic(report.to_record(), output_path)


def audit_report_sha256(report: DataTrustAuditReport) -> str:
    return hashlib.sha256(canonical_json_bytes(report.to_record())).hexdigest()
