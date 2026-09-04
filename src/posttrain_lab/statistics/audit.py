"""Git-bound synthetic audit for the D08 paired-statistics implementation."""

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

import numpy as np

from posttrain_lab.data import canonical_json_bytes, strict_json_loads, write_json_atomic

from .contracts import (
    _load_paired_panel_with_raw_sha256,
    _load_statistics_protocol_with_raw_sha256,
)
from .inference import ConfirmatoryAnalysisReport, run_confirmatory_analysis

STATISTICS_AUDIT_EXPECTATION_SCHEMA_VERSION = "d08-statistics-audit-expectation-v1"
STATISTICS_AUDIT_REPORT_SCHEMA_VERSION = "d08-statistics-audit-report-v1"

_MAX_CONTROL_FILE_BYTES = 16 * 1024**2
_ENVIRONMENT_PATHS = (".python-version", "pyproject.toml", "uv.lock")


class StatisticsAuditError(ValueError):
    """Raised when a D08 audit expectation is malformed."""


class StatisticsAuditProvenanceError(RuntimeError):
    """Raised when audit execution cannot be tied to one clean Git revision."""


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise StatisticsAuditError(f"{field} must be Boolean")
    return value


def _require_int(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise StatisticsAuditError(f"{field} must be an integer in {interval}")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StatisticsAuditError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StatisticsAuditError(f"{field} must be a string-keyed mapping")
    return value


def _require_exact_keys(raw: Mapping[str, Any], *, required: set[str], field: str) -> None:
    missing = required - set(raw)
    extra = set(raw) - required
    if missing or extra:
        raise StatisticsAuditError(
            f"{field} has invalid keys; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _load_json_object(path: Path, *, field: str) -> tuple[str, Mapping[str, Any]]:
    try:
        size = path.stat().st_size
        if size > _MAX_CONTROL_FILE_BYTES:
            raise StatisticsAuditError(f"{field} exceeds {_MAX_CONTROL_FILE_BYTES} bytes")
        raw = path.read_bytes()
    except OSError as error:
        raise StatisticsAuditError(f"cannot read {field}: {error}") from error
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise StatisticsAuditError(f"{field} must be canonical UTF-8 without BOM/CR")
    try:
        parsed = strict_json_loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError) as error:
        raise StatisticsAuditError(f"invalid strict {field}: {error}") from error
    return hashlib.sha256(raw).hexdigest(), _require_mapping(parsed, field=field)


@dataclass(frozen=True, slots=True)
class StatisticsAuditExpectation:
    panel_sha256: str
    statistics_protocol_sha256: str
    analysis_report_sha256: str
    c1a_point_ppm: int
    c1a_randomization_extreme_count: int
    c1a_holm_adjusted_p_ppm: int
    c1a_practical_success: bool
    c1b_point_ppm: int
    c1b_randomization_extreme_count: int
    c1b_holm_adjusted_p_ppm: int
    c1b_practical_success: bool
    c2_point_ppm: int
    c2_classification: str
    c2_equivalence_assessed: bool

    def __post_init__(self) -> None:
        for field in (
            "panel_sha256",
            "statistics_protocol_sha256",
            "analysis_report_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        for field in (
            "c1a_randomization_extreme_count",
            "c1b_randomization_extreme_count",
        ):
            _require_int(getattr(self, field), field=field, minimum=0, maximum=10_000_000)
        for field in ("c1a_holm_adjusted_p_ppm", "c1b_holm_adjusted_p_ppm"):
            _require_int(getattr(self, field), field=field, minimum=0, maximum=1_000_000)
        for field in ("c1a_point_ppm", "c1b_point_ppm", "c2_point_ppm"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -1_000_000 <= value <= 1_000_000
            ):
                raise StatisticsAuditError(f"{field} must be signed integer ppm")
        for field in (
            "c1a_practical_success",
            "c1b_practical_success",
            "c2_equivalence_assessed",
        ):
            _require_bool(getattr(self, field), field=field)
        if self.c2_classification not in {
            "superior_A3",
            "superior_A4",
            "practical_equivalence",
            "inconclusive",
        }:
            raise StatisticsAuditError("c2_classification is invalid")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StatisticsAuditExpectation:
        required = {
            "schema_version",
            "panel_sha256",
            "statistics_protocol_sha256",
            "analysis_report_sha256",
            "c1a_point_ppm",
            "c1a_randomization_extreme_count",
            "c1a_holm_adjusted_p_ppm",
            "c1a_practical_success",
            "c1b_point_ppm",
            "c1b_randomization_extreme_count",
            "c1b_holm_adjusted_p_ppm",
            "c1b_practical_success",
            "c2_point_ppm",
            "c2_classification",
            "c2_equivalence_assessed",
        }
        _require_exact_keys(raw, required=required, field="statistics audit expectation")
        if raw["schema_version"] != STATISTICS_AUDIT_EXPECTATION_SCHEMA_VERSION:
            raise StatisticsAuditError("unsupported statistics-audit expectation schema")
        return cls(
            panel_sha256=raw["panel_sha256"],
            statistics_protocol_sha256=raw["statistics_protocol_sha256"],
            analysis_report_sha256=raw["analysis_report_sha256"],
            c1a_point_ppm=raw["c1a_point_ppm"],
            c1a_randomization_extreme_count=raw["c1a_randomization_extreme_count"],
            c1a_holm_adjusted_p_ppm=raw["c1a_holm_adjusted_p_ppm"],
            c1a_practical_success=raw["c1a_practical_success"],
            c1b_point_ppm=raw["c1b_point_ppm"],
            c1b_randomization_extreme_count=raw["c1b_randomization_extreme_count"],
            c1b_holm_adjusted_p_ppm=raw["c1b_holm_adjusted_p_ppm"],
            c1b_practical_success=raw["c1b_practical_success"],
            c2_point_ppm=raw["c2_point_ppm"],
            c2_classification=raw["c2_classification"],
            c2_equivalence_assessed=raw["c2_equivalence_assessed"],
        )

    def expected_summary(self) -> dict[str, object]:
        return {
            "panel_sha256": self.panel_sha256,
            "statistics_protocol_sha256": self.statistics_protocol_sha256,
            "analysis_report_sha256": self.analysis_report_sha256,
            "c1a_point_ppm": self.c1a_point_ppm,
            "c1a_randomization_extreme_count": self.c1a_randomization_extreme_count,
            "c1a_holm_adjusted_p_ppm": self.c1a_holm_adjusted_p_ppm,
            "c1a_practical_success": self.c1a_practical_success,
            "c1b_point_ppm": self.c1b_point_ppm,
            "c1b_randomization_extreme_count": self.c1b_randomization_extreme_count,
            "c1b_holm_adjusted_p_ppm": self.c1b_holm_adjusted_p_ppm,
            "c1b_practical_success": self.c1b_practical_success,
            "c2_point_ppm": self.c2_point_ppm,
            "c2_classification": self.c2_classification,
            "c2_equivalence_assessed": self.c2_equivalence_assessed,
        }


def load_statistics_audit_expectation(
    path: str | Path,
) -> tuple[str, StatisticsAuditExpectation]:
    raw_sha256, parsed = _load_json_object(Path(path), field="statistics audit expectation")
    return raw_sha256, StatisticsAuditExpectation.from_mapping(parsed)


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def _repository_relative_path(root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise StatisticsAuditProvenanceError(
            f"audit path must resolve inside repository: {path}"
        ) from error
    if candidate.is_symlink():
        raise StatisticsAuditProvenanceError(f"audit path must not be a symlink: {path}")
    return relative.as_posix()


def _loaded_project_source_paths(root: Path) -> tuple[str, ...]:
    paths: set[str] = set()
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "posttrain_lab" and not module_name.startswith("posttrain_lab."):
            continue
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            raise StatisticsAuditProvenanceError(
                f"runtime module source is unavailable: {module_name}"
            )
        try:
            actual = Path(source).resolve(strict=True)
            relative = actual.relative_to(root)
        except (OSError, ValueError) as error:
            raise StatisticsAuditProvenanceError(
                f"runtime module source is outside audited repository: {module_name}"
            ) from error
        if actual.suffix != ".py":
            raise StatisticsAuditProvenanceError(
                f"runtime module does not resolve to tracked Python source: {module_name}"
            )
        paths.add(relative.as_posix())
    required = {
        "src/posttrain_lab/statistics/__init__.py",
        "src/posttrain_lab/statistics/contracts.py",
        "src/posttrain_lab/statistics/inference.py",
        "src/posttrain_lab/statistics/audit.py",
    }
    if not required.issubset(paths):
        missing = sorted(required - paths)
        raise StatisticsAuditProvenanceError(
            f"runtime statistics source closure is incomplete: {missing}"
        )
    return tuple(sorted(paths))


@dataclass(frozen=True, slots=True)
class StatisticsAuditProvenance:
    git_revision: str
    tracked_inputs_match_git: bool
    implementation_source_paths: tuple[str, ...]
    tracked_input_paths: tuple[str, ...]
    tracked_files_sha256: Mapping[str, str]
    implementation_source_sha256: str
    runtime_versions: Mapping[str, str]

    def formal_validation_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if not self.tracked_inputs_match_git:
            failures.append("tracked inputs do not match Git revision")
        if not self.git_revision or len(self.git_revision) not in {40, 64}:
            failures.append("Git revision is not full-length")
        return tuple(failures)

    def assert_formal(self) -> None:
        failures = self.formal_validation_failures()
        if failures:
            raise StatisticsAuditProvenanceError("; ".join(failures))

    def to_record(self) -> dict[str, object]:
        return {
            "git_revision": self.git_revision,
            "tracked_inputs_match_git": self.tracked_inputs_match_git,
            "implementation_source_paths": list(self.implementation_source_paths),
            "tracked_input_paths": list(self.tracked_input_paths),
            "tracked_files_sha256": dict(sorted(self.tracked_files_sha256.items())),
            "implementation_source_sha256": self.implementation_source_sha256,
            "runtime_versions": dict(sorted(self.runtime_versions.items())),
        }


def collect_statistics_audit_provenance(
    repository_root: str | Path,
    *,
    cli_source: str | Path,
    input_paths: Sequence[str | Path],
    require_clean: bool = True,
) -> StatisticsAuditProvenance:
    """Bind runtime sources, environment, fixtures, and expectations to Git HEAD."""

    root = Path(repository_root).resolve(strict=True)
    top_level = _run_git(root, "rev-parse", "--show-toplevel")
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root:
        raise StatisticsAuditProvenanceError("repository_root must be the Git worktree root")
    revision_result = _run_git(root, "rev-parse", "HEAD")
    revision = revision_result.stdout.strip()
    if (
        revision_result.returncode != 0
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise StatisticsAuditProvenanceError("cannot resolve a full Git HEAD revision")
    runtime_sources = _loaded_project_source_paths(root)
    implementation_paths = tuple(
        dict.fromkeys(
            (
                *runtime_sources,
                _repository_relative_path(root, cli_source),
                *(_repository_relative_path(root, path) for path in _ENVIRONMENT_PATHS),
            )
        )
    )
    all_paths = tuple(
        dict.fromkeys(
            (
                *implementation_paths,
                *(_repository_relative_path(root, path) for path in input_paths),
            )
        )
    )
    tracked_hashes: dict[str, str] = {}
    tracked_match = True
    for relative in all_paths:
        tracked = _run_git(root, "ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            raise StatisticsAuditProvenanceError(f"audit input is not Git tracked: {relative}")
        head_bytes = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if head_bytes.returncode != 0:
            raise StatisticsAuditProvenanceError(
                f"cannot read {relative} from recorded revision {revision}"
            )
        try:
            working_bytes = (root / relative).read_bytes()
        except OSError as error:
            raise StatisticsAuditProvenanceError(
                f"cannot read audit input {relative}: {error}"
            ) from error
        if working_bytes != head_bytes.stdout:
            tracked_match = False
            if require_clean:
                raise StatisticsAuditProvenanceError(
                    f"audit input differs from recorded Git revision: {relative}"
                )
        tracked_hashes[relative] = hashlib.sha256(working_bytes).hexdigest()
    final_revision = _run_git(root, "rev-parse", "HEAD")
    if final_revision.returncode != 0 or final_revision.stdout.strip() != revision:
        raise StatisticsAuditProvenanceError("Git HEAD changed while provenance was collected")
    implementation_hash = hashlib.sha256(
        canonical_json_bytes(
            {path: tracked_hashes[path] for path in sorted(implementation_paths)}
        )
    ).hexdigest()
    return StatisticsAuditProvenance(
        git_revision=revision,
        tracked_inputs_match_git=tracked_match,
        implementation_source_paths=tuple(sorted(implementation_paths)),
        tracked_input_paths=all_paths,
        tracked_files_sha256=tracked_hashes,
        implementation_source_sha256=implementation_hash,
        runtime_versions={
            "numpy": np.__version__,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "unicode_database": unicodedata.unidata_version,
        },
    )


def _assert_inputs_unchanged(root: Path, provenance: StatisticsAuditProvenance) -> None:
    for relative, expected_sha256 in provenance.tracked_files_sha256.items():
        try:
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as error:
            raise StatisticsAuditProvenanceError(
                f"cannot reread audit input {relative}: {error}"
            ) from error
        if actual != expected_sha256:
            raise StatisticsAuditProvenanceError(
                f"audit input changed during execution: {relative}"
            )


def _assert_head_unchanged(root: Path, provenance: StatisticsAuditProvenance) -> None:
    current = _run_git(root, "rev-parse", "HEAD")
    if current.returncode != 0 or current.stdout.strip() != provenance.git_revision:
        raise StatisticsAuditProvenanceError("Git HEAD changed during statistics audit")


def _assert_consumed_digest(
    provenance: StatisticsAuditProvenance,
    *,
    relative: str,
    actual_sha256: str,
) -> None:
    if provenance.tracked_files_sha256.get(relative) != actual_sha256:
        raise StatisticsAuditProvenanceError(
            f"audit loader consumed bytes outside captured provenance: {relative}"
        )


def _observed_summary(report: ConfirmatoryAnalysisReport) -> dict[str, object]:
    c1a, c1b = report.c1_results
    return {
        "panel_sha256": report.panel_sha256,
        "statistics_protocol_sha256": report.statistics_protocol_sha256,
        "analysis_report_sha256": report.analysis_report_sha256,
        "c1a_point_ppm": c1a.point_estimate.ppm,
        "c1a_randomization_extreme_count": c1a.randomization.extreme_count,
        "c1a_holm_adjusted_p_ppm": c1a.holm_adjusted_p_value.ppm,
        "c1a_practical_success": c1a.practical_success,
        "c1b_point_ppm": c1b.point_estimate.ppm,
        "c1b_randomization_extreme_count": c1b.randomization.extreme_count,
        "c1b_holm_adjusted_p_ppm": c1b.holm_adjusted_p_value.ppm,
        "c1b_practical_success": c1b.practical_success,
        "c2_point_ppm": report.c2_result.point_estimate.ppm,
        "c2_classification": report.c2_result.classification.value,
        "c2_equivalence_assessed": report.c2_result.equivalence_assessed,
    }


@dataclass(frozen=True, slots=True)
class StatisticsAuditReport:
    provenance: StatisticsAuditProvenance
    panel_file_sha256: str
    protocol_file_sha256: str
    expectation_file_sha256: str
    analysis: ConfirmatoryAnalysisReport
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures and not self.provenance.formal_validation_failures()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": STATISTICS_AUDIT_REPORT_SCHEMA_VERSION,
            "passed": self.passed,
            "provenance": self.provenance.to_record(),
            "panel_file_sha256": self.panel_file_sha256,
            "protocol_file_sha256": self.protocol_file_sha256,
            "expectation_file_sha256": self.expectation_file_sha256,
            "analysis": self.analysis.to_record(),
            "failures": list(self.failures),
        }


def run_statistics_audit(
    repository_root: str | Path,
    *,
    cli_source: str | Path,
    panel_path: str | Path,
    protocol_path: str | Path,
    expectation_path: str | Path,
) -> StatisticsAuditReport:
    """Run the formal full-repetition D08 synthetic oracle."""

    root = Path(repository_root).resolve(strict=True)
    requested = (panel_path, protocol_path, expectation_path)
    relative = tuple(_repository_relative_path(root, path) for path in requested)
    provenance = collect_statistics_audit_provenance(
        root,
        cli_source=cli_source,
        input_paths=relative,
        require_clean=True,
    )
    provenance.assert_formal()
    _assert_inputs_unchanged(root, provenance)
    panel_file_sha256, panel = _load_paired_panel_with_raw_sha256(root / relative[0])
    _assert_consumed_digest(
        provenance,
        relative=relative[0],
        actual_sha256=panel_file_sha256,
    )
    protocol_file_sha256, protocol = _load_statistics_protocol_with_raw_sha256(
        root / relative[1]
    )
    _assert_consumed_digest(
        provenance,
        relative=relative[1],
        actual_sha256=protocol_file_sha256,
    )
    expectation_file_sha256, expectation = load_statistics_audit_expectation(
        root / relative[2]
    )
    _assert_consumed_digest(
        provenance,
        relative=relative[2],
        actual_sha256=expectation_file_sha256,
    )
    protocol.assert_preregistered()
    analysis = run_confirmatory_analysis(panel, protocol, require_preregistered=True)
    expected = expectation.expected_summary()
    observed = _observed_summary(analysis)
    failures = tuple(
        f"{field} differs from frozen expectation"
        for field in expected
        if observed[field] != expected[field]
    )
    report = StatisticsAuditReport(
        provenance=provenance,
        panel_file_sha256=panel_file_sha256,
        protocol_file_sha256=protocol_file_sha256,
        expectation_file_sha256=expectation_file_sha256,
        analysis=analysis,
        failures=failures,
    )
    _assert_inputs_unchanged(root, provenance)
    _assert_head_unchanged(root, provenance)
    return report


def write_statistics_audit(report: StatisticsAuditReport, path: str | Path) -> None:
    write_json_atomic(report.to_record(), path)


def statistics_audit_report_sha256(report: StatisticsAuditReport) -> str:
    return hashlib.sha256(canonical_json_bytes(report.to_record())).hexdigest()


__all__ = [
    "STATISTICS_AUDIT_EXPECTATION_SCHEMA_VERSION",
    "STATISTICS_AUDIT_REPORT_SCHEMA_VERSION",
    "StatisticsAuditError",
    "StatisticsAuditExpectation",
    "StatisticsAuditProvenance",
    "StatisticsAuditProvenanceError",
    "StatisticsAuditReport",
    "collect_statistics_audit_provenance",
    "load_statistics_audit_expectation",
    "run_statistics_audit",
    "statistics_audit_report_sha256",
    "write_statistics_audit",
]
