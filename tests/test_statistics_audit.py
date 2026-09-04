from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import posttrain_lab.statistics.audit as audit_module
from posttrain_lab.statistics import (
    StatisticsAuditError,
    StatisticsAuditProvenanceError,
    load_statistics_audit_expectation,
    run_statistics_audit,
)

FIXTURE_ROOT = Path("tests/fixtures/paired_statistics")
INPUT_PATHS = (
    "tests/fixtures/paired_statistics/panel.json",
    "tests/fixtures/paired_statistics/protocol.json",
    "tests/fixtures/paired_statistics/expectation.json",
)


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def initialize_formal_repository(root: Path) -> None:
    shutil.copytree("src", root / "src")
    (root / "scripts").mkdir()
    shutil.copy2("scripts/audit_paired_statistics.py", root / "scripts")
    (root / "tests/fixtures").mkdir(parents=True)
    shutil.copytree(FIXTURE_ROOT, root / FIXTURE_ROOT)
    for path in (".python-version", "pyproject.toml", "uv.lock"):
        shutil.copy2(path, root / path)
    git(root, "init")
    git(root, "config", "user.email", "d08-audit@example.invalid")
    git(root, "config", "user.name", "D08 Audit")
    git(root, "add", ".")
    git(root, "commit", "-m", "D08 formal fixture")


def run_formal_subprocess(
    root: Path,
    *,
    output: str = "artifact.json",
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "scripts/audit_paired_statistics.py", "--output", output],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_expectation_loader_is_strict_and_rejects_boolean_counts(tmp_path: Path) -> None:
    raw_sha256, expectation = load_statistics_audit_expectation(
        FIXTURE_ROOT / "expectation.json"
    )
    assert len(raw_sha256) == 64
    assert expectation.c1a_randomization_extreme_count == 371

    payload = json.loads((FIXTURE_ROOT / "expectation.json").read_text(encoding="utf-8"))
    payload["c1a_randomization_extreme_count"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StatisticsAuditError, match="integer"):
        load_statistics_audit_expectation(path)
    payload = json.loads((FIXTURE_ROOT / "expectation.json").read_text(encoding="utf-8"))
    payload["unexpected"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StatisticsAuditError, match="invalid keys"):
        load_statistics_audit_expectation(path)


def test_formal_cli_is_deterministic_git_bound_and_text_free(tmp_path: Path) -> None:
    initialize_formal_repository(tmp_path)
    first = run_formal_subprocess(tmp_path, output="first.json")
    second = run_formal_subprocess(tmp_path, output="second.json")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    summary = json.loads(first.stdout)
    artifact = json.loads((tmp_path / "first.json").read_text(encoding="utf-8"))
    assert artifact["passed"]
    assert artifact["failures"] == []
    assert artifact["provenance"]["tracked_inputs_match_git"]
    assert artifact["provenance"]["git_revision"] == git(
        tmp_path, "rev-parse", "HEAD"
    ).stdout.strip()
    assert summary["analysis_report_sha256"] == artifact["analysis"][
        "analysis_report_sha256"
    ]
    raw = (tmp_path / "first.json").read_bytes().lower()
    for forbidden in (
        b'"prompt"',
        b'"generated_text"',
        b'"output_token_ids"',
        b'"reference"',
        b'"sealed_answer"',
    ):
        assert forbidden not in raw


@pytest.mark.parametrize(
    "relative",
    [
        "tests/fixtures/paired_statistics/panel.json",
        "src/posttrain_lab/statistics/inference.py",
        "uv.lock",
    ],
)
def test_formal_cli_rejects_dirty_audited_bytes(tmp_path: Path, relative: str) -> None:
    initialize_formal_repository(tmp_path)
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b"\n")
    result = run_formal_subprocess(tmp_path)
    assert result.returncode != 0
    assert "differs from recorded Git revision" in result.stderr


def test_formal_cli_rejects_nonpreregistered_protocol_even_when_committed(
    tmp_path: Path,
) -> None:
    initialize_formal_repository(tmp_path)
    path = tmp_path / INPUT_PATHS[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bootstrap_repetitions"] = 9_999
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    git(tmp_path, "add", INPUT_PATHS[1])
    git(tmp_path, "commit", "-m", "alter protocol")
    result = run_formal_subprocess(tmp_path)
    assert result.returncode != 0
    assert "exact preregistered" in result.stderr


def test_formal_cli_records_expectation_mismatch_as_failed_audit(tmp_path: Path) -> None:
    initialize_formal_repository(tmp_path)
    path = tmp_path / INPUT_PATHS[2]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["c1a_randomization_extreme_count"] = 388
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    git(tmp_path, "add", INPUT_PATHS[2])
    git(tmp_path, "commit", "-m", "alter expectation")
    result = run_formal_subprocess(tmp_path)
    assert result.returncode == 1
    artifact = json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8"))
    assert not artifact["passed"]
    assert artifact["failures"] == [
        "c1a_randomization_extreme_count differs from frozen expectation"
    ]


def test_formal_api_rejects_runtime_imported_from_another_checkout(tmp_path: Path) -> None:
    initialize_formal_repository(tmp_path)
    with pytest.raises(StatisticsAuditProvenanceError, match="outside audited repository"):
        run_statistics_audit(
            tmp_path,
            cli_source="scripts/audit_paired_statistics.py",
            panel_path=INPUT_PATHS[0],
            protocol_path=INPUT_PATHS[1],
            expectation_path=INPUT_PATHS[2],
        )


def test_loader_consumed_bytes_are_bound_even_if_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_formal_repository(tmp_path)
    runtime_paths = tuple(
        path.relative_to(tmp_path).as_posix()
        for path in (tmp_path / "src/posttrain_lab").rglob("*.py")
    )
    monkeypatch.setattr(audit_module, "_loaded_project_source_paths", lambda root: runtime_paths)
    original_loader = audit_module._load_paired_panel_with_raw_sha256

    def swap_while_loading(path: Path):
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            return original_loader(path)
        finally:
            path.write_bytes(original)

    monkeypatch.setattr(audit_module, "_load_paired_panel_with_raw_sha256", swap_while_loading)
    with pytest.raises(StatisticsAuditProvenanceError, match="consumed bytes"):
        run_statistics_audit(
            tmp_path,
            cli_source="scripts/audit_paired_statistics.py",
            panel_path=INPUT_PATHS[0],
            protocol_path=INPUT_PATHS[1],
            expectation_path=INPUT_PATHS[2],
        )


def test_audit_rejects_symlink_input(tmp_path: Path) -> None:
    initialize_formal_repository(tmp_path)
    alias = tmp_path / "panel-alias.json"
    alias.symlink_to(tmp_path / INPUT_PATHS[0])
    with pytest.raises(StatisticsAuditProvenanceError, match="symlink"):
        run_statistics_audit(
            tmp_path,
            cli_source="scripts/audit_paired_statistics.py",
            panel_path=alias,
            protocol_path=INPUT_PATHS[1],
            expectation_path=INPUT_PATHS[2],
        )
