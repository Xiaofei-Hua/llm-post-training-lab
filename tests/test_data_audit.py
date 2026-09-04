from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import pytest

import posttrain_lab.data.audit as audit_module
from posttrain_lab.data import (
    DATA_TRUST_AUDIT_SCHEMA_VERSION,
    DATA_TRUST_EXPECTATION_SCHEMA_VERSION,
    DataTrustAuditError,
    DataTrustProvenance,
    DataTrustProvenanceError,
    LoadedDataTrustExpectation,
    audit_report_sha256,
    collect_data_trust_provenance,
    load_data_records,
    load_data_trust_expectation,
    load_family_split_policy,
    load_source_registry,
    load_transform_registry,
    run_data_trust_audit,
    write_data_trust_audit,
)
from posttrain_lab.data.audit import _run_data_trust_audit_core

FIXTURE_ROOT = Path("tests/fixtures/data_trust")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def dummy_provenance() -> DataTrustProvenance:
    source_paths = (
        "src/posttrain_lab/__init__.py",
        "src/posttrain_lab/data/__init__.py",
        "src/posttrain_lab/data/registry.py",
        "src/posttrain_lab/data/contamination.py",
        "src/posttrain_lab/data/audit.py",
        "scripts/audit_data_trust.py",
    )
    source_hashes = {path: digest(path) for path in source_paths}
    lock_hash = digest("uv.lock")
    tracked_paths = (*source_paths, "uv.lock", "synthetic")
    tracked_hashes = {
        **source_hashes,
        "uv.lock": lock_hash,
        "synthetic": digest("synthetic"),
    }
    implementation_hash = hashlib.sha256(
        json.dumps(
            source_hashes,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return DataTrustProvenance(
        git_revision="0" * 40,
        tracked_inputs_match_git=True,
        tracked_input_paths=tracked_paths,
        tracked_files_sha256=tracked_hashes,
        source_files_sha256=source_hashes,
        implementation_source_sha256=implementation_hash,
        dependency_lock_sha256=lock_hash,
    )


def run_fixture_audit(
    *,
    expectation: LoadedDataTrustExpectation | None = None,
    provenance: DataTrustProvenance | None = None,
):
    return _run_data_trust_audit_core(
        loaded_registry=load_source_registry(FIXTURE_ROOT / "source_registry.json"),
        loaded_transform_registry=load_transform_registry(
            FIXTURE_ROOT / "transform_registry.json"
        ),
        loaded_candidates=load_data_records(FIXTURE_ROOT / "candidate_records.jsonl"),
        loaded_evaluation=load_data_records(FIXTURE_ROOT / "evaluation_records.jsonl"),
        loaded_split_policy=load_family_split_policy(FIXTURE_ROOT / "split_policy.json"),
        loaded_parent_ledger=None,
        loaded_expectation=expectation
        or load_data_trust_expectation(FIXTURE_ROOT / "expectation.json"),
        provenance=provenance or dummy_provenance(),
    )


def test_frozen_data_trust_fixture_passes_all_expected_gates() -> None:
    report = run_fixture_audit()
    assert report.passed
    assert report.failures == ()
    assert report.dirty_match_counts == {"exact": 1, "fuzzy": 2, "review": 0}
    assert report.quarantined_record_ids == tuple(
        f"train:{index:03d}" for index in range(6)
    )
    assert report.clean_training_record_count == 9
    assert report.evaluation_record_count == 4
    assert report.split_assignment.component_count == 12
    assert set(report.split_assignment.split_counts) == {
        "D_anchor",
        "D_core",
        "D_dev",
        "D_select",
        "D_teacher_gate",
    }
    assert report.manifest_split_counts["E"] == 4
    assert len(report.manifest_sha256) == 64


def test_audit_expectation_mismatch_is_a_structured_failure() -> None:
    loaded = load_data_trust_expectation(FIXTURE_ROOT / "expectation.json")
    wrong = replace(loaded.expectation, assignment_sha256=digest("wrong-assignment"))
    report = run_fixture_audit(
        expectation=replace(loaded, expectation=wrong)
    )
    assert not report.passed
    assert len(report.failures) == 1
    assert report.failures[0].startswith("assignment_sha256 mismatch")


@pytest.mark.parametrize(
    "mutation",
    (
        {"tracked_inputs_match_git": False},
        {"git_revision": "not-a-commit"},
        {"dependency_lock_sha256": "0" * 64},
    ),
)
def test_audit_report_cannot_claim_pass_with_forged_provenance(
    mutation: dict[str, object],
) -> None:
    provenance = replace(dummy_provenance(), **mutation)
    report = run_fixture_audit(provenance=provenance)
    assert not report.passed
    with pytest.raises(DataTrustProvenanceError):
        provenance.assert_formal()


def test_audit_report_is_deterministic_parseable_and_raw_text_free(tmp_path: Path) -> None:
    report = run_fixture_audit()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_data_trust_audit(report, first)
    write_data_trust_audit(report, second)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema_version"] == DATA_TRUST_AUDIT_SCHEMA_VERSION
    assert payload["passed"] is True
    assert payload["dirty_match_counts"] == {"exact": 1, "fuzzy": 2, "review": 0}
    assert payload["provenance"]["tracked_inputs_match_git"] is True
    assert audit_report_sha256(report) == hashlib.sha256(
        json.dumps(
            report.to_record(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    encoded = first.read_bytes()
    assert b"merchant has forty red boxes" not in encoded.lower()
    assert b"positive integers a and b" not in encoded.lower()


def test_expectation_loader_tracks_raw_hash_and_schema(tmp_path: Path) -> None:
    source = FIXTURE_ROOT / "expectation.json"
    raw = source.read_bytes()
    copied = tmp_path / "expectation.json"
    copied.write_bytes(raw)
    loaded = load_data_trust_expectation(copied)
    assert loaded.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert loaded.expectation.to_record()["schema_version"] == (
        DATA_TRUST_EXPECTATION_SCHEMA_VERSION
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update({"extra": True}), "invalid keys"),
        (lambda raw: raw.update({"schema_version": "future"}), "unsupported"),
        (lambda raw: raw.update({"assignment_sha256": "short"}), "SHA-256"),
        (lambda raw: raw.update({"clean_training_record_count": True}), "positive integer"),
        (lambda raw: raw.update({"quarantined_record_ids": ["z", "a"]}), "sorted"),
    ],
)
def test_expectation_loader_rejects_schema_violations(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    raw = json.loads((FIXTURE_ROOT / "expectation.json").read_text(encoding="utf-8"))
    mutation(raw)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataTrustAuditError, match=message):
        load_data_trust_expectation(path)


def test_expectation_loader_rejects_bom_and_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "expectation.json"
    path.write_bytes(b"\xef\xbb\xbf{}")
    with pytest.raises(DataTrustAuditError, match="BOM"):
        load_data_trust_expectation(path)
    path.write_bytes(b"\xff")
    with pytest.raises(DataTrustAuditError, match="UTF-8"):
        load_data_trust_expectation(path)


def initialize_provenance_repository(tmp_path: Path) -> tuple[str, ...]:
    paths = (
        "src/posttrain_lab/__init__.py",
        "src/posttrain_lab/data/__init__.py",
        "src/posttrain_lab/data/registry.py",
        "src/posttrain_lab/data/contamination.py",
        "src/posttrain_lab/data/audit.py",
        "scripts/audit_data_trust.py",
        "uv.lock",
        "fixtures/sources.json",
        "fixtures/candidates.jsonl",
        "fixtures/evaluation.jsonl",
        "fixtures/split.json",
        "fixtures/expectation.json",
    )
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "d06-test@example.invalid")
    git("config", "user.name", "D06 Test")
    git("add", ".")
    git("commit", "-m", "fixture")
    return paths


def test_provenance_binds_code_lock_and_all_inputs_to_git(tmp_path: Path) -> None:
    paths = initialize_provenance_repository(tmp_path)
    input_paths = paths[7:]
    provenance = collect_data_trust_provenance(
        tmp_path,
        input_paths=input_paths,
    )
    assert provenance.tracked_inputs_match_git
    assert len(provenance.git_revision) == 40
    assert provenance.tracked_input_paths == paths
    assert set(provenance.source_files_sha256) == set(paths[:6])
    assert set(provenance.tracked_files_sha256) == set(paths)
    assert provenance.dependency_lock_sha256 == hashlib.sha256(
        (tmp_path / "uv.lock").read_bytes()
    ).hexdigest()


def test_provenance_uses_recorded_revision_if_head_moves_mid_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = initialize_provenance_repository(tmp_path)

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    original_branch = git("branch", "--show-current").stdout.strip()
    git("switch", "-c", "race-b")
    raced_input = tmp_path / paths[7]
    raced_input.write_text("bytes from race-b\n", encoding="utf-8")
    git("add", paths[7])
    git("commit", "-m", "race-b")
    git("switch", original_branch)

    original_run_git = audit_module._run_git
    switched = False

    def move_head_after_revision(
        repository_root: Path, *args: str
    ) -> subprocess.CompletedProcess[str]:
        nonlocal switched
        result = original_run_git(repository_root, *args)
        if args == ("rev-parse", "HEAD") and not switched:
            git("switch", "race-b")
            switched = True
        return result

    monkeypatch.setattr(audit_module, "_run_git", move_head_after_revision)
    with pytest.raises(DataTrustProvenanceError, match="recorded Git revision"):
        collect_data_trust_provenance(tmp_path, input_paths=paths[7:])


def test_provenance_rejects_dirty_or_untracked_inputs(tmp_path: Path) -> None:
    paths = initialize_provenance_repository(tmp_path)
    dirty_path = tmp_path / paths[7]
    dirty_path.write_text("modified\n", encoding="utf-8")
    with pytest.raises(DataTrustProvenanceError, match="differs"):
        collect_data_trust_provenance(tmp_path, input_paths=paths[7:])
    assert not collect_data_trust_provenance(
        tmp_path,
        input_paths=paths[7:],
        require_clean=False,
    ).tracked_inputs_match_git

    dirty_path.write_text(f"fixture for {paths[7]}\n", encoding="utf-8")
    untracked = tmp_path / "fixtures/untracked.json"
    untracked.write_text("{}\n", encoding="utf-8")
    with pytest.raises(DataTrustProvenanceError, match="not Git tracked"):
        collect_data_trust_provenance(tmp_path, input_paths=(*paths[7:], untracked))


def test_provenance_rejects_root_mismatch_and_outside_input(tmp_path: Path) -> None:
    paths = initialize_provenance_repository(tmp_path)
    nested = tmp_path / "src"
    with pytest.raises(DataTrustProvenanceError, match="worktree root"):
        collect_data_trust_provenance(nested, input_paths=paths[7:])
    outside = tmp_path.parent / "outside-d06.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(DataTrustProvenanceError, match="inside repository"):
        collect_data_trust_provenance(tmp_path, input_paths=(*paths[7:], outside))


def initialize_formal_audit_repository(tmp_path: Path) -> None:
    relative_paths = (
        "src/posttrain_lab/__init__.py",
        "src/posttrain_lab/data/__init__.py",
        "src/posttrain_lab/data/registry.py",
        "src/posttrain_lab/data/contamination.py",
        "src/posttrain_lab/data/audit.py",
        "scripts/audit_data_trust.py",
        "uv.lock",
        "tests/fixtures/data_trust/source_registry.json",
        "tests/fixtures/data_trust/transform_registry.json",
        "tests/fixtures/data_trust/candidate_records.jsonl",
        "tests/fixtures/data_trust/evaluation_records.jsonl",
        "tests/fixtures/data_trust/split_policy.json",
        "tests/fixtures/data_trust/expectation.json",
        "tests/fixtures/data_trust/ingest_transform.py",
        "tests/fixtures/data_trust/ingest_transform_config.json",
    )
    for relative in relative_paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(relative, target)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "d06-formal@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "D06 Formal Test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "formal-fixture"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def run_formal_audit(repository: Path):
    return run_data_trust_audit(
        repository,
        source_registry_path="tests/fixtures/data_trust/source_registry.json",
        transform_registry_path="tests/fixtures/data_trust/transform_registry.json",
        candidate_records_path="tests/fixtures/data_trust/candidate_records.jsonl",
        evaluation_records_path="tests/fixtures/data_trust/evaluation_records.jsonl",
        split_policy_path="tests/fixtures/data_trust/split_policy.json",
        expectation_path="tests/fixtures/data_trust/expectation.json",
    )


def test_formal_audit_collects_its_own_complete_provenance(tmp_path: Path) -> None:
    initialize_formal_audit_repository(tmp_path)
    report = run_formal_audit(tmp_path)
    assert report.passed
    assert report.provenance.tracked_inputs_match_git
    assert "src/posttrain_lab/__init__.py" in report.provenance.source_files_sha256
    assert "src/posttrain_lab/data/__init__.py" in report.provenance.source_files_sha256
    assert (
        "tests/fixtures/data_trust/ingest_transform.py"
        in report.provenance.tracked_files_sha256
    )


def test_formal_audit_rejects_dirty_public_api_and_stale_transform_hash(
    tmp_path: Path,
) -> None:
    initialize_formal_audit_repository(tmp_path)
    public_api = tmp_path / "src/posttrain_lab/__init__.py"
    public_api.write_text(public_api.read_text(encoding="utf-8") + "# dirty\n")
    with pytest.raises(DataTrustProvenanceError, match="differs"):
        run_formal_audit(tmp_path)

    copyfile("src/posttrain_lab/__init__.py", public_api)
    transform = tmp_path / "tests/fixtures/data_trust/ingest_transform.py"
    transform.write_text(transform.read_text(encoding="utf-8") + "# new tracked bytes\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "stale-transform-registry"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    with pytest.raises(DataTrustProvenanceError, match="transform artifact hash"):
        run_formal_audit(tmp_path)
