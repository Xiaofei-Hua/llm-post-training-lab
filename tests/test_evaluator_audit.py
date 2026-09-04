from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from shutil import copyfile

import pytest

import posttrain_lab.evaluation.audit as audit_module
from posttrain_lab.evaluation import (
    EVALUATOR_AUDIT_REPORT_SCHEMA_VERSION,
    EvaluatorAuditError,
    EvaluatorAuditProvenanceError,
    collect_evaluator_audit_provenance,
    load_benchmark_descriptor,
    load_evaluator_audit_expectation,
    load_fixture_predictions,
    load_generation_protocol,
    load_public_benchmark,
    run_evaluator_audit,
)

FIXTURE_ROOT = Path("tests/fixtures/evaluation_contract")
FIXTURE_PATHS = (
    "tests/fixtures/evaluation_contract/benchmark_descriptor.json",
    "tests/fixtures/evaluation_contract/public_items.jsonl",
    "tests/fixtures/evaluation_contract/sealed_references.jsonl",
    "tests/fixtures/evaluation_contract/greedy_protocol.json",
    "tests/fixtures/evaluation_contract/sampling_protocol.json",
    "tests/fixtures/evaluation_contract/fixture_predictions.json",
    "tests/fixtures/evaluation_contract/expectation.json",
)


def git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def initialize_formal_repository(tmp_path: Path) -> None:
    for relative in (*audit_module._IMPLEMENTATION_SOURCE_PATHS, *FIXTURE_PATHS):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(relative, target)
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "d07-test@example.invalid")
    git(tmp_path, "config", "user.name", "D07 Test")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "formal evaluator fixture")


def run_formal(repository: Path):
    return run_evaluator_audit(
        repository,
        descriptor_path=FIXTURE_PATHS[0],
        public_items_path=FIXTURE_PATHS[1],
        sealed_references_path=FIXTURE_PATHS[2],
        greedy_protocol_path=FIXTURE_PATHS[3],
        sampling_protocol_path=FIXTURE_PATHS[4],
        fixture_predictions_path=FIXTURE_PATHS[5],
        expectation_path=FIXTURE_PATHS[6],
    )


def run_formal_subprocess(
    repository: Path,
    *,
    output_name: str = "audit.json",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (repository / "src").as_posix()
    output = repository / output_name
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_evaluator.py",
            "--output",
            output.name,
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, output


def test_formal_evaluator_audit_passes_with_complete_git_provenance(
    tmp_path: Path,
) -> None:
    initialize_formal_repository(tmp_path)
    result, output = run_formal_subprocess(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["failures"] == []
    assert report["provenance"]["tracked_inputs_match_git"] is True
    assert report["runs"]["greedy"]["answer_accuracy_ppm"] == 500_000
    assert report["runs"]["sampling"]["answer_accuracy_ppm"] == 437_500
    assert report["runs"]["sampling"]["pass_at_k_ppm"] == {
        "1": 437_500,
        "8": 833_333,
    }
    assert report["runs"]["greedy"]["truncation_count"] == 1
    assert report["runs"]["sampling"]["truncation_count"] == 1
    assert set(report["provenance"]["runtime_versions"]) == {
        "python",
        "python_implementation",
        "unicode_database",
    }


def test_expectation_mismatch_is_structured_failure(tmp_path: Path) -> None:
    initialize_formal_repository(tmp_path)
    expectation_path = tmp_path / FIXTURE_PATHS[6]
    payload = json.loads(expectation_path.read_text(encoding="utf-8"))
    payload["runs"]["greedy"]["answer_accuracy_ppm"] = 0
    expectation_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git(tmp_path, "add", FIXTURE_PATHS[6])
    git(tmp_path, "commit", "-m", "change expected accuracy")
    result, output = run_formal_subprocess(tmp_path)
    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["failures"] == [
        "greedy.answer_accuracy_ppm differs from frozen expectation"
    ]


def test_audit_report_is_deterministic_and_contains_no_raw_surfaces(
    tmp_path: Path,
) -> None:
    initialize_formal_repository(tmp_path)
    first_result, first = run_formal_subprocess(tmp_path, output_name="first.json")
    second_result, second = run_formal_subprocess(tmp_path, output_name="second.json")
    assert first_result.returncode == second_result.returncode == 0
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema_version"] == EVALUATOR_AUDIT_REPORT_SCHEMA_VERSION
    assert payload["passed"] is True
    summary = json.loads(first_result.stdout.strip().splitlines()[-1])
    assert (
        summary["evaluator_audit_report_sha256"]
        == hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    raw = first.read_bytes().lower()
    assert b"compute six multiplied by seven" not in raw
    assert b"final answer: 42" not in raw
    assert b'"reference":"42"' not in raw


def test_prediction_and_expectation_loaders_fail_closed(tmp_path: Path) -> None:
    descriptor = load_benchmark_descriptor(FIXTURE_ROOT / "benchmark_descriptor.json")
    public = load_public_benchmark(descriptor, FIXTURE_ROOT / "public_items.jsonl")
    greedy = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    sampling = load_generation_protocol(FIXTURE_ROOT / "sampling_protocol.json")
    payload = json.loads((FIXTURE_ROOT / "fixture_predictions.json").read_text(encoding="utf-8"))
    payload["items"][0]["reference"] = "42"
    invalid_predictions = tmp_path / "predictions.json"
    invalid_predictions.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluatorAuditError, match="invalid keys"):
        load_fixture_predictions(
            invalid_predictions,
            public=public,
            greedy_protocol=greedy,
            sampling_protocol=sampling,
        )

    expectation = json.loads((FIXTURE_ROOT / "expectation.json").read_text(encoding="utf-8"))
    expectation["runs"]["sampling"]["sample_count"] = True
    invalid_expectation = tmp_path / "expectation.json"
    invalid_expectation.write_text(json.dumps(expectation), encoding="utf-8")
    with pytest.raises(EvaluatorAuditError, match="non-negative integer"):
        load_evaluator_audit_expectation(invalid_expectation)


def test_provenance_rejects_dirty_untracked_root_and_outside_inputs(
    tmp_path: Path,
) -> None:
    initialize_formal_repository(tmp_path)
    dirty = tmp_path / FIXTURE_PATHS[0]
    dirty.write_text(dirty.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(EvaluatorAuditProvenanceError, match="differs"):
        collect_evaluator_audit_provenance(tmp_path, input_paths=FIXTURE_PATHS)
    assert not collect_evaluator_audit_provenance(
        tmp_path,
        input_paths=FIXTURE_PATHS,
        require_clean=False,
    ).tracked_inputs_match_git

    git(tmp_path, "checkout", "--", FIXTURE_PATHS[0])
    untracked = tmp_path / "untracked.json"
    untracked.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluatorAuditProvenanceError, match="not Git tracked"):
        collect_evaluator_audit_provenance(
            tmp_path,
            input_paths=(*FIXTURE_PATHS, untracked),
        )
    with pytest.raises(EvaluatorAuditProvenanceError, match="worktree root"):
        collect_evaluator_audit_provenance(
            tmp_path / "src",
            input_paths=FIXTURE_PATHS,
        )
    outside = tmp_path.parent / "outside-d07.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluatorAuditProvenanceError, match="inside repository"):
        collect_evaluator_audit_provenance(
            tmp_path,
            input_paths=(*FIXTURE_PATHS, outside),
        )


def test_provenance_rejects_head_change_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_formal_repository(tmp_path)
    original_branch = git(tmp_path, "branch", "--show-current").stdout.strip()
    git(tmp_path, "switch", "-c", "head-race")
    git(tmp_path, "commit", "--allow-empty", "-m", "move head only")
    git(tmp_path, "switch", original_branch)
    original_run_git = audit_module._run_git
    switched = False

    def move_after_capture(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        nonlocal switched
        result = original_run_git(root, *args)
        if args == ("rev-parse", "HEAD") and not switched:
            git(tmp_path, "switch", "head-race")
            switched = True
        return result

    monkeypatch.setattr(audit_module, "_run_git", move_after_capture)
    with pytest.raises(EvaluatorAuditProvenanceError, match="HEAD changed"):
        collect_evaluator_audit_provenance(tmp_path, input_paths=FIXTURE_PATHS)


def test_formal_audit_rejects_foreign_checkout_runtime(tmp_path: Path) -> None:
    initialize_formal_repository(tmp_path)
    with pytest.raises(EvaluatorAuditProvenanceError, match="runtime module source"):
        run_formal(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "src/posttrain_lab/data/audit.py",
        "src/posttrain_lab/data/contamination.py",
        "src/posttrain_lab/rewards/audit.py",
    ],
)
def test_formal_audit_rejects_dirty_runtime_import_closure(
    tmp_path: Path,
    relative: str,
) -> None:
    initialize_formal_repository(tmp_path)
    path = tmp_path / relative
    path.write_text(path.read_text(encoding="utf-8") + "\n# dirty\n", encoding="utf-8")
    result, _ = run_formal_subprocess(tmp_path)
    assert result.returncode != 0
    assert "differs" in result.stderr


def test_formal_audit_detects_input_change_after_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_formal_repository(tmp_path)
    monkeypatch.setattr(audit_module, "_assert_runtime_source_origin", lambda root: None)
    original_loader = audit_module.load_fixture_predictions

    def mutate_after_loading(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        path = tmp_path / FIXTURE_PATHS[5]
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return loaded

    monkeypatch.setattr(audit_module, "load_fixture_predictions", mutate_after_loading)
    with pytest.raises(EvaluatorAuditProvenanceError, match="changed during execution"):
        run_formal(tmp_path)


def test_formal_audit_binds_bytes_consumed_even_if_input_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_formal_repository(tmp_path)
    monkeypatch.setattr(audit_module, "_assert_runtime_source_origin", lambda root: None)
    original_loader = audit_module.load_fixture_predictions

    def swap_only_while_loading(*args, **kwargs):
        path = Path(args[0])
        original = path.read_bytes()
        payload = json.loads(original)
        payload["items"][0]["greedy"] = "Final answer: 0"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            return original_loader(*args, **kwargs)
        finally:
            path.write_bytes(original)

    monkeypatch.setattr(audit_module, "load_fixture_predictions", swap_only_while_loading)
    with pytest.raises(EvaluatorAuditProvenanceError, match="consumed bytes"):
        run_formal(tmp_path)


def test_formal_audit_detects_head_change_during_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_formal_repository(tmp_path)
    monkeypatch.setattr(audit_module, "_assert_runtime_source_origin", lambda root: None)
    original_branch = git(tmp_path, "branch", "--show-current").stdout.strip()
    git(tmp_path, "switch", "-c", "audit-head-race")
    git(tmp_path, "commit", "--allow-empty", "-m", "move head during audit")
    git(tmp_path, "switch", original_branch)
    original_loader = audit_module.load_fixture_predictions

    def move_head_after_loading(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        git(tmp_path, "switch", "audit-head-race")
        return loaded

    monkeypatch.setattr(audit_module, "load_fixture_predictions", move_head_after_loading)
    with pytest.raises(EvaluatorAuditProvenanceError, match="HEAD changed"):
        run_formal(tmp_path)


def test_evaluator_audit_cli_help_is_available() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit_evaluator.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--sealed-references" in result.stdout
    assert "--expectation" in result.stdout
