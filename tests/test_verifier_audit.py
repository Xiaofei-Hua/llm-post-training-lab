from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from posttrain_lab.rewards import (
    AuditCorpusError,
    AuditProvenanceError,
    ExactMathVerifier,
    LoadedAuditCorpus,
    VerificationStatus,
    VerifierAuditCase,
    collect_audit_provenance,
    load_audit_corpus,
    run_verifier_audit,
    write_audit_report,
)

FIXTURE_PATH = Path("tests/fixtures/verifier_adversarial.jsonl")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_frozen_adversarial_corpus_has_required_size_and_all_cases_pass() -> None:
    corpus = load_audit_corpus(FIXTURE_PATH)
    assert 100 <= len(corpus.cases) <= 300
    assert len({case.case_id for case in corpus.cases}) == len(corpus.cases)
    assert len({case.category for case in corpus.cases}) >= 10
    report = run_verifier_audit(corpus)
    assert report.passed
    assert report.passed_cases == report.total_cases
    assert report.failed_cases == 0
    assert report.failures == ()
    assert all(counts["failed"] == 0 for counts in report.category_counts.values())
    assert report.category_counts["invalid_reference"]["passed"] == 6


def test_corpus_digest_is_raw_byte_sensitive_and_deterministic(tmp_path: Path) -> None:
    row = {
        "case_id": "one",
        "category": "unit",
        "reference": "1",
        "prediction": "1",
        "expected_reward": 1,
        "expected_status": "match",
    }
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, [row])
    first = load_audit_corpus(path)
    second = load_audit_corpus(path)
    assert first.sha256 == second.sha256
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert load_audit_corpus(path).sha256 != first.sha256


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "at least one"),
        ([{"case_id": "x"}], "missing fields"),
        (
            [
                {
                    "case_id": "x",
                    "category": "c",
                    "reference": "1",
                    "prediction": "1",
                    "expected_reward": 1,
                    "surprise": True,
                }
            ],
            "unknown fields",
        ),
        (
            [
                {
                    "case_id": "x",
                    "category": "c",
                    "reference": "1",
                    "prediction": "1",
                    "expected_reward": 1,
                    "expected_status": "unknown",
                }
            ],
            "invalid expected_status",
        ),
        (
            [
                {
                    "case_id": "x",
                    "category": "c",
                    "reference": "1",
                    "prediction": "1",
                    "expected_reward": True,
                }
            ],
            "expected_reward",
        ),
    ],
)
def test_loader_rejects_schema_violations(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(AuditCorpusError, match=message):
        load_audit_corpus(path)


def test_loader_rejects_duplicate_ids_and_invalid_json(tmp_path: Path) -> None:
    row = {
        "case_id": "duplicate",
        "category": "c",
        "reference": "1",
        "prediction": "1",
        "expected_reward": 1,
    }
    duplicate_path = tmp_path / "duplicate.jsonl"
    _write_jsonl(duplicate_path, [row, row])
    with pytest.raises(AuditCorpusError, match="duplicate case_id"):
        load_audit_corpus(duplicate_path)

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(AuditCorpusError, match="invalid JSON"):
        load_audit_corpus(invalid_path)


def test_case_count_gate_is_strict() -> None:
    case = VerifierAuditCase(
        case_id="one",
        category="unit",
        reference="1",
        prediction="1",
        expected_reward=1.0,
    )
    corpus = LoadedAuditCorpus(
        schema_version="d05-verifier-audit-case-v1",
        path="memory",
        sha256="0" * 64,
        cases=(case,),
    )
    with pytest.raises(AuditCorpusError, match="100-300"):
        run_verifier_audit(corpus)
    with pytest.raises(ValueError, match="bounds"):
        run_verifier_audit(corpus, minimum_cases=2, maximum_cases=1)


def test_failed_expectation_is_retained_with_structured_decision() -> None:
    case = VerifierAuditCase(
        case_id="wrong-expectation",
        category="unit",
        reference="1",
        prediction="2",
        expected_reward=1.0,
        expected_status=VerificationStatus.MATCH,
    )
    corpus = LoadedAuditCorpus(
        schema_version="d05-verifier-audit-case-v1",
        path="memory",
        sha256="0" * 64,
        cases=(case,),
    )
    report = run_verifier_audit(
        corpus,
        verifier=ExactMathVerifier(),
        minimum_cases=1,
        maximum_cases=1,
    )
    assert not report.passed
    assert report.failed_cases == 1
    failure = report.failures[0]
    assert failure.case_id == "wrong-expectation"
    assert failure.actual_reward == 0.0
    assert failure.actual_status == "mismatch"
    assert failure.result.to_record()["prediction"]["status"] == "parsed"


def test_report_writer_is_deterministic_and_parseable(tmp_path: Path) -> None:
    corpus = load_audit_corpus(FIXTURE_PATH)
    report = run_verifier_audit(corpus)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_audit_report(report, first)
    write_audit_report(report, second)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "d05-verifier-audit-report-v1"
    assert payload["total_cases"] == len(corpus.cases)
    assert payload["passed"] is True
    assert payload["backend_versions"]["math-verify"] == "0.9.0"
    assert len(payload["provenance"]["implementation_source_sha256"]) == 64
    assert len(payload["provenance"]["dependency_lock_sha256"]) == 64
    assert set(payload["provenance"]["source_files_sha256"]) == {
        "scripts/audit_verifier.py",
        "src/posttrain_lab/rewards/audit.py",
        "src/posttrain_lab/rewards/verifier.py",
    }


def test_provenance_requires_tracked_sources_that_match_git(tmp_path: Path) -> None:
    source_paths = (
        "src/posttrain_lab/rewards/verifier.py",
        "src/posttrain_lab/rewards/audit.py",
        "scripts/audit_verifier.py",
    )
    for relative in (*source_paths, "uv.lock"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "d05-test@example.invalid")
    git("config", "user.name", "D05 Test")
    git("add", ".")
    git("commit", "-m", "fixture")

    provenance = collect_audit_provenance(tmp_path)
    assert provenance.tracked_inputs_match_git
    assert len(provenance.git_revision) == 40
    expected_verifier_digest = hashlib.sha256(
        (tmp_path / source_paths[0]).read_bytes()
    ).hexdigest()
    assert provenance.source_files_sha256[source_paths[0]] == expected_verifier_digest

    (tmp_path / source_paths[0]).write_text("modified\n", encoding="utf-8")
    with pytest.raises(AuditProvenanceError, match="must be tracked and match HEAD"):
        collect_audit_provenance(tmp_path)
    assert not collect_audit_provenance(
        tmp_path,
        require_clean=False,
    ).tracked_inputs_match_git


def test_provenance_binds_additional_corpus_input_to_git(tmp_path: Path) -> None:
    source_paths = (
        "src/posttrain_lab/rewards/verifier.py",
        "src/posttrain_lab/rewards/audit.py",
        "scripts/audit_verifier.py",
    )
    corpus_path = "tests/fixtures/verifier_adversarial.jsonl"
    for relative in (*source_paths, "uv.lock", corpus_path):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "d05-test@example.invalid")
    git("config", "user.name", "D05 Test")
    git("add", ".")
    git("commit", "-m", "fixture")

    provenance = collect_audit_provenance(
        tmp_path,
        additional_input_paths=(tmp_path / corpus_path,),
    )
    assert provenance.tracked_inputs_match_git
    assert corpus_path in provenance.tracked_input_paths

    (tmp_path / corpus_path).write_text("modified corpus\n", encoding="utf-8")
    with pytest.raises(AuditProvenanceError, match="must be tracked and match HEAD"):
        collect_audit_provenance(
            tmp_path,
            additional_input_paths=(corpus_path,),
        )


def test_provenance_rejects_additional_input_outside_repository(tmp_path: Path) -> None:
    outside_path = tmp_path.parent / "outside-audit-input.jsonl"
    with pytest.raises(AuditProvenanceError, match="inside repository root"):
        collect_audit_provenance(
            tmp_path,
            require_clean=False,
            additional_input_paths=(outside_path,),
        )
