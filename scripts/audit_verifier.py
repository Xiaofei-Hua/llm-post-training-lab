#!/usr/bin/env python3
"""Thin CLI for the deterministic D05 verifier audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_lab.rewards import (
    collect_audit_provenance,
    load_audit_corpus,
    run_verifier_audit,
    write_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("tests/fixtures/verifier_adversarial.jsonl"),
        help="strict JSONL adversarial corpus",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/D05_VERIFIER_AUDIT.json"),
        help="machine-readable audit report",
    )
    parser.add_argument("--minimum-cases", type=int, default=100)
    parser.add_argument("--maximum-cases", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    corpus = load_audit_corpus(args.cases)
    report = run_verifier_audit(
        corpus,
        provenance=collect_audit_provenance(
            Path.cwd(),
            require_clean=True,
            additional_input_paths=(args.cases,),
        ),
        minimum_cases=args.minimum_cases,
        maximum_cases=args.maximum_cases,
    )
    write_audit_report(report, args.output)
    print(
        json.dumps(
            {
                "passed": report.passed,
                "total_cases": report.total_cases,
                "passed_cases": report.passed_cases,
                "failed_cases": report.failed_cases,
                "corpus_sha256": report.corpus_sha256,
                "policy_sha256": report.policy_sha256,
                "implementation_source_sha256": (
                    report.provenance.implementation_source_sha256
                ),
                "dependency_lock_sha256": report.provenance.dependency_lock_sha256,
                "git_revision": report.provenance.git_revision,
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
