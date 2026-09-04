#!/usr/bin/env python3
"""Run the deterministic D06 data-registry and contamination audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_lab.data import (
    audit_report_sha256,
    run_data_trust_audit,
    write_data_trust_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    fixture_root = Path("tests/fixtures/data_trust")
    parser.add_argument(
        "--sources",
        type=Path,
        default=fixture_root / "source_registry.json",
    )
    parser.add_argument(
        "--transforms",
        type=Path,
        default=fixture_root / "transform_registry.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=fixture_root / "candidate_records.jsonl",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=fixture_root / "evaluation_records.jsonl",
    )
    parser.add_argument(
        "--split-policy",
        type=Path,
        default=fixture_root / "split_policy.json",
    )
    parser.add_argument(
        "--expectation",
        type=Path,
        default=fixture_root / "expectation.json",
    )
    parser.add_argument(
        "--parent-ledger",
        type=Path,
        default=None,
        help="optional frozen external-parent payload ledger",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/D06_DATA_TRUST_AUDIT.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_data_trust_audit(
        Path.cwd(),
        source_registry_path=args.sources,
        transform_registry_path=args.transforms,
        candidate_records_path=args.candidates,
        evaluation_records_path=args.evaluation,
        split_policy_path=args.split_policy,
        expectation_path=args.expectation,
        parent_ledger_path=args.parent_ledger,
    )
    write_data_trust_audit(report, args.output)
    print(
        json.dumps(
            {
                "passed": report.passed,
                "dirty_match_counts": report.dirty_match_counts,
                "quarantined_records": len(report.quarantined_record_ids),
                "clean_training_records": report.clean_training_record_count,
                "evaluation_records": report.evaluation_record_count,
                "manifest_sha256": report.manifest_sha256,
                "audit_report_sha256": audit_report_sha256(report),
                "git_revision": report.provenance.git_revision,
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
