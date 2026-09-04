#!/usr/bin/env python3
"""Run the deterministic, Git-bound D07 sealed-evaluator audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_lab.evaluation import (
    evaluator_audit_report_sha256,
    run_evaluator_audit,
    write_evaluator_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    fixture_root = Path("tests/fixtures/evaluation_contract")
    parser.add_argument(
        "--descriptor",
        type=Path,
        default=fixture_root / "benchmark_descriptor.json",
    )
    parser.add_argument(
        "--public-items",
        type=Path,
        default=fixture_root / "public_items.jsonl",
    )
    parser.add_argument(
        "--sealed-references",
        type=Path,
        default=fixture_root / "sealed_references.jsonl",
    )
    parser.add_argument(
        "--greedy-protocol",
        type=Path,
        default=fixture_root / "greedy_protocol.json",
    )
    parser.add_argument(
        "--sampling-protocol",
        type=Path,
        default=fixture_root / "sampling_protocol.json",
    )
    parser.add_argument(
        "--fixture-predictions",
        type=Path,
        default=fixture_root / "fixture_predictions.json",
    )
    parser.add_argument(
        "--expectation",
        type=Path,
        default=fixture_root / "expectation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/D07_EVALUATOR_AUDIT.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_evaluator_audit(
        Path.cwd(),
        descriptor_path=args.descriptor,
        public_items_path=args.public_items,
        sealed_references_path=args.sealed_references,
        greedy_protocol_path=args.greedy_protocol,
        sampling_protocol_path=args.sampling_protocol,
        fixture_predictions_path=args.fixture_predictions,
        expectation_path=args.expectation,
    )
    write_evaluator_audit(report, args.output)
    print(
        json.dumps(
            {
                "passed": report.passed,
                "greedy_accuracy_ppm": report.runs["greedy"].answer_accuracy_ppm,
                "sampling_accuracy_ppm": report.runs["sampling"].answer_accuracy_ppm,
                "sampling_pass_at_8_ppm": report.runs["sampling"].pass_at_k_ppm["8"],
                "evaluator_audit_report_sha256": evaluator_audit_report_sha256(report),
                "implementation_source_sha256": (report.provenance.implementation_source_sha256),
                "git_revision": report.provenance.git_revision,
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
