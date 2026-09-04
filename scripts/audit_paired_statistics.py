#!/usr/bin/env python3
"""Run the deterministic, Git-bound D08 paired-statistics audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_lab.statistics import (
    run_statistics_audit,
    statistics_audit_report_sha256,
    write_statistics_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    fixture_root = Path("tests/fixtures/paired_statistics")
    parser.add_argument("--panel", type=Path, default=fixture_root / "panel.json")
    parser.add_argument("--protocol", type=Path, default=fixture_root / "protocol.json")
    parser.add_argument("--expectation", type=Path, default=fixture_root / "expectation.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/D08_PAIRED_STATISTICS_AUDIT.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_statistics_audit(
        Path.cwd(),
        cli_source=Path(__file__),
        panel_path=args.panel,
        protocol_path=args.protocol,
        expectation_path=args.expectation,
    )
    write_statistics_audit(report, args.output)
    c1a, c1b = report.analysis.c1_results
    print(
        json.dumps(
            {
                "passed": report.passed,
                "c1a_point_ppm": c1a.point_estimate.ppm,
                "c1a_holm_adjusted_p_ppm": c1a.holm_adjusted_p_value.ppm,
                "c1b_point_ppm": c1b.point_estimate.ppm,
                "c2_classification": report.analysis.c2_result.classification.value,
                "analysis_report_sha256": report.analysis.analysis_report_sha256,
                "statistics_audit_report_sha256": statistics_audit_report_sha256(report),
                "implementation_source_sha256": report.provenance.implementation_source_sha256,
                "git_revision": report.provenance.git_revision,
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
