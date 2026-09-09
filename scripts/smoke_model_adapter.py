"""Run D09's CPU-only synthetic model integration example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_lab.models.smoke import model_adapter_smoke_config, run_model_adapter_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print config without running models"
    )
    parser.add_argument("--output", type=Path, help="Save the CPU evidence JSON")
    args = parser.parse_args()
    result = model_adapter_smoke_config() if args.dry_run else run_model_adapter_smoke()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
