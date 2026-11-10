"""Command line entry point for reporting already collected JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from modeldiet.visualization.html_report import write_html_report


def main() -> None:
    """Generate one HTML report from one or more ModelDiet JSON files."""
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        sys.argv.pop(1)
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML report from existing ModelDiet JSON results."
    )
    parser.add_argument(
        "results", nargs="+", type=Path, help="Existing benchmark result JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modeldiet-report.html"),
        help="HTML output path",
    )
    parser.add_argument(
        "--experiment", help="Dataset or experiment name shown in the report"
    )
    parser.add_argument(
        "--include-json",
        action="store_true",
        help="Write a results.json next to the HTML report",
    )
    args = parser.parse_args()
    records: dict[str, dict[str, Any]] = {}
    for result_path in args.results:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read result file {result_path}: {exc}")
        if not isinstance(payload, dict):
            parser.error(f"result file must contain a JSON object: {result_path}")
        name = str(payload.get("model") or result_path.stem)
        if name in records:
            parser.error(f"duplicate model name '{name}' in supplied result files")
        records[name] = payload
    try:
        path = write_html_report(
            records,
            args.output,
            experiment_name=args.experiment,
            include_json=args.include_json,
        )
    except (ImportError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(path)


if __name__ == "__main__":
    main()
