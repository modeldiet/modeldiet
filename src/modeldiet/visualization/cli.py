"""Command line entry point for reporting already collected JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from modeldiet.visualization.api import ModelDietVisualizer


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
        if "model" in payload:
            named_records = {str(payload["model"]): payload}
        elif payload and all(isinstance(value, dict) for value in payload.values()):
            named_records = payload
        else:
            named_records = {result_path.stem: payload}
        duplicates = set(records).intersection(named_records)
        if duplicates:
            parser.error(
                f"duplicate model name(s) in supplied result files: {', '.join(sorted(duplicates))}"
            )
        records.update(named_records)
    try:
        path = ModelDietVisualizer(records).generate_report(
            args.output, experiment_name=args.experiment, include_json=args.include_json
        )
    except (ImportError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(path)


if __name__ == "__main__":
    main()
