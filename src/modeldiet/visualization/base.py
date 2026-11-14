"""Shared setup for the public visualization family.

New chart families should inherit :class:`VisualizationBase` and keep their
metric-specific methods in a small subclass. Input validation, model
selection, and theme styling belong here so every chart behaves consistently.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from modeldiet.result import ProfileResult, RepeatedProfileResult
from modeldiet.visualization.data import Results, available_metrics, validate_results


class VisualizationBase:
    """Common result loading, validation, selection, and figure styling."""

    def __init__(self, results: Results, theme: str = "light") -> None:
        if theme not in {"light", "dark"}:
            raise ValueError("theme must be 'light' or 'dark'")
        self.results: Results = validate_results(results)
        self.theme = theme

    @classmethod
    def from_file(cls, path: str | Path, theme: str = "light") -> Self:
        """Load a combined results file or a single benchmark-worker record."""
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not load ModelDiet results from {source}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("results JSON must contain an object")
        results: Results
        if payload.get("result_type") == "modeldiet.repeated_profile":
            repeated = RepeatedProfileResult.from_dict(payload)
            name = (
                repeated.experiment_name or repeated.model_class or "Repeated profile"
            )
            results = {name: repeated}
        elif "schema_version" in payload and "measurements" in payload:
            single = ProfileResult.from_dict(payload)
            results = {single.model_class or "Profile": single}
        else:
            results = (
                {str(payload["model"]): payload} if "model" in payload else payload
            )
        return cls(results, theme=theme)

    def _select(self, models: Sequence[str] | None) -> Results:
        if models is None:
            return self.results
        unknown = [name for name in models if name not in self.results]
        if unknown:
            raise ValueError(f"Unknown model name(s): {', '.join(unknown)}")
        if not models:
            raise ValueError("models must select at least one model")
        if len(set(models)) != len(models):
            raise ValueError("models must not contain duplicate names")
        return {name: self.results[name] for name in models}

    @property
    def available_metrics(self) -> dict[str, list[str]]:
        """Map each available metric identifier to the models containing it."""
        return available_metrics(self.results)
