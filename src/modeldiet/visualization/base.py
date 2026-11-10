"""Shared setup for the public visualization family.

New chart families should inherit :class:`VisualizationBase` and keep their
metric-specific methods in a small subclass. Input validation, model
selection, and theme styling belong here so every chart behaves consistently.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from modeldiet.result import ProfileResult
from modeldiet.visualization.charts import Results


class VisualizationBase:
    """Common result loading, validation, selection, and figure styling."""

    def __init__(self, results: Results, theme: str = "light") -> None:
        if not isinstance(results, Mapping):
            raise TypeError(
                "results must be a mapping of model names to profile results"
            )
        if theme not in {"light", "dark"}:
            raise ValueError("theme must be 'light' or 'dark'")
        for name, record in results.items():
            if not isinstance(name, str) or not name:
                raise ValueError("model names must be non-empty strings")
            if not isinstance(record, (ProfileResult, Mapping)):
                raise TypeError(
                    f"result for '{name}' must be a ProfileResult or mapping"
                )
        self.results: Results = dict(results)
        self.theme = theme

    @classmethod
    def from_file(cls, path: str | Path, theme: str = "light") -> VisualizationBase:
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
        results = {str(payload["model"]): payload} if "model" in payload else payload
        return cls(results, theme=theme)

    def _select(self, models: Sequence[str] | None) -> Results:
        if models is None:
            return self.results
        unknown = [name for name in models if name not in self.results]
        if unknown:
            raise ValueError(f"Unknown model name(s): {', '.join(unknown)}")
        if not models:
            raise ValueError("models must select at least one model")
        return {name: self.results[name] for name in models}

    def _style(self, figure: Any) -> Any:
        if self.theme == "dark":
            figure.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#e5e7eb"
            )
            figure.update_xaxes(
                gridcolor="#374151",
                linecolor="#4b5563",
                title_font_color="#e5e7eb",
                tickfont_color="#cbd5e1",
            )
            figure.update_yaxes(
                gridcolor="#374151",
                linecolor="#4b5563",
                title_font_color="#e5e7eb",
                tickfont_color="#cbd5e1",
            )
        return figure
