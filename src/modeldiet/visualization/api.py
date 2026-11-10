"""Stable public facade for ModelDiet's interactive visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from modeldiet.visualization import charts
from modeldiet.visualization.base import VisualizationBase
from modeldiet.visualization.html_report import write_html_report


class ModelDietVisualizer(VisualizationBase):
    """Explore and compare already collected ModelDiet profile results.

    The class does not run models. Plot methods return Plotly ``Figure``
    objects, which can be shown in notebooks or exported by Plotly.
    """

    def plot_latency_comparison(
        self,
        models: Sequence[str] | None = None,
        percentiles: Sequence[str] = ("p50_ms", "p95_ms", "p99_ms"),
    ) -> Any:
        """Return a percentile comparison figure, optionally for selected models."""
        if not percentiles or any(
            item not in {"p50_ms", "p95_ms", "p99_ms"} for item in percentiles
        ):
            raise ValueError(
                "percentiles must contain one or more of: p50_ms, p95_ms, p99_ms"
            )
        return charts.create_latency_comparison_chart(
            self._select(models), percentiles, theme=self.theme
        )

    def plot_mean_latency(self, models: Sequence[str] | None = None) -> Any:
        """Return a mean single-row latency comparison in milliseconds."""
        return charts.create_latency_comparison_chart(
            self._select(models), ("mean_ms",), theme=self.theme
        )

    def plot_latency_distribution(
        self, models: Sequence[str] | None = None, chart_type: str = "ecdf"
    ) -> Any:
        """Plot retained raw observations as ECDF, box, or violin; never infer samples."""
        selected = self._select(models)
        if chart_type == "ecdf":
            return charts.create_latency_distribution_chart(selected, theme=self.theme)
        return charts.create_latency_shape_chart(selected, chart_type, theme=self.theme)

    def plot_memory_comparison(self, models: Sequence[str] | None = None) -> Any:
        """Return the process RSS comparison (whole-process endpoint snapshots)."""
        return charts.create_rss_comparison_chart(
            self._select(models), theme=self.theme
        )

    def plot_python_memory(self, models: Sequence[str] | None = None) -> Any:
        """Return Python-traced peak allocation comparison, in MiB."""
        return charts.create_python_memory_chart(self._select(models), theme=self.theme)

    def plot_loading_time(self, models: Sequence[str] | None = None) -> Any:
        """Return model loading duration comparison."""
        return charts.create_loading_time_chart(self._select(models), theme=self.theme)

    def plot_artifact_size(self, models: Sequence[str] | None = None) -> Any:
        """Return serialized model representation size comparison."""
        return charts.create_artifact_size_chart(self._select(models), theme=self.theme)

    def plot_accuracy_latency(self, models: Sequence[str] | None = None) -> Any:
        """Return the observed test accuracy versus P99 latency scatter plot."""
        return charts.create_accuracy_latency_chart(
            self._select(models), theme=self.theme
        )

    def plot_normalized_comparison(
        self, metrics: Sequence[str], baseline: str, models: Sequence[str] | None = None
    ) -> Any:
        """Show metric values indexed to a baseline model (baseline = 100)."""
        return charts.create_normalized_comparison_chart(
            self._select(models), metrics, baseline, theme=self.theme
        )

    def compare_models(self, model_a: str, model_b: str) -> Any:
        """Compare two models, treating model_a as reference and model_b as comparison."""
        return charts.create_pairwise_comparison_chart(
            self.results, model_a, model_b, theme=self.theme
        )

    def generate_report(
        self,
        output_path: str | Path,
        *,
        experiment_name: str | None = None,
        include_json: bool = False,
    ) -> Path:
        """Export a standalone report using the existing report composer."""
        return write_html_report(
            self.results,
            output_path,
            experiment_name=experiment_name,
            include_json=include_json,
            theme=self.theme,
        )

    def show(self, *, experiment_name: str | None = None) -> None:
        """Display results in a native local window without opening a browser."""
        from modeldiet.visualization.desktop import show_results

        show_results(self.results, title=experiment_name or "Model comparison")
