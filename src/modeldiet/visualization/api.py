"""Stable public facade for ModelDiet's interactive visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from modeldiet.visualization import charts
from modeldiet.visualization.base import VisualizationBase
from modeldiet.visualization.charts import _go
from modeldiet.visualization.comparison import METRICS, pairwise_rows, value_for
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
        selected = self._select(models)
        aliases = {
            "p50_ms": ("P50", "latency", "p50_ms"),
            "p95_ms": ("P95", "latency", "p95_ms"),
            "p99_ms": ("P99", "latency", "p99_ms"),
        }
        if not percentiles or any(item not in aliases for item in percentiles):
            raise ValueError(
                "percentiles must contain one or more of: p50_ms, p95_ms, p99_ms"
            )
        return self._style(
            charts._bar(
                selected,
                "Inference latency by percentile",
                "Single-row prediction latency · milliseconds",
                [aliases[item] for item in percentiles],
            )
        )

    def plot_mean_latency(self, models: Sequence[str] | None = None) -> Any:
        """Return a mean single-row latency comparison in milliseconds."""
        return self._style(
            charts._bar(
                self._select(models),
                "Mean inference latency",
                "Single-row prediction latency · milliseconds",
                [("Mean", "latency", "mean_ms")],
            )
        )

    def plot_latency_distribution(
        self, models: Sequence[str] | None = None, chart_type: str = "ecdf"
    ) -> Any:
        """Plot retained raw observations as ECDF, box, or violin; never infer samples."""
        selected = self._select(models)
        if chart_type == "ecdf":
            return self._style(charts.create_latency_distribution_chart(selected))
        if chart_type not in {"box", "violin"}:
            raise ValueError("chart_type must be 'ecdf', 'box', or 'violin'")
        go = _go()
        figure = go.Figure()
        for name, record in selected.items():
            raw = (
                record.get("latency_observations_ms")
                if isinstance(record, Mapping)
                else None
            )
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                continue
            values = [
                value for item in raw if (value := charts._number(item)) is not None
            ]
            if values:
                trace_type = go.Box if chart_type == "box" else go.Violin
                figure.add_trace(
                    trace_type(
                        name=name, y=values, box_visible=True, meanline_visible=True
                    )
                    if chart_type == "violin"
                    else trace_type(name=name, y=values, boxpoints="outliers")
                )
        if not figure.data:
            figure.add_annotation(
                text="Distribution plots require retained raw latency observations.",
                showarrow=False,
            )
        figure.update_layout(
            title=f"Latency {chart_type} plot", yaxis_title="Latency (ms)"
        )
        return self._style(figure)

    def plot_memory_comparison(self, models: Sequence[str] | None = None) -> Any:
        """Return the process RSS comparison (whole-process endpoint snapshots)."""
        return self._style(charts.create_rss_comparison_chart(self._select(models)))

    def plot_python_memory(self, models: Sequence[str] | None = None) -> Any:
        """Return Python-traced peak allocation comparison, in MiB."""
        return self._style(charts.create_python_memory_chart(self._select(models)))

    def plot_loading_time(self, models: Sequence[str] | None = None) -> Any:
        """Return model loading duration comparison."""
        return self._style(charts.create_loading_time_chart(self._select(models)))

    def plot_artifact_size(self, models: Sequence[str] | None = None) -> Any:
        """Return serialized model representation size comparison."""
        return self._style(charts.create_artifact_size_chart(self._select(models)))

    def plot_accuracy_latency(self, models: Sequence[str] | None = None) -> Any:
        """Return the observed test accuracy versus P99 latency scatter plot."""
        return self._style(charts.create_accuracy_latency_chart(self._select(models)))

    def plot_normalized_comparison(
        self, metrics: Sequence[str], baseline: str, models: Sequence[str] | None = None
    ) -> Any:
        """Show metric values indexed to a baseline model (baseline = 100)."""
        selected = self._select(models)
        if baseline not in selected:
            raise ValueError("baseline must be included in selected models")
        if not metrics:
            raise ValueError("metrics must contain at least one supported metric")
        unknown = [metric for metric in metrics if metric not in METRICS]
        if unknown:
            raise ValueError(f"Unknown metric(s): {', '.join(unknown)}")
        go = _go()
        figure = go.Figure()
        for metric in metrics:
            base_value = value_for(selected[baseline], metric)
            values = []
            for record in selected.values():
                current = value_for(record, metric)
                values.append(
                    current / base_value * 100
                    if current is not None and base_value not in (None, 0)
                    else None
                )
            figure.add_trace(
                go.Bar(name=METRICS[metric][2], x=list(selected), y=values)
            )
        figure.update_layout(
            barmode="group",
            title="Baseline-relative comparison",
            yaxis_title=f"Index (baseline {baseline} = 100)",
            xaxis_title="Model",
        )
        figure.add_annotation(
            text="Relative index only · raw measurements remain available in individual charts and source results.",
            x=0,
            y=-0.22,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return self._style(figure)

    def compare_models(self, model_a: str, model_b: str) -> Any:
        """Compare two models, treating model_a as reference and model_b as comparison."""
        go = _go()
        rows = pairwise_rows(self.results, model_a, model_b)
        table = go.Table(
            header={
                "values": [
                    "Metric",
                    f"Reference: {model_a}",
                    f"Comparison: {model_b}",
                    "Difference (B − A)",
                    "% change vs A",
                ]
            },
            cells={
                "values": [
                    [row["label"] for row in rows],
                    [row["reference"] for row in rows],
                    [row["comparison"] for row in rows],
                    [row["difference"] for row in rows],
                    [row["percent_change"] for row in rows],
                ]
            },
        )
        figure = go.Figure(data=[table])
        figure.update_layout(title="Pairwise model comparison", height=560)
        return self._style(figure)

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
