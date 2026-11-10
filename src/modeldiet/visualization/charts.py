"""Convert validated result records into interactive Plotly figures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from modeldiet.visualization.data import (
    Results,
    accuracy_value,
    latency_observations,
    metric_number,
)
from modeldiet.visualization.theme import ACCENTS, apply_theme


def _go() -> Any:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        if exc.name != "plotly":
            raise
        raise ImportError(
            "Interactive reports require Plotly. Install it with "
            '`python -m pip install "modeldiet[viz]"`. '
        ) from exc
    return go


def _models(results: Results) -> list[str]:
    return [str(name) for name in results]


def _bar(
    results: Results,
    title: str,
    subtitle: str,
    specs: Sequence[tuple[str, str, str]],
    *,
    horizontal: bool = False,
    theme: str = "light",
) -> Any:
    go = _go()
    names = _models(results)
    fig = go.Figure()
    missing = []
    for index, (label, measurement, key) in enumerate(specs):
        values = [metric_number(results[name], measurement, key) for name in results]
        if not any(value is not None for value in values):
            missing.append(label)
        fig.add_trace(
            go.Bar(
                name=label,
                x=values if horizontal else names,
                y=names if horizontal else values,
                orientation="h" if horizontal else "v",
                marker_color=ACCENTS[index % len(ACCENTS)],
                customdata=[value for value in values],
                hovertemplate=f"Model: %{{y}}<br>{label}: %{{x:.6g}}<extra></extra>"
                if horizontal
                else f"Model: %{{x}}<br>{label}: %{{y:.6g}}<extra></extra>",
            )
        )
    if not names:
        fig.add_annotation(
            text="No model results supplied.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    elif missing:
        fig.add_annotation(
            text=f"Unavailable measurement(s): {', '.join(missing)}",
            showarrow=False,
            x=1,
            y=1.12,
            xref="paper",
            yref="paper",
            xanchor="right",
            font={"size": 11},
        )
    fig.update_layout(barmode="group")
    apply_theme(fig, title, subtitle, theme)
    return fig


def create_latency_percentile_chart(results: Results, *, theme: str = "light") -> Any:
    """Compare measured P50, P95, and P99 latency in milliseconds."""
    fig = _bar(
        results,
        "Inference latency by percentile",
        "Single-row prediction latency · milliseconds",
        [
            ("P50", "latency", "p50_ms"),
            ("P95", "latency", "p95_ms"),
            ("P99", "latency", "p99_ms"),
        ],
        theme=theme,
    )
    fig.update_yaxes(title="Latency (ms)")
    fig.update_xaxes(title="Model")
    return fig


def create_latency_comparison_chart(
    results: Results, percentiles: Sequence[str], *, theme: str = "light"
) -> Any:
    """Compare selected latency percentiles, or mean latency, by model."""
    available = {
        "mean_ms": ("Mean", "mean_ms"),
        "p50_ms": ("P50", "p50_ms"),
        "p95_ms": ("P95", "p95_ms"),
        "p99_ms": ("P99", "p99_ms"),
    }
    if not percentiles or any(item not in available for item in percentiles):
        raise ValueError(
            "percentiles must contain one or more of: mean_ms, p50_ms, p95_ms, p99_ms"
        )
    specs = [
        (available[item][0], "latency", available[item][1]) for item in percentiles
    ]
    title = (
        "Inference latency by percentile"
        if any(item.startswith("p") for item in percentiles)
        else "Mean inference latency"
    )
    fig = _bar(
        results,
        title,
        "Single-row prediction latency · milliseconds",
        specs,
        theme=theme,
    )
    fig.update_yaxes(title="Latency (ms)")
    fig.update_xaxes(title="Model")
    return fig


def create_latency_distribution_chart(results: Results, *, theme: str = "light") -> Any:
    """Show an ECDF only when original per-observation latency samples exist."""
    go = _go()
    fig = go.Figure()
    found = False
    for name, record in results.items():
        values = sorted(latency_observations(record))
        if not values:
            continue
        found = True
        fig.add_trace(
            go.Scatter(
                x=values,
                y=[(i + 1) / len(values) for i in range(len(values))],
                mode="lines",
                name=name,
                line={"color": ACCENTS[len(fig.data) % len(ACCENTS)]},
                hovertemplate="Latency: %{x:.6g} ms<br>Cumulative fraction: %{y:.1%}<extra>%{fullData.name}</extra>",
            )
        )
    if not found:
        fig.add_annotation(
            text="Latency distributions require retained raw observations. This result contains only summary percentiles.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            align="center",
        )
    apply_theme(
        fig,
        "Latency distribution",
        "ECDF of recorded per-observation latency · milliseconds",
        theme,
    )
    fig.update_xaxes(title="Latency (ms)")
    fig.update_yaxes(title="Cumulative fraction", tickformat=".0%", range=[0, 1])
    return fig


def create_latency_shape_chart(
    results: Results, chart_type: str, *, theme: str = "light"
) -> Any:
    """Create a box or violin latency chart from retained raw observations."""
    if chart_type not in {"box", "violin"}:
        raise ValueError("chart_type must be 'box', 'violin', or 'ecdf'")
    go = _go()
    figure = go.Figure()
    for name, record in results.items():
        values = latency_observations(record)
        if not values:
            continue
        if chart_type == "violin":
            figure.add_trace(
                go.Violin(name=name, y=values, box_visible=True, meanline_visible=True)
            )
        else:
            figure.add_trace(go.Box(name=name, y=values, boxpoints="outliers"))
    if not figure.data:
        figure.add_annotation(
            text="Distribution plots require retained raw latency observations.",
            showarrow=False,
        )
    apply_theme(
        figure,
        f"Latency {chart_type} plot",
        "Retained per-observation latency · milliseconds",
        theme,
    )
    figure.update_yaxes(title="Latency (ms)")
    return figure


def create_normalized_comparison_chart(
    results: Results, metrics: Sequence[str], baseline: str, *, theme: str = "light"
) -> Any:
    """Create a metric index chart where the selected baseline equals 100."""
    from modeldiet.visualization.comparison import METRICS, value_for

    if baseline not in results:
        raise ValueError("baseline must be included in selected models")
    if not metrics:
        raise ValueError("metrics must contain at least one supported metric")
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown)}")
    go = _go()
    figure = go.Figure()
    unavailable = []
    for metric in metrics:
        base_value = value_for(results[baseline], metric)
        if base_value in (None, 0):
            unavailable.append(metric)
        values = []
        for record in results.values():
            current = value_for(record, metric)
            values.append(
                current / base_value * 100
                if current is not None and base_value not in (None, 0)
                else None
            )
        figure.add_trace(go.Bar(name=METRICS[metric][2], x=list(results), y=values))
    figure.update_layout(barmode="group")
    figure.add_annotation(
        text=f"Relative index only · baseline {baseline} = 100",
        x=0,
        y=-0.22,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    if unavailable:
        figure.add_annotation(
            text=f"Cannot index metric(s): baseline value is missing or zero ({', '.join(unavailable)})",
            x=1,
            y=1.12,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
        )
    apply_theme(figure, "Baseline-relative comparison", theme=theme)
    figure.update_yaxes(title=f"Index (baseline {baseline} = 100)")
    figure.update_xaxes(title="Model")
    return figure


def create_pairwise_comparison_chart(
    results: Results, model_a: str, model_b: str, *, theme: str = "light"
) -> Any:
    """Create a table comparing two models across the supported metrics."""
    from modeldiet.visualization.comparison import pairwise_rows

    go = _go()
    rows = pairwise_rows(results, model_a, model_b)
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
    apply_theme(figure, "Pairwise model comparison", theme=theme)
    figure.update_layout(height=560)
    return figure


def create_rss_comparison_chart(results: Results, *, theme: str = "light") -> Any:
    """Compare process-wide RSS endpoint values and the separately labelled delta."""
    fig = _bar(
        results,
        "Process memory (RSS)",
        "Whole-process resident set size · MiB; endpoints are snapshots, delta is change",
        [
            ("Before inference", "process_rss", "rss_before_inference_mb"),
            ("After inference", "process_rss", "rss_after_inference_mb"),
            ("RSS change", "process_rss", "rss_delta_mb"),
        ],
        theme=theme,
    )
    fig.update_yaxes(title="RSS (MiB)")
    fig.update_xaxes(title="Model")
    return fig


def create_python_memory_chart(results: Results, *, theme: str = "light") -> Any:
    """Compare Python-traced peak inference allocations in MiB."""
    fig = _bar(
        results,
        "Python-traced memory",
        "Peak allocations observed by tracemalloc · MiB; native allocations are excluded",
        [
            (
                "Peak traced allocation",
                "python_traced_allocation",
                "traced_peak_inference_allocation_mb",
            ),
        ],
        theme=theme,
    )
    fig.update_yaxes(title="Python-traced allocation (MiB)")
    fig.update_xaxes(title="Model")
    return fig


def create_loading_time_chart(results: Results, *, theme: str = "light") -> Any:
    """Compare measured model deserialization time in seconds."""
    fig = _bar(
        results,
        "Model loading time",
        "Measured deserialization duration · seconds",
        [("Loading time", "", "loading_time")],
        horizontal=True,
        theme=theme,
    )
    fig.update_xaxes(title="Loading time (seconds)")
    fig.update_yaxes(title="Model", autorange="reversed")
    return fig


def create_artifact_size_chart(results: Results, *, theme: str = "light") -> Any:
    """Compare serialized model artifact representations in MiB."""
    fig = _bar(
        results,
        "Serialized model size",
        "Python pickle representation · MiB; not in-memory size",
        [("Serialized size", "pickle_size", "serialized_model_size_mb")],
        horizontal=True,
        theme=theme,
    )
    fig.update_xaxes(title="Serialized size (MiB)")
    fig.update_yaxes(title="Model", autorange="reversed")
    return fig


def create_accuracy_latency_chart(results: Results, *, theme: str = "light") -> Any:
    """Plot observed test accuracy against P99 latency when both are available."""
    go = _go()
    fig = go.Figure()
    for index, (name, record) in enumerate(results.items()):
        accuracy = accuracy_value(record)
        latency = metric_number(record, "latency", "p99_ms")
        if accuracy is None or latency is None:
            continue
        rss = metric_number(record, "process_rss", "rss_after_inference_mb")
        size = metric_number(record, "pickle_size", "serialized_model_size_mb")
        fig.add_trace(
            go.Scatter(
                x=[accuracy],
                y=[latency],
                mode="markers+text",
                text=[name],
                textposition="top center",
                name=name,
                marker={"size": 12, "color": ACCENTS[index % len(ACCENTS)]},
                customdata=[[rss, size]],
                hovertemplate="Model: %{text}<br>Test accuracy: %{x:.4%}<br>P99 latency: %{y:.6g} ms<br>Process RSS after inference: %{customdata[0]:.6g} MiB<br>Serialized pickle size: %{customdata[1]:.6g} MiB<extra></extra>",
            )
        )
    if not fig.data:
        fig.add_annotation(
            text="Trade-off chart requires both test accuracy and P99 latency.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    apply_theme(
        fig,
        "Predictive performance vs inference cost",
        "Each point is a model with both observed test accuracy and P99 latency",
        theme,
    )
    fig.update_xaxes(title="Test accuracy", tickformat=".0%")
    fig.update_yaxes(title="P99 inference latency (ms)")
    return fig
