"""Convert validated result records into interactive Plotly figures."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from modeldiet.result import ProfileResult
from modeldiet.visualization.theme import ACCENTS, apply_theme

Results = Mapping[str, ProfileResult | Mapping[str, Any]]


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


def _metric(
    record: ProfileResult | Mapping[str, Any], measurement: str, key: str
) -> Any:
    if isinstance(record, ProfileResult):
        result = record.measurements.get(measurement)
        value = result.metrics.get(key) if result else None
        return value.value if value is not None else None
    aliases = {
        "latency": "latency",
        "process_rss": "process_memory",
        "python_traced_allocation": "allocation",
        "pickle_size": "serialization",
    }
    nested = record.get("measurement_details", {}).get(
        aliases.get(measurement, measurement), {}
    )
    if key in nested:
        return nested[key]
    flat_aliases = {
        ("latency", "mean_ms"): "mean_inference_latency",
        ("latency", "p50_ms"): "p50_latency",
        ("latency", "p95_ms"): "p95_latency",
        ("latency", "p99_ms"): "p99_latency",
        ("latency", "sample_count"): "sample_count",
        ("process_rss", "rss_before_inference_mb"): "rss_before_inference_mb",
        ("process_rss", "rss_after_inference_mb"): "rss_after_inference_mb",
        ("process_rss", "rss_delta_mb"): "rss_delta_mb",
        (
            "python_traced_allocation",
            "traced_peak_inference_allocation_mb",
        ): "traced_peak_inference_allocation_mb",
        ("pickle_size", "serialized_model_size_mb"): "serialized_model_size",
    }
    if measurement == "latency" and key == "sample_count":
        return record.get("sample_count")
    return record.get(flat_aliases.get((measurement, key), key))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _models(results: Results) -> list[str]:
    return [str(name) for name in results]


def _bar(
    results: Results,
    title: str,
    subtitle: str,
    specs: Sequence[tuple[str, str, str]],
    *,
    horizontal: bool = False,
) -> Any:
    go = _go()
    names = _models(results)
    fig = go.Figure()
    for index, (label, measurement, key) in enumerate(specs):
        values = [_number(_metric(results[name], measurement, key)) for name in results]
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
    fig.update_layout(barmode="group")
    apply_theme(fig, title, subtitle)
    return fig


def create_latency_percentile_chart(results: Results) -> Any:
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
    )
    fig.update_yaxes(title="Latency (ms)")
    fig.update_xaxes(title="Model")
    return fig


def create_latency_distribution_chart(results: Results) -> Any:
    """Show an ECDF only when original per-observation latency samples exist."""
    go = _go()
    fig = go.Figure()
    found = False
    for name, record in results.items():
        raw = (
            record.get("latency_observations_ms")
            if isinstance(record, Mapping)
            else None
        )
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        values = sorted(
            number for value in raw if (number := _number(value)) is not None
        )
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
    )
    fig.update_xaxes(title="Latency (ms)")
    fig.update_yaxes(title="Cumulative fraction", tickformat=".0%", range=[0, 1])
    return fig


def create_rss_comparison_chart(results: Results) -> Any:
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
    )
    fig.update_yaxes(title="RSS (MiB)")
    fig.update_xaxes(title="Model")
    return fig


def create_python_memory_chart(results: Results) -> Any:
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
    )
    fig.update_yaxes(title="Python-traced allocation (MiB)")
    fig.update_xaxes(title="Model")
    return fig


def create_loading_time_chart(results: Results) -> Any:
    """Compare measured model deserialization time in seconds."""
    fig = _bar(
        results,
        "Model loading time",
        "Measured deserialization duration · seconds",
        [("Loading time", "", "loading_time")],
        horizontal=True,
    )
    fig.update_xaxes(title="Loading time (seconds)")
    fig.update_yaxes(title="Model", autorange="reversed")
    return fig


def create_artifact_size_chart(results: Results) -> Any:
    """Compare serialized model artifact representations in MiB."""
    fig = _bar(
        results,
        "Serialized model size",
        "Python pickle representation · MiB; not in-memory size",
        [("Serialized size", "pickle_size", "serialized_model_size_mb")],
        horizontal=True,
    )
    fig.update_xaxes(title="Serialized size (MiB)")
    fig.update_yaxes(title="Model", autorange="reversed")
    return fig


def create_accuracy_latency_chart(results: Results) -> Any:
    """Plot observed test accuracy against P99 latency when both are available."""
    go = _go()
    fig = go.Figure()
    for index, (name, record) in enumerate(results.items()):
        accuracy = (
            _number(record.get("accuracy")) if isinstance(record, Mapping) else None
        )
        latency = _number(_metric(record, "latency", "p99_ms"))
        if accuracy is None or latency is None:
            continue
        rss = _number(_metric(record, "process_rss", "rss_after_inference_mb"))
        size = _number(_metric(record, "pickle_size", "serialized_model_size_mb"))
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
    )
    fig.update_xaxes(title="Test accuracy", tickformat=".0%")
    fig.update_yaxes(title="P99 inference latency (ms)")
    return fig
