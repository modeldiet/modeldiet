"""Convert validated result records into interactive Plotly figures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from modeldiet.visualization.data import (
    Results,
    accuracy_value,
    latency_observations,
    metric_number,
    repeated_metric_values,
    repeated_series,
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
    """Plot raw latency ECDFs, using reported percentiles when samples are absent."""
    go = _go()
    fig = go.Figure()
    found = False
    summaries = 0
    for name, record in results.items():
        values = sorted(latency_observations(record))
        color = ACCENTS[len(fig.data) % len(ACCENTS)]
        if values:
            found = True
            fig.add_trace(
                go.Scatter(
                    x=values,
                    y=[(i + 1) / len(values) for i in range(len(values))],
                    mode="lines",
                    name=f"Observed ECDF · {name}",
                    line={"color": color},
                    hovertemplate="Latency: %{x:.6g} ms<br>Cumulative fraction: %{y:.1%}<extra>%{fullData.name}</extra>",
                )
            )
            continue

        # Keep a visible latency-distribution comparison even when the source
        # retained percentiles only. These are reported points, not synthesized
        # samples or an estimated ECDF.
        quantiles = [
            (q, metric_number(record, "latency", key))
            for q, key in ((0.50, "p50_ms"), (0.95, "p95_ms"), (0.99, "p99_ms"))
        ]
        quantiles = [(q, value) for q, value in quantiles if value is not None]
        if quantiles:
            found = True
            summaries += 1
            fig.add_trace(
                go.Scatter(
                    x=[value for _, value in quantiles],
                    y=[q for q, _ in quantiles],
                    mode="lines+markers",
                    name=f"Reported percentiles · {name}",
                    line={"color": color, "dash": "dash"},
                    marker={"size": 8},
                    hovertemplate="Reported latency: %{x:.6g} ms<br>Percentile: %{y:.0%}<extra>%{fullData.name}</extra>",
                )
            )
    if not found:
        fig.add_annotation(
            text="Latency distribution needs recorded samples or at least one reported latency percentile.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            align="center",
        )
    subtitle = "Recorded observations where available; dashed lines connect reported percentiles otherwise"
    apply_theme(fig, "Latency distribution", subtitle, theme)
    if summaries:
        fig.add_annotation(
            text="Dashed lines show reported percentiles only; they do not represent raw samples.",
            showarrow=False,
            x=1,
            y=1.12,
            xref="paper",
            yref="paper",
            xanchor="right",
            font={"size": 10},
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


def create_repetition_variability_chart(
    results: Results, metric: str = "latency.p99", *, theme: str = "light"
) -> Any:
    """Show one selected run-level metric value per independent repetition."""
    go = _go()
    figure = go.Figure()
    for name, record in results.items():
        values = repeated_metric_values(record, metric)
        if values:
            figure.add_trace(
                go.Box(
                    name=name,
                    y=values,
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    marker_color=ACCENTS[len(figure.data) % len(ACCENTS)],
                    hovertemplate="Independent-run value: %{y:.6g}<extra>%{fullData.name}</extra>",
                )
            )
    if not figure.data:
        figure.add_annotation(
            text="Repeated-run results with the selected run-level metric are required.",
            showarrow=False,
        )
    apply_theme(
        figure,
        "Run-to-run variability",
        f"One {metric} value per independent repetition · no prediction observations pooled",
        theme,
    )
    figure.update_yaxes(title=f"{metric} · metric unit")
    figure.update_xaxes(title="Model or experiment")
    return figure


def create_batch_throughput_chart(results: Results, *, theme: str = "light") -> Any:
    """Compare run-level mean throughput by batch size."""
    go = _go()
    figure = go.Figure()
    for name, record in results.items():
        means = repeated_series(record, "batch_throughput.samples_per_second.")
        if not means:
            continue
        sizes = list(means)
        identifiers = [
            f"batch_throughput.samples_per_second.batch_size_{size}" for size in sizes
        ]
        errors = [
            getattr(getattr(record, "summary", {}).get(identifier), "std_dev", 0.0)
            for identifier in identifiers
        ]
        figure.add_trace(
            go.Scatter(
                name=name,
                x=sizes,
                y=[means[size] for size in sizes],
                mode="lines+markers",
                error_y={"type": "data", "array": errors, "visible": True},
                hovertemplate="Batch size: %{x}<br>Mean throughput: %{y:.6g} samples/s<extra>%{fullData.name}</extra>",
            )
        )
    if not figure.data:
        figure.add_annotation(
            text="Repeated results with BatchInferenceMeasurement are required.",
            showarrow=False,
        )
    apply_theme(
        figure,
        "Throughput by batch size",
        "Mean of each repetition's measured samples per second; error bars show run-to-run standard deviation",
        theme,
    )
    figure.update_xaxes(title="Batch size", type="linear")
    figure.update_yaxes(title="Throughput (samples/s)")
    return figure


def create_batch_latency_chart(results: Results, *, theme: str = "light") -> Any:
    """Compare mean and P95 batch-call latency across batch sizes."""
    go = _go()
    figure = go.Figure()
    for name, record in results.items():
        for percentile, prefix, dash in (
            ("Mean batch latency", "batch_latency.mean.", "solid"),
            ("P95 batch latency", "batch_latency.p95.", "dash"),
        ):
            values = repeated_series(record, prefix)
            if not values:
                continue
            figure.add_trace(
                go.Scatter(
                    name=f"{name} · {percentile}",
                    x=list(values),
                    y=list(values.values()),
                    mode="lines+markers",
                    line={"dash": dash},
                    hovertemplate="Batch size: %{x}<br>Mean of per-run batch latency: %{y:.6g} ms<extra>%{fullData.name}</extra>",
                )
            )
    if not figure.data:
        figure.add_annotation(
            text="Repeated results with BatchInferenceMeasurement are required.",
            showarrow=False,
        )
    apply_theme(
        figure,
        "Batch-call latency by batch size",
        "Mean and P95 summarize batch calls within each run; plotted values are means across independent runs",
        theme,
    )
    figure.update_xaxes(title="Batch size", type="linear")
    figure.update_yaxes(title="Batch-call latency (ms)")
    return figure


def create_peak_rss_comparison_chart(results: Results, *, theme: str = "light") -> Any:
    """Compare means of sampled peak RSS across independent repetitions."""
    go = _go()
    figure = go.Figure()
    names: list[str] = []
    means: list[float] = []
    errors: list[float] = []
    methods: list[str] = []
    for name, record in results.items():
        aggregate = getattr(record, "summary", {}).get("peak_rss.sampled_peak")
        if aggregate is None:
            continue
        names.append(name)
        means.append(aggregate.mean)
        errors.append(aggregate.std_dev)
        method = "sampled process RSS"
        for run in getattr(record, "runs", ()):
            if run.result is None:
                continue
            measurement = run.result.measurements.get("peak_process_rss")
            value = measurement.metrics.get("peak_rss_method") if measurement else None
            if value is not None:
                method = str(value.value)
                break
        methods.append(method)
    if names:
        figure.add_trace(
            go.Bar(
                name="Mean sampled peak RSS",
                x=names,
                y=means,
                error_y={"type": "data", "array": errors, "visible": True},
                customdata=methods,
                hovertemplate="Model: %{x}<br>Mean sampled peak RSS: %{y:.6g} MiB<br>Method: %{customdata}<extra></extra>",
            )
        )
    else:
        figure.add_annotation(
            text="Repeated results with PeakRSSMeasurement are required.",
            showarrow=False,
        )
    apply_theme(
        figure,
        "Sampled peak process RSS",
        "Whole-process RSS sampled during inference; brief spikes may be missed and values are not model-only",
        theme,
    )
    figure.update_xaxes(title="Model or experiment")
    figure.update_yaxes(title="Sampled peak RSS (MiB)")
    return figure
