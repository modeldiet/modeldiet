"""Metric extraction and safe pairwise comparison helpers."""

from __future__ import annotations

from typing import Any

from modeldiet.visualization.data import (
    Record,
    Results,
    accuracy_value,
    loading_time_value,
    metric_number,
)

METRICS: dict[str, tuple[str, str, str]] = {
    "accuracy": ("", "accuracy", "accuracy"),
    "mean_latency_ms": ("latency", "mean_ms", "Mean latency (ms)"),
    "p50_ms": ("latency", "p50_ms", "P50 latency (ms)"),
    "p95_ms": ("latency", "p95_ms", "P95 latency (ms)"),
    "p99_ms": ("latency", "p99_ms", "P99 latency (ms)"),
    "rss_before_mib": ("process_rss", "rss_before_inference_mb", "RSS before (MiB)"),
    "rss_after_mib": ("process_rss", "rss_after_inference_mb", "RSS after (MiB)"),
    "rss_delta_mib": ("process_rss", "rss_delta_mb", "RSS change (MiB)"),
    "python_traced_peak_mib": (
        "python_traced_allocation",
        "traced_peak_inference_allocation_mb",
        "Python traced peak (MiB)",
    ),
    "loading_time_s": ("", "loading_time", "Loading time (s)"),
    "serialized_size_mib": (
        "pickle_size",
        "serialized_model_size_mb",
        "Serialized size (MiB)",
    ),
}


def value_for(record: Record, metric: str) -> float | None:
    """Extract a finite value for a public comparison metric identifier."""
    if metric not in METRICS:
        raise ValueError(
            f"Unknown comparison metric '{metric}'. Choose from: {', '.join(METRICS)}"
        )
    measurement, key, _ = METRICS[metric]
    if metric == "accuracy":
        return accuracy_value(record)
    if metric == "loading_time_s":
        return loading_time_value(record)
    return metric_number(record, measurement, key)


def pairwise_rows(
    results: Results,
    model_a: str,
    model_b: str,
) -> list[dict[str, Any]]:
    """Return raw values, absolute differences and safe relative changes."""
    if model_a not in results or model_b not in results:
        missing = [name for name in (model_a, model_b) if name not in results]
        raise ValueError(f"Unknown model name(s): {', '.join(missing)}")
    rows = []
    for metric, (_, _, label) in METRICS.items():
        a = value_for(results[model_a], metric)
        b = value_for(results[model_b], metric)
        if a is None or b is None:
            difference = None
            percentage = None
        else:
            difference = b - a
            percentage = difference / abs(a) * 100 if a != 0 else None
        rows.append(
            {
                "metric": metric,
                "label": label,
                "reference": a,
                "comparison": b,
                "difference": difference,
                "percent_change": percentage,
            }
        )
    return rows
