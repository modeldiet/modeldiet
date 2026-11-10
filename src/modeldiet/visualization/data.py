"""Shared accessors for supported profiling result representations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeAlias

from modeldiet.result import ProfileResult

Record: TypeAlias = ProfileResult | Mapping[str, Any]
Results: TypeAlias = Mapping[str, Record]

# One canonical mapping for ProfileResult measurement names, legacy JSON names,
# and the flat keys used by older benchmark records.
MEASUREMENT_ALIASES = {
    "latency": "latency",
    "process_rss": "process_memory",
    "python_traced_allocation": "allocation",
    "pickle_size": "serialization",
}
FLAT_ALIASES = {
    ("latency", "mean_ms"): "mean_inference_latency",
    ("latency", "p50_ms"): "p50_latency",
    ("latency", "p95_ms"): "p95_latency",
    ("latency", "p99_ms"): "p99_latency",
    ("process_rss", "rss_before_inference_mb"): "rss_before_inference_mb",
    ("process_rss", "rss_after_inference_mb"): "rss_after_inference_mb",
    ("process_rss", "rss_delta_mb"): "rss_delta_mb",
    (
        "python_traced_allocation",
        "traced_peak_inference_allocation_mb",
    ): "traced_peak_inference_allocation_mb",
    ("pickle_size", "serialized_model_size_mb"): "serialized_model_size",
}


def validate_results(results: Any) -> Results:
    """Validate and copy a name-to-result mapping with actionable errors."""
    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping of model names to profile results")
    for name, record in results.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model names must be non-empty strings")
        if not isinstance(record, (ProfileResult, Mapping)):
            raise TypeError(f"result for '{name}' must be a ProfileResult or mapping")
    return dict(results)


def metric_value(record: Record, measurement: str, key: str) -> Any:
    """Read a metric from ProfileResult, nested mappings, or legacy flat JSON."""
    if isinstance(record, ProfileResult):
        measurement_result = record.measurements.get(measurement)
        value = measurement_result.metrics.get(key) if measurement_result else None
        return value.value if value is not None else None
    details = record.get("measurement_details", {})
    if not isinstance(details, Mapping):
        details = {}
    nested = details.get(MEASUREMENT_ALIASES.get(measurement, measurement), {})
    if not isinstance(nested, Mapping):
        nested = {}
    if key in nested:
        return nested[key]
    if measurement == "latency" and key == "sample_count":
        return record.get("sample_count")
    return record.get(FLAT_ALIASES.get((measurement, key), key))


def number(value: Any) -> float | None:
    """Return finite numeric values; booleans and non-finite values are missing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def metric_number(record: Record, measurement: str, key: str) -> float | None:
    """Read and validate a numeric measurement value."""
    return number(metric_value(record, measurement, key))


def accuracy_value(record: Record) -> float | None:
    """Read numeric test accuracy where supplied by the result mapping."""
    return number(record.get("accuracy") if isinstance(record, Mapping) else None)


def loading_time_value(record: Record) -> float | None:
    """Read numeric loading time where supplied by the result mapping."""
    return number(record.get("loading_time") if isinstance(record, Mapping) else None)


def latency_observations(record: Record) -> list[float]:
    """Return finite retained latency samples, without synthesizing observations."""
    raw = record.get("latency_observations_ms") if isinstance(record, Mapping) else None
    if not isinstance(raw, (list, tuple)):
        return []
    return [value for item in raw if (value := number(item)) is not None]


def available_metrics(results: Results) -> dict[str, list[str]]:
    """Return metric identifiers mapped to the models that contain each value."""
    fields = {
        "accuracy": ("", "accuracy"),
        "loading_time_s": ("", "loading_time"),
        "mean_latency_ms": ("latency", "mean_ms"),
        "p50_ms": ("latency", "p50_ms"),
        "p95_ms": ("latency", "p95_ms"),
        "p99_ms": ("latency", "p99_ms"),
        "rss_before_mib": ("process_rss", "rss_before_inference_mb"),
        "rss_after_mib": ("process_rss", "rss_after_inference_mb"),
        "rss_delta_mib": ("process_rss", "rss_delta_mb"),
        "python_traced_peak_mib": (
            "python_traced_allocation",
            "traced_peak_inference_allocation_mb",
        ),
        "serialized_size_mib": ("pickle_size", "serialized_model_size_mb"),
    }
    found: dict[str, list[str]] = {}
    for metric, (measurement, key) in fields.items():
        models = []
        for name, record in results.items():
            raw = (
                accuracy_value(record)
                if metric == "accuracy"
                else loading_time_value(record)
                if metric == "loading_time_s"
                else metric_number(record, measurement, key)
            )
            if raw is not None:
                models.append(name)
        if models:
            found[metric] = models
    return found
