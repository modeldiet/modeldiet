"""Run measurement strategies independently and assemble profile results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from modeldiet.profiling.measurements import BaseMeasurement
from modeldiet.result import (
    SCHEMA_VERSION,
    MeasurementError,
    MeasurementResult,
    MetricDefinition,
    MetricValue,
    ProfileResult,
)

_DEFINITIONS = {
    ("latency", "mean_ms"): MetricDefinition(
        "latency.mean",
        "ms",
        "Mean per-sample inference latency",
        "Single-row predict duration measured without tracemalloc",
    ),
    ("latency", "p50_ms"): MetricDefinition(
        "latency.p50",
        "ms",
        "50th percentile per-sample inference latency",
        "Single-row predict duration measured without tracemalloc",
    ),
    ("latency", "p90_ms"): MetricDefinition(
        "latency.p90",
        "ms",
        "90th percentile per-sample inference latency",
        "Single-row predict duration measured without tracemalloc",
    ),
    ("latency", "p95_ms"): MetricDefinition(
        "latency.p95",
        "ms",
        "95th percentile per-sample inference latency",
        "Single-row predict duration measured without tracemalloc",
    ),
    ("latency", "p99_ms"): MetricDefinition(
        "latency.p99",
        "ms",
        "99th percentile per-sample inference latency",
        "Single-row predict duration measured without tracemalloc",
    ),
    ("latency", "sample_count"): MetricDefinition(
        "latency.sample_count",
        "count",
        "Number of samples timed",
        "Count of samples included in the latency distribution",
    ),
    (
        "python_traced_allocation",
        "traced_peak_inference_allocation_mb",
    ): MetricDefinition(
        "python_traced.peak_allocation",
        "MiB",
        "Peak Python allocation",
        "Peak traced Python allocations during a separate inference pass; excludes untracked native allocations",
    ),
    ("process_rss", "rss_before_inference_mb"): MetricDefinition(
        "process_rss.before",
        "MiB",
        "Process RSS before inference",
        "Whole-process RSS endpoint snapshot; not model-only or peak RSS",
    ),
    ("process_rss", "rss_after_inference_mb"): MetricDefinition(
        "process_rss.after",
        "MiB",
        "Process RSS after inference",
        "Whole-process RSS endpoint snapshot; not model-only or peak RSS",
    ),
    ("process_rss", "rss_delta_mb"): MetricDefinition(
        "process_rss.delta",
        "MiB",
        "Process RSS change",
        "Difference between whole-process RSS endpoint snapshots",
    ),
    ("peak_process_rss", "rss_initial_mb"): MetricDefinition(
        "peak_rss.initial",
        "MiB",
        "Whole-process RSS at the start of the sampled inference pass",
        "Process-wide RSS starting value; includes the Python process and is not model-only",
    ),
    ("peak_process_rss", "rss_peak_observed_mb"): MetricDefinition(
        "peak_rss.sampled_peak",
        "MiB",
        "Maximum whole-process RSS observed by periodic sampling",
        "Sampled process RSS; short-lived peaks may be missed and this is not an OS high-water mark",
    ),
    ("peak_process_rss", "rss_peak_increment_mb"): MetricDefinition(
        "peak_rss.sampled_increment",
        "MiB",
        "Sampled peak RSS above the initial process RSS",
        "Difference between sampled whole-process peak and starting RSS; not model-only memory",
    ),
    ("peak_process_rss", "peak_rss_method"): MetricDefinition(
        "peak_rss.method",
        "method",
        "Method used to observe process RSS",
        "Identifies the sampled measurement method and its limitations",
    ),
    ("peak_process_rss", "sampling_interval_ms"): MetricDefinition(
        "peak_rss.sampling_interval",
        "ms",
        "Configured interval between RSS samples",
        "A short interval improves temporal resolution but cannot guarantee every allocation peak is observed",
    ),
    ("pickle_size", "serialized_model_size_mb"): MetricDefinition(
        "pickle.size",
        "MiB",
        "Pickle representation size",
        "Size of pickle.dumps(model) bytes; does not imply portable or safe loading",
    ),
    ("pickle_size", "serialization_format"): MetricDefinition(
        "pickle.format", "format", "Serialization format", "Python pickle format"
    ),
}

_BATCH_KEY = re.compile(
    r"batch_size_(?P<size>\d+)_(?P<metric>latency_(?:mean|p50|p95|p99)_ms|batch_count|sample_count|throughput_samples_per_second)"
)


def _batch_definition(key: str) -> MetricDefinition | None:
    match = _BATCH_KEY.fullmatch(key)
    if match is None:
        return None
    size = match.group("size")
    metric = match.group("metric")
    if metric.startswith("latency_"):
        percentile = metric.removeprefix("latency_").removesuffix("_ms")
        labels = {"mean": "Mean", "p50": "P50", "p95": "P95", "p99": "P99"}
        identifier = f"batch_latency.{percentile}.batch_size_{size}"
        return MetricDefinition(
            identifier,
            "ms",
            f"{labels[percentile]} latency across measured batch calls",
            f"Batch-call latency for batches of {size}; percentiles are across calls, not individual samples",
        )
    if metric == "throughput_samples_per_second":
        return MetricDefinition(
            f"batch_throughput.samples_per_second.batch_size_{size}",
            "samples/s",
            "Measured samples processed per second",
            f"Number of samples divided by total measured predict time for batch size {size}; warmups are excluded",
        )
    label = "batches" if metric == "batch_count" else "samples"
    return MetricDefinition(
        f"batch_latency.{metric}.batch_size_{size}",
        "count",
        f"Number of {label} processed",
        f"Measured inference work for batch size {size}; warmup calls are excluded",
    )


def _definition(name: str, key: str) -> MetricDefinition:
    if name == "batch_inference":
        batch_definition = _batch_definition(key)
        if batch_definition is not None:
            return batch_definition
    return _DEFINITIONS.get(
        (name, key),
        MetricDefinition(
            f"{name}.{key}",
            "unspecified",
            key,
            "Custom measurement value; unit and semantics were not specified by a built-in definition",
        ),
    )


def run_measurements(
    model: Any,
    X: Any,
    measurements: Sequence[BaseMeasurement],
    *,
    on_error: Literal["raise", "record"],
) -> ProfileResult:
    names = [getattr(m, "name", m.__class__.__name__) for m in measurements]
    if len(set(names)) != len(names):
        raise ValueError("measurement names must be unique")
    results: dict[str, MeasurementResult] = {}
    for name, measurement in zip(names, measurements):
        try:
            raw = measurement.measure(model, X)
            metrics = {
                key: MetricValue(_definition(name, key), value)
                for key, value in raw.items()
            }
            results[name] = MeasurementResult(name=name, metrics=metrics)
        except Exception as exc:
            if on_error == "raise":
                raise RuntimeError(f"Measurement '{name}' failed: {exc}") from exc
            results[name] = MeasurementResult(
                name=name, error=MeasurementError(type(exc).__name__, str(exc))
            )
    shape = getattr(X, "shape", None)
    sample_count = int(shape[0]) if shape is not None else len(X)
    feature_count = (
        int(shape[1])
        if shape is not None and len(shape) > 1
        else (len(X[0]) if sample_count else None)
    )
    return ProfileResult(
        SCHEMA_VERSION,
        results,
        model.__class__.__name__,
        model.__class__.__module__,
        type(X).__name__,
        sample_count,
        feature_count,
    )
