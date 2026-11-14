"""Batch inference latency and throughput measurement."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from modeldiet.profiling.measurements import BaseMeasurement, _validate_samples


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must contain positive integers")
    return value


def _slice_batch(X: Any, start: int, stop: int) -> Any:
    """Slice rows without converting the caller's container type."""
    if hasattr(X, "iloc"):
        return X.iloc[start:stop]
    return X[start:stop]


class BatchInferenceMeasurement(BaseMeasurement):
    """Measure batch-call latency and samples per second for each batch size.

    Batch latency statistics describe measured calls to ``predict``. They are
    not per-sample latency statistics. Throughput is the number of measured
    samples divided by the total measured inference time.
    """

    name = "batch_inference"

    def __init__(self, batch_sizes: Sequence[int], warmup_iterations: int = 1) -> None:
        sizes = tuple(_positive_integer(size, "batch_sizes") for size in batch_sizes)
        if not sizes:
            raise ValueError("batch_sizes must contain at least one batch size")
        if len(set(sizes)) != len(sizes):
            raise ValueError("batch_sizes must not contain duplicates")
        if (
            isinstance(warmup_iterations, bool)
            or not isinstance(warmup_iterations, int)
            or warmup_iterations < 0
        ):
            raise ValueError("warmup_iterations must be a non-negative integer")
        self.batch_sizes = sizes
        self.warmup_iterations = warmup_iterations

    def measure(self, model: Any, X: Any = None) -> dict[str, Any]:
        sample_count = _validate_samples(X)
        metrics: dict[str, Any] = {}
        for batch_size in self.batch_sizes:
            first_batch = _slice_batch(X, 0, min(batch_size, sample_count))
            for _ in range(self.warmup_iterations):
                model.predict(first_batch)

            durations_s: list[float] = []
            measured_samples = 0
            for start in range(0, sample_count, batch_size):
                batch = _slice_batch(X, start, min(start + batch_size, sample_count))
                start_time = time.perf_counter()
                model.predict(batch)
                durations_s.append(time.perf_counter() - start_time)
                measured_samples += min(batch_size, sample_count - start)

            durations_ms = np.asarray(durations_s, dtype=float) * 1000.0
            total_seconds = float(sum(durations_s))
            prefix = f"batch_size_{batch_size}"
            metrics.update(
                {
                    f"{prefix}_latency_mean_ms": float(np.mean(durations_ms)),
                    f"{prefix}_latency_p50_ms": float(np.percentile(durations_ms, 50)),
                    f"{prefix}_latency_p95_ms": float(np.percentile(durations_ms, 95)),
                    f"{prefix}_latency_p99_ms": float(np.percentile(durations_ms, 99)),
                    f"{prefix}_batch_count": len(durations_s),
                    f"{prefix}_sample_count": measured_samples,
                    f"{prefix}_throughput_samples_per_second": (
                        measured_samples / total_seconds if total_seconds > 0 else 0.0
                    ),
                }
            )
        return metrics
