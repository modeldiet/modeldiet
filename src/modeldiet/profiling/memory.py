"""Process-wide peak RSS sampling measurement."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any

import psutil

from modeldiet.profiling.measurements import BaseMeasurement, _sample, _validate_samples


@dataclass(frozen=True)
class PeakRSSMetrics:
    """Sampled whole-process RSS values, in MiB."""

    rss_initial_mb: float
    rss_peak_observed_mb: float
    rss_peak_increment_mb: float
    peak_rss_method: str
    sampling_interval_ms: int


class PeakRSSMeasurement(BaseMeasurement):
    """Sample process RSS while a separate prediction pass is running.

    The periodic sampler can miss short-lived spikes. RSS includes the full
    Python process and is not memory consumed exclusively by the model.
    """

    name = "peak_process_rss"

    def __init__(self, sampling_interval_ms: int = 5) -> None:
        if (
            isinstance(sampling_interval_ms, bool)
            or not isinstance(sampling_interval_ms, int)
            or sampling_interval_ms < 1
        ):
            raise ValueError("sampling_interval_ms must be a positive integer")
        self.sampling_interval_ms = sampling_interval_ms

    def measure(self, model: Any, X: Any = None) -> dict[str, Any]:
        sample_count = _validate_samples(X)
        process = psutil.Process()
        rss_lock = threading.Lock()
        started = threading.Event()
        stopped = threading.Event()
        initial_rss = process.memory_info().rss
        peak_rss = [initial_rss]
        monitor_errors: list[Exception] = []

        def sample_rss() -> None:
            try:
                current_rss = process.memory_info().rss
                with rss_lock:
                    peak_rss[0] = max(peak_rss[0], current_rss)
            except Exception as exc:  # pragma: no cover - process disappears mid-run
                monitor_errors.append(exc)

        def monitor() -> None:
            sample_rss()
            started.set()
            while not stopped.wait(self.sampling_interval_ms / 1000.0):
                sample_rss()

        thread = threading.Thread(
            target=monitor,
            name="modeldiet-peak-rss-sampler",
            daemon=True,
        )
        thread.start()
        try:
            started.wait()
            for index in range(sample_count):
                model.predict(_sample(X, index))
        finally:
            stopped.set()
            thread.join()
            sample_rss()

        if monitor_errors:
            raise RuntimeError(f"RSS monitor failed: {monitor_errors[0]}")
        with rss_lock:
            observed_peak = peak_rss[0]
        mb = 1024.0 * 1024.0
        initial_mb = initial_rss / mb
        peak_mb = max(initial_rss, observed_peak) / mb
        return asdict(
            PeakRSSMetrics(
                rss_initial_mb=float(initial_mb),
                rss_peak_observed_mb=float(peak_mb),
                rss_peak_increment_mb=float(max(0.0, peak_mb - initial_mb)),
                peak_rss_method="sampled process RSS via psutil polling thread",
                sampling_interval_ms=self.sampling_interval_ms,
            )
        )
