"""Public Model Diet API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from modeldiet.profiling.measurements import BaseMeasurement, LatencyMeasurement
from modeldiet.profiling.orchestrator import run_measurements
from modeldiet.result import ProfileResult


def profile(
    model: Any,
    X: Any,
    *,
    measurements: Sequence[BaseMeasurement] | None = None,
    on_error: Literal["raise", "record"] = "raise",
    include_environment: bool = False,
) -> ProfileResult:
    """Profile a fitted estimator using the selected measurement strategies."""
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    if not callable(getattr(model, "predict", None)):
        raise TypeError("model must provide a callable predict(X) method")
    if X is None:
        raise ValueError("X must contain representative tabular data")
    selected: list[BaseMeasurement]
    if measurements is None:
        selected = [LatencyMeasurement()]
    else:
        selected = list(measurements)
    if not selected:
        raise ValueError("measurements must contain at least one measurement")
    if any(not isinstance(m, BaseMeasurement) for m in selected):
        raise TypeError("each measurement must be a BaseMeasurement instance")
    result = run_measurements(model, X, selected, on_error=on_error)
    if include_environment:
        from dataclasses import replace

        from modeldiet.profiling.environment import collect_environment

        return replace(result, environment=collect_environment())
    return result


# Imported after the single-run API is defined: the isolated worker imports
# ``profile`` from this module as its spawn-safe execution entry point.
from modeldiet.profiling.repeated import (  # noqa: E402
    RepeatedProfilingError,  # noqa: F401
    profile_repeated,  # noqa: F401
)
