"""Repeated profiling orchestration and independent-run aggregation."""

from __future__ import annotations

import math
import multiprocessing
import pickle
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

from modeldiet.api import profile
from modeldiet.profiling.environment import collect_environment
from modeldiet.profiling.measurements import (
    BaseMeasurement,
    LatencyMeasurement,
    _validate_samples,
)
from modeldiet.profiling.worker import run_profile_worker
from modeldiet.result import (
    REPEATED_SCHEMA_VERSION,
    AggregateMetricStatistics,
    MetricDefinition,
    ProfileResult,
    RepeatedProfileResult,
    RepeatedRunResult,
)


class RepeatedProfilingError(RuntimeError):
    """Raised under ``on_error='raise'`` with the complete partial result."""

    def __init__(self, result: RepeatedProfileResult) -> None:
        failed = ", ".join(
            f"{run.repetition_index} ({run.error_type}: {run.error_message})"
            for run in result.failed_runs
        )
        super().__init__(f"Repeated profiling failed for repetition(s): {failed}")
        self.result = result


def _features(X: Any, sample_count: int) -> int | None:
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) > 1:
        return int(shape[1])
    if sample_count:
        try:
            return len(X[0])
        except (TypeError, KeyError):
            return None
    return None


def _measurement_configuration(measurement: BaseMeasurement) -> dict[str, Any]:
    configuration: dict[str, Any] = {}
    for key, value in vars(measurement).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            configuration[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, int, float, bool)) or item is None for item in value
        ):
            configuration[key] = list(value)
    return {
        "name": str(getattr(measurement, "name", type(measurement).__name__)),
        "class": f"{type(measurement).__module__}.{type(measurement).__qualname__}",
        "configuration": configuration,
    }


def _failure(
    repetition_index: int,
    error_type: str,
    message: str,
    *,
    isolated: bool,
    start_method: str,
    timeout: bool = False,
) -> RepeatedRunResult:
    prefix = f"Repetition {repetition_index} failed"
    if timeout:
        prefix = f"Repetition {repetition_index} timed out"
    return RepeatedRunResult(
        repetition_index=repetition_index,
        status="failed",
        error_type=error_type,
        error_message=f"{prefix}: {message}",
        execution={
            "isolated_process": isolated,
            "process_start_method": start_method,
            "timeout": timeout,
        },
    )


def _run_isolated(
    *,
    repetition_index: int,
    model: Any,
    X: Any,
    measurements: tuple[BaseMeasurement, ...],
    on_error: Literal["raise", "record"],
    include_environment: bool,
    timeout_s: float | None,
) -> RepeatedRunResult:
    start_method = "spawn"
    context = multiprocessing.get_context(start_method)
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(  # type: ignore[attr-defined]
        target=run_profile_worker,
        args=(
            child_connection,
            model,
            X,
            measurements,
            on_error,
            include_environment,
        ),
        name=f"modeldiet-profile-{repetition_index}",
    )
    started_at = time.monotonic()
    try:
        process.start()
    except Exception as exc:
        child_connection.close()
        parent_connection.close()
        return _failure(
            repetition_index,
            type(exc).__name__,
            f"Could not start isolated worker: {exc}",
            isolated=True,
            start_method=start_method,
        )
    child_connection.close()
    try:
        remaining = (
            None
            if timeout_s is None
            else max(0.0, timeout_s - (time.monotonic() - started_at))
        )
        if not parent_connection.poll(remaining):
            return _failure(
                repetition_index,
                "WorkerTimeoutError",
                f"worker exceeded the {timeout_s:g} second timeout",
                isolated=True,
                start_method=start_method,
                timeout=True,
            )
        try:
            message = parent_connection.recv()
        except EOFError:
            process.join(timeout=0.1)
            return _failure(
                repetition_index,
                "WorkerExitedError",
                f"worker exited without returning a result (exit code {process.exitcode})",
                isolated=True,
                start_method=start_method,
            )
        process.join(timeout=1.0)
        if process.is_alive():
            return _failure(
                repetition_index,
                "WorkerExitError",
                "worker sent a result but did not exit cleanly",
                isolated=True,
                start_method=start_method,
            )
        if not isinstance(message, tuple) or not message:
            return _failure(
                repetition_index,
                "WorkerProtocolError",
                "worker returned an invalid response",
                isolated=True,
                start_method=start_method,
            )
        if message[0] == "completed" and isinstance(message[1], ProfileResult):
            return RepeatedRunResult(
                repetition_index=repetition_index,
                status="completed",
                result=message[1],
                environment=message[1].environment,
                execution={
                    "isolated_process": True,
                    "process_start_method": start_method,
                    "worker_startup_included_in_predict_timing": False,
                },
            )
        if message[0] == "failed":
            return _failure(
                repetition_index,
                str(message[1]),
                str(message[2]),
                isolated=True,
                start_method=start_method,
            )
        return _failure(
            repetition_index,
            "WorkerProtocolError",
            "worker returned an unknown response",
            isolated=True,
            start_method=start_method,
        )
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
        if process.pid is not None:
            process.close()


def _run_in_process(
    *,
    repetition_index: int,
    model: Any,
    X: Any,
    measurements: tuple[BaseMeasurement, ...],
    on_error: Literal["raise", "record"],
    include_environment: bool,
) -> RepeatedRunResult:
    try:
        result = profile(
            model,
            X,
            measurements=measurements,
            on_error=on_error,
            include_environment=include_environment,
        )
    except Exception as exc:
        return _failure(
            repetition_index,
            type(exc).__name__,
            str(exc),
            isolated=False,
            start_method="in-process",
        )
    return RepeatedRunResult(
        repetition_index=repetition_index,
        status="completed",
        result=result,
        environment=result.environment,
        execution={
            "isolated_process": False,
            "process_start_method": "in-process",
            "worker_startup_included_in_predict_timing": False,
        },
    )


def _aggregate(
    runs: Sequence[RepeatedRunResult],
) -> dict[str, AggregateMetricStatistics]:
    observations: dict[str, tuple[MetricDefinition, list[float]]] = {}
    for run in runs:
        if run.status != "completed" or run.result is None:
            continue
        for measurement in run.result.measurements.values():
            if not measurement.succeeded:
                continue
            for metric in measurement.metrics.values():
                value = metric.value
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    continue
                key = metric.definition.identifier
                if key not in observations:
                    observations[key] = (metric.definition, [])
                observations[key][1].append(numeric)

    summary: dict[str, AggregateMetricStatistics] = {}
    for identifier, (definition, values) in observations.items():
        array = np.asarray(values, dtype=float)
        mean = float(np.mean(array))
        standard_deviation = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        variation = standard_deviation / abs(mean) * 100.0 if mean != 0 else 0.0
        summary[identifier] = AggregateMetricStatistics(
            definition=definition,
            n=len(values),
            mean=mean,
            median=float(np.median(array)),
            std_dev=standard_deviation,
            minimum=float(np.min(array)),
            maximum=float(np.max(array)),
            coefficient_of_variation=float(variation),
            p05=float(np.percentile(array, 5)),
            p95=float(np.percentile(array, 95)),
        )
    return summary


def profile_repeated(
    model: Any,
    X: Any,
    *,
    repetitions: int = 5,
    measurements: Sequence[BaseMeasurement] | None = None,
    isolate_process: bool = True,
    worker_timeout_s: float | None = 300.0,
    on_error: Literal["raise", "record"] = "raise",
    include_environment: bool = False,
    experiment_name: str | None = None,
) -> RepeatedProfileResult:
    """Profile a fitted model repeatedly, optionally in a fresh spawned process.

    Isolated mode requires the model, input, and strategies to be pickleable.
    Every attempted repetition is retained in ``runs``. With ``on_error='raise'``
    any failed repetition raises :class:`RepeatedProfilingError`, whose
    ``result`` attribute contains completed and failed attempts.
    """
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("repetitions must be a positive integer")
    if not callable(getattr(model, "predict", None)):
        raise TypeError("model must provide a callable predict(X) method")
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    if worker_timeout_s is not None and (
        isinstance(worker_timeout_s, bool)
        or not isinstance(worker_timeout_s, (int, float))
        or not math.isfinite(float(worker_timeout_s))
        or worker_timeout_s <= 0
    ):
        raise ValueError("worker_timeout_s must be a positive finite number or None")
    if experiment_name is not None and (
        not isinstance(experiment_name, str) or not experiment_name.strip()
    ):
        raise ValueError("experiment_name must be a non-empty string or None")
    sample_count = _validate_samples(X)
    selected = (LatencyMeasurement(),) if measurements is None else tuple(measurements)
    if not selected:
        raise ValueError("measurements must contain at least one measurement")
    if any(not isinstance(item, BaseMeasurement) for item in selected):
        raise TypeError("each measurement must be a BaseMeasurement instance")
    names = [str(getattr(item, "name", type(item).__name__)) for item in selected]
    if len(set(names)) != len(names):
        raise ValueError("measurement names must be unique")

    start_method = "spawn" if isolate_process else "in-process"
    if isolate_process:
        try:
            pickle.dumps((model, X, selected))
        except Exception as exc:
            raise ValueError(
                "Isolated profiling requires the model, input data, and measurement "
                "strategies to be serializable with pickle. Use module-level model "
                "and strategy classes, or pass isolate_process=False. "
                f"Serialization failed: {type(exc).__name__}: {exc}"
            ) from exc

    runs: list[RepeatedRunResult] = []
    for repetition_index in range(1, repetitions + 1):
        if isolate_process:
            run = _run_isolated(
                repetition_index=repetition_index,
                model=model,
                X=X,
                measurements=selected,
                on_error=on_error,
                include_environment=include_environment,
                timeout_s=(
                    float(worker_timeout_s) if worker_timeout_s is not None else None
                ),
            )
        else:
            run = _run_in_process(
                repetition_index=repetition_index,
                model=model,
                X=X,
                measurements=selected,
                on_error=on_error,
                include_environment=include_environment,
            )
        if run.result is not None and include_environment:
            environment = dict(run.result.environment or {})
            environment.update(
                {
                    "isolated_process": isolate_process,
                    "process_start_method": start_method,
                }
            )
            profiled = replace(run.result, environment=environment)
            run = replace(run, result=profiled, environment=environment)
        runs.append(run)

    completed = sum(run.status == "completed" for run in runs)
    model_class = type(model).__name__
    model_module = type(model).__module__
    features = _features(X, sample_count)
    configuration = [_measurement_configuration(item) for item in selected]
    workload = {
        "input_type": type(X).__name__,
        "sample_count": sample_count,
        "feature_count": features,
        "repetitions_requested": repetitions,
        "measurements": configuration,
        "isolate_process": isolate_process,
        "process_start_method": start_method,
        "worker_timeout_s": worker_timeout_s,
        "on_error": on_error,
    }
    result = RepeatedProfileResult(
        schema_version=REPEATED_SCHEMA_VERSION,
        experiment_id=str(uuid.uuid4()),
        experiment_name=experiment_name,
        generated_at=datetime.now(UTC).isoformat(),
        repetitions_requested=repetitions,
        repetitions_completed=completed,
        model_class=model_class,
        model_module=model_module,
        workload=workload,
        runs=tuple(runs),
        summary=_aggregate(runs),
        environment=collect_environment() if include_environment else None,
    )
    if on_error == "raise" and result.failed_runs:
        raise RepeatedProfilingError(result)
    return result
