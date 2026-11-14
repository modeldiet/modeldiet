"""Structured, serializable results from a Model Diet profile."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REPEATED_SCHEMA_VERSION = 1
REPEATED_RESULT_TYPE = "modeldiet.repeated_profile"


def _clean_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


@dataclass(frozen=True)
class MetricDefinition:
    identifier: str
    unit: str
    description: str
    semantics: str


@dataclass(frozen=True)
class MetricValue:
    definition: MetricDefinition
    value: int | float | str | bool | None


@dataclass(frozen=True)
class MeasurementError:
    exception_type: str
    message: str


@dataclass(frozen=True)
class MeasurementResult:
    name: str
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    error: MeasurementError | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ProfileResult:
    schema_version: int
    measurements: Mapping[str, MeasurementResult]
    model_class: str
    model_module: str
    input_type: str
    sample_count: int | None
    feature_count: int | None
    environment: Mapping[str, Any] | None = None

    def _json_value(self) -> dict[str, Any]:
        return _clean_json(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProfileResult:
        """Reconstruct a schema-version-1 single-run result from JSON data."""
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ProfileResult schema version: {payload.get('schema_version')}"
            )
        measurements: dict[str, MeasurementResult] = {}
        raw_measurements = payload.get("measurements", {})
        if not isinstance(raw_measurements, Mapping):
            raise ValueError("ProfileResult measurements must be an object")
        for name, raw_result in raw_measurements.items():
            if not isinstance(raw_result, Mapping):
                raise ValueError(f"Measurement '{name}' must be an object")
            raw_metrics = raw_result.get("metrics", {})
            if not isinstance(raw_metrics, Mapping):
                raise ValueError(f"Metrics for '{name}' must be an object")
            metrics: dict[str, MetricValue] = {}
            for key, raw_metric in raw_metrics.items():
                if not isinstance(raw_metric, Mapping):
                    raise ValueError(f"Metric '{key}' must be an object")
                raw_definition = raw_metric.get("definition", {})
                if not isinstance(raw_definition, Mapping):
                    raise ValueError(f"Definition for metric '{key}' must be an object")
                metrics[str(key)] = MetricValue(
                    definition=MetricDefinition(
                        identifier=str(raw_definition.get("identifier", "")),
                        unit=str(raw_definition.get("unit", "unspecified")),
                        description=str(raw_definition.get("description", "")),
                        semantics=str(raw_definition.get("semantics", "")),
                    ),
                    value=raw_metric.get("value"),
                )
            raw_error = raw_result.get("error")
            error = (
                MeasurementError(
                    str(raw_error.get("exception_type", "Exception")),
                    str(raw_error.get("message", "")),
                )
                if isinstance(raw_error, Mapping)
                else None
            )
            measurements[str(name)] = MeasurementResult(
                name=str(raw_result.get("name", name)), metrics=metrics, error=error
            )
        environment = payload.get("environment")
        if environment is not None and not isinstance(environment, Mapping):
            raise ValueError("ProfileResult environment must be an object or null")
        return cls(
            schema_version=SCHEMA_VERSION,
            measurements=measurements,
            model_class=str(payload.get("model_class", "")),
            model_module=str(payload.get("model_module", "")),
            input_type=str(payload.get("input_type", "")),
            sample_count=payload.get("sample_count"),
            feature_count=payload.get("feature_count"),
            environment=environment,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ProfileResult:
        """Load an existing single-run schema-version-1 JSON result."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not load ProfileResult from {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError("ProfileResult JSON must contain an object")
        return cls.from_dict(payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self._json_value(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def to_markdown(self, path: str | Path) -> None:
        lines = [
            "# Model Diet profile",
            "",
            f"- Schema version: {self.schema_version}",
            f"- Model: `{self.model_module}.{self.model_class}`",
            f"- Input: `{self.input_type}` ({self.sample_count} samples, {self.feature_count} features)",
            "",
        ]
        for name, result in self.measurements.items():
            lines.extend([f"## {name}", ""])
            if result.error:
                lines.append(
                    f"Failed: `{result.error.exception_type}`: {result.error.message}"
                )
            else:
                lines.extend(
                    [
                        "| Metric | Value | Unit | Description | Semantics |",
                        "|---|---:|---|---|---|",
                    ]
                )
                for metric in result.metrics.values():
                    definition = metric.definition
                    lines.append(
                        f"| `{definition.identifier}` | {metric.value} | {definition.unit} | {definition.description} | {definition.semantics} |"
                    )
            lines.append("")
        if self.environment is not None:
            lines.extend(
                [
                    "## Environment",
                    "",
                    "```json",
                    json.dumps(self.environment, indent=2, allow_nan=False),
                    "```",
                    "",
                ]
            )
        Path(path).write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class AggregateMetricStatistics:
    """Statistics across independent repetition-level metric values."""

    definition: MetricDefinition
    n: int
    mean: float
    median: float
    std_dev: float
    minimum: float
    maximum: float
    coefficient_of_variation: float
    p05: float
    p95: float


@dataclass(frozen=True)
class RepeatedRunResult:
    """One attempted repetition, including a failure when it did not complete."""

    repetition_index: int
    status: str
    result: ProfileResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    environment: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepeatedProfileResult:
    """Typed result for repeated independent profiles of one fitted model."""

    schema_version: int
    experiment_id: str
    experiment_name: str | None
    generated_at: str
    repetitions_requested: int
    repetitions_completed: int
    model_class: str
    model_module: str
    workload: Mapping[str, Any]
    runs: tuple[RepeatedRunResult, ...]
    summary: Mapping[str, AggregateMetricStatistics]
    environment: Mapping[str, Any] | None = None
    result_type: str = REPEATED_RESULT_TYPE

    @property
    def repetitions_attempted(self) -> int:
        """Number of requested repetitions that produced a run record."""
        return len(self.runs)

    @property
    def failed_runs(self) -> tuple[RepeatedRunResult, ...]:
        """Failed repetition records; no failed attempts are omitted."""
        return tuple(run for run in self.runs if run.status != "completed")

    def _json_value(self) -> dict[str, Any]:
        return _clean_json(asdict(self))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self._json_value(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def to_markdown(self, path: str | Path) -> None:
        lines = [
            "# ModelDiet repeated profile",
            "",
            f"- Experiment: {self.experiment_name or self.experiment_id}",
            f"- Experiment ID: `{self.experiment_id}`",
            f"- Generated: {self.generated_at}",
            f"- Model: `{self.model_module}.{self.model_class}`",
            f"- Repetitions: {self.repetitions_completed}/{self.repetitions_requested} completed; {len(self.failed_runs)} failed",
            "",
            "## Run-level aggregate summary",
            "",
            "Statistics below aggregate the corresponding metric once per independent repetition. They do not pool per-prediction observations.",
            "",
            "| Metric | Mean | Median | Std dev | Min | Max | CV (%) | n | P05 | P95 | Unit |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for stats in self.summary.values():
            definition = stats.definition
            lines.append(
                f"| `{definition.identifier}` | {stats.mean:.6g} | {stats.median:.6g} | {stats.std_dev:.6g} | {stats.minimum:.6g} | {stats.maximum:.6g} | {stats.coefficient_of_variation:.6g} | {stats.n} | {stats.p05:.6g} | {stats.p95:.6g} | {definition.unit} |"
            )
        lines.extend(["", "## Repetitions", ""])
        for run in self.runs:
            lines.append(f"### Repetition {run.repetition_index}: {run.status}")
            lines.append("")
            if run.error_message:
                lines.append(f"`{run.error_type or 'Error'}`: {run.error_message}")
            elif run.result is not None:
                for name, measurement in run.result.measurements.items():
                    if measurement.error:
                        lines.append(
                            f"- {name}: failed ({measurement.error.exception_type}: {measurement.error.message})"
                        )
                    else:
                        values = ", ".join(
                            f"`{metric.definition.identifier}` = {metric.value} {metric.definition.unit}"
                            for metric in measurement.metrics.values()
                        )
                        lines.append(f"- {name}: {values}")
            lines.append("")
        if self.environment is not None:
            lines.extend(
                [
                    "## Launcher environment",
                    "",
                    "```json",
                    json.dumps(self.environment, indent=2),
                    "```",
                    "",
                ]
            )
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepeatedProfileResult:
        """Reconstruct the explicit repeated-result JSON representation."""
        if payload.get("result_type") != REPEATED_RESULT_TYPE:
            raise ValueError("JSON result_type is not a repeated ModelDiet profile")
        if payload.get("schema_version") != REPEATED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported repeated result schema version: {payload.get('schema_version')}"
            )
        raw_runs = payload.get("runs", [])
        if not isinstance(raw_runs, list):
            raise ValueError("RepeatedProfileResult runs must be an array")
        runs: list[RepeatedRunResult] = []
        for raw_run in raw_runs:
            if not isinstance(raw_run, Mapping):
                raise ValueError("Each repeated run must be an object")
            raw_profile = raw_run.get("result")
            raw_environment = raw_run.get("environment")
            raw_execution = raw_run.get("execution", {})
            runs.append(
                RepeatedRunResult(
                    repetition_index=int(raw_run.get("repetition_index", 0)),
                    status=str(raw_run.get("status", "failed")),
                    result=(
                        ProfileResult.from_dict(raw_profile)
                        if isinstance(raw_profile, Mapping)
                        else None
                    ),
                    error_type=(
                        str(raw_run["error_type"])
                        if raw_run.get("error_type") is not None
                        else None
                    ),
                    error_message=(
                        str(raw_run["error_message"])
                        if raw_run.get("error_message") is not None
                        else None
                    ),
                    environment=(
                        raw_environment
                        if isinstance(raw_environment, Mapping)
                        else None
                    ),
                    execution=(
                        raw_execution if isinstance(raw_execution, Mapping) else {}
                    ),
                )
            )
        raw_summary = payload.get("summary", {})
        if not isinstance(raw_summary, Mapping):
            raise ValueError("RepeatedProfileResult summary must be an object")
        summary: dict[str, AggregateMetricStatistics] = {}
        for identifier, raw_stats in raw_summary.items():
            if not isinstance(raw_stats, Mapping):
                raise ValueError(f"Aggregate '{identifier}' must be an object")
            raw_definition = raw_stats.get("definition", {})
            if not isinstance(raw_definition, Mapping):
                raise ValueError(
                    f"Definition for aggregate '{identifier}' must be an object"
                )
            definition = MetricDefinition(
                identifier=str(raw_definition.get("identifier", identifier)),
                unit=str(raw_definition.get("unit", "unspecified")),
                description=str(raw_definition.get("description", "")),
                semantics=str(raw_definition.get("semantics", "")),
            )
            summary[str(identifier)] = AggregateMetricStatistics(
                definition=definition,
                n=int(raw_stats.get("n", 0)),
                mean=float(raw_stats.get("mean", 0.0)),
                median=float(raw_stats.get("median", 0.0)),
                std_dev=float(raw_stats.get("std_dev", 0.0)),
                minimum=float(raw_stats.get("minimum", 0.0)),
                maximum=float(raw_stats.get("maximum", 0.0)),
                coefficient_of_variation=float(
                    raw_stats.get("coefficient_of_variation", 0.0)
                ),
                p05=float(raw_stats.get("p05", 0.0)),
                p95=float(raw_stats.get("p95", 0.0)),
            )
        raw_workload = payload.get("workload", {})
        raw_environment = payload.get("environment")
        return cls(
            schema_version=REPEATED_SCHEMA_VERSION,
            experiment_id=str(payload.get("experiment_id", "")),
            experiment_name=(
                str(payload["experiment_name"])
                if payload.get("experiment_name") is not None
                else None
            ),
            generated_at=str(payload.get("generated_at", "")),
            repetitions_requested=int(payload.get("repetitions_requested", len(runs))),
            repetitions_completed=int(
                payload.get(
                    "repetitions_completed",
                    sum(run.status == "completed" for run in runs),
                )
            ),
            model_class=str(payload.get("model_class", "")),
            model_module=str(payload.get("model_module", "")),
            workload=raw_workload if isinstance(raw_workload, Mapping) else {},
            runs=tuple(runs),
            summary=summary,
            environment=(
                raw_environment if isinstance(raw_environment, Mapping) else None
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> RepeatedProfileResult:
        """Load a repeated profile from its discriminated JSON format."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not load repeated ModelDiet result from {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError("RepeatedProfileResult JSON must contain an object")
        return cls.from_dict(payload)
