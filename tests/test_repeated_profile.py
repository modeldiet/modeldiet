"""Tests for general-purpose repeated profiling and its result model."""

import json
import os
import time

import numpy as np
import pytest

from modeldiet import RepeatedProfileResult, profile_repeated
from modeldiet.profiling import (
    BaseMeasurement,
    BatchInferenceMeasurement,
    LatencyMeasurement,
    PeakRSSMeasurement,
)
from modeldiet.profiling.repeated import RepeatedProfilingError


class CountingModel:
    def __init__(self, delay_s=0.0):
        self.calls = 0
        self.delay_s = delay_s

    def predict(self, X):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return np.zeros(len(X))


class RunStateMeasurement(BaseMeasurement):
    name = "run_state"

    def measure(self, model, X=None):
        model.predict(X[:1])
        return {"pid": os.getpid(), "model_calls": model.calls}


class IncrementingMeasurement(BaseMeasurement):
    name = "incrementing"

    def __init__(self):
        self.calls = 0

    def measure(self, model, X=None):
        self.calls += 1
        return {"value": float(self.calls)}


class FailingMeasurement(BaseMeasurement):
    name = "broken"

    def measure(self, model, X=None):
        raise ValueError("intentional measurement failure")


class UnpickleableModel(CountingModel):
    def __init__(self):
        super().__init__()
        self.callback = lambda: None


def _value(result, measurement, key):
    return result.runs[0].result.measurements[measurement].metrics[key].value


def test_repeated_profile_runs_in_fresh_spawned_processes():
    result = profile_repeated(
        CountingModel(),
        np.ones((3, 2)),
        repetitions=2,
        measurements=[RunStateMeasurement()],
    )

    assert result.repetitions_requested == 2
    assert result.repetitions_completed == 2
    assert len(result.runs) == 2
    assert all(run.execution["isolated_process"] for run in result.runs)
    assert all(run.execution["process_start_method"] == "spawn" for run in result.runs)
    assert {_value(result, "run_state", "model_calls")} == {1}
    pids = [
        run.result.measurements["run_state"].metrics["pid"].value for run in result.runs
    ]
    assert len(set(pids)) == 2


def test_in_process_runs_preserve_per_run_results_and_aggregate_statistics():
    measurement = IncrementingMeasurement()
    result = profile_repeated(
        CountingModel(),
        [[1, 2], [3, 4]],
        repetitions=3,
        measurements=[measurement],
        isolate_process=False,
    )

    stats = result.summary["incrementing.value"]
    assert [
        run.result.measurements["incrementing"].metrics["value"].value
        for run in result.runs
    ] == [1.0, 2.0, 3.0]
    assert stats.n == 3
    assert stats.mean == 2.0
    assert stats.median == 2.0
    assert stats.std_dev == pytest.approx(1.0)
    assert stats.minimum == 1.0
    assert stats.maximum == 3.0
    assert stats.coefficient_of_variation == pytest.approx(50.0)
    assert result.workload["sample_count"] == 2
    assert result.workload["feature_count"] == 2


def test_repeated_environment_and_result_json_markdown_round_trip(tmp_path):
    result = profile_repeated(
        CountingModel(),
        np.ones((2, 1)),
        repetitions=1,
        measurements=[LatencyMeasurement(warmup_iterations=0)],
        isolate_process=False,
        include_environment=True,
        experiment_name="smoke experiment",
    )
    json_path = tmp_path / "repeat.json"
    markdown_path = tmp_path / "repeat.md"
    result.to_json(json_path)
    result.to_markdown(markdown_path)
    payload = json.loads(json_path.read_text())

    assert payload["result_type"] == "modeldiet.repeated_profile"
    assert payload["schema_version"] == 1
    assert payload["environment"]["python_version"]
    assert payload["runs"][0]["environment"]["isolated_process"] is False
    restored = RepeatedProfileResult.from_json(json_path)
    assert restored.experiment_id == result.experiment_id
    assert restored.summary["latency.p99"].mean == result.summary["latency.p99"].mean
    assert "one independent repetition" in markdown_path.read_text()


def test_single_repetition_standard_deviation_and_zero_mean_cv():
    class ZeroMeasurement(BaseMeasurement):
        name = "zero"

        def measure(self, model, X=None):
            return {"value": 0.0}

    result = profile_repeated(
        CountingModel(), [[1]], measurements=[ZeroMeasurement()], isolate_process=False
    )
    stats = result.summary["zero.value"]
    assert stats.n == 1
    assert stats.std_dev == 0
    assert stats.coefficient_of_variation == 0


@pytest.mark.parametrize("repetitions", [0, -1, True, 1.5])
def test_repetition_count_validation(repetitions):
    with pytest.raises(ValueError, match="positive integer"):
        profile_repeated(
            CountingModel(), [[1]], repetitions=repetitions, isolate_process=False
        )


def test_pickle_failure_is_actionable_and_in_process_fallback_works():
    with pytest.raises(ValueError, match="isolate_process=False"):
        profile_repeated(UnpickleableModel(), [[1]])

    result = profile_repeated(
        UnpickleableModel(),
        [[1]],
        measurements=[LatencyMeasurement(warmup_iterations=0)],
        isolate_process=False,
    )
    assert result.repetitions_completed == 1


def test_measurement_failure_is_preserved_in_result_or_partial_exception():
    recorded = profile_repeated(
        CountingModel(),
        [[1]],
        measurements=[FailingMeasurement()],
        on_error="record",
        isolate_process=False,
    )
    assert (
        recorded.runs[0].result.measurements["broken"].error.message
        == "intentional measurement failure"
    )
    assert not recorded.summary

    with pytest.raises(RepeatedProfilingError) as error:
        profile_repeated(
            CountingModel(),
            [[1]],
            measurements=[FailingMeasurement()],
            isolate_process=False,
        )
    assert error.value.result.repetitions_attempted == 1
    assert "Repetition 1" in str(error.value)


def test_timeout_terminates_worker_and_keeps_failure_record():
    result = profile_repeated(
        CountingModel(delay_s=10),
        np.ones((1, 1)),
        measurements=[LatencyMeasurement(warmup_iterations=0)],
        worker_timeout_s=2.0,
        on_error="record",
    )

    assert result.repetitions_completed == 0
    assert len(result.failed_runs) == 1
    assert result.failed_runs[0].execution["timeout"] is True
    assert "Repetition 1 timed out" in result.failed_runs[0].error_message


def test_batch_measurement_preserves_batches_and_partial_final_batch(monkeypatch):
    import modeldiet.profiling.batch as batch_module

    class RecordingModel:
        def __init__(self):
            self.lengths = []

        def predict(self, X):
            self.lengths.append(len(X))

    ticks = iter([0.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(batch_module.time, "perf_counter", lambda: next(ticks))
    model = RecordingModel()
    measurement = BatchInferenceMeasurement([4], warmup_iterations=1)
    values = measurement.measure(model, [[1], [2], [3], [4], [5]])

    assert model.lengths == [4, 4, 1]
    assert values["batch_size_4_batch_count"] == 2
    assert values["batch_size_4_sample_count"] == 5
    assert values["batch_size_4_latency_mean_ms"] == pytest.approx(1500.0)
    assert values["batch_size_4_throughput_samples_per_second"] == pytest.approx(5 / 3)


def test_batch_measurement_supports_numpy_tuple_and_dataframe():
    class TypeModel:
        def __init__(self):
            self.types = []

        def predict(self, X):
            self.types.append(type(X))

    for source in (np.ones((3, 2)), ((1, 2), (3, 4), (5, 6))):
        model = TypeModel()
        BatchInferenceMeasurement([8], warmup_iterations=0).measure(model, source)
        assert all(kind is type(source) for kind in model.types)
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    model = TypeModel()
    BatchInferenceMeasurement([1], warmup_iterations=0).measure(model, frame)
    assert all(kind is pd.DataFrame for kind in model.types)


@pytest.mark.parametrize("sizes", [[], [1, 1], [0], [True], [1.5]])
def test_batch_measurement_rejects_invalid_batch_sizes(sizes):
    with pytest.raises(ValueError):
        BatchInferenceMeasurement(sizes)


def test_batch_metrics_have_explicit_batch_level_definitions():
    result = profile_repeated(
        CountingModel(),
        np.ones((3, 1)),
        repetitions=1,
        measurements=[BatchInferenceMeasurement([1, 8], warmup_iterations=0)],
        isolate_process=False,
    )
    batch_measurement = result.runs[0].result.measurements["batch_inference"]
    definition = batch_measurement.metrics["batch_size_1_latency_p95_ms"].definition
    assert definition.unit == "ms"
    assert "batch calls" in definition.description
    assert "not individual samples" in definition.semantics
    assert result.summary["batch_throughput.samples_per_second.batch_size_8"].n == 1


def test_peak_rss_is_sampled_and_monitor_is_cleaned_up_after_success_and_error():
    import threading

    measurement = PeakRSSMeasurement(sampling_interval_ms=1)
    values = measurement.measure(CountingModel(), [[1], [2]])
    assert values["rss_initial_mb"] > 0
    assert values["rss_peak_observed_mb"] >= values["rss_initial_mb"]
    assert values["rss_peak_increment_mb"] >= 0
    assert "sampled" in values["peak_rss_method"]
    assert values["sampling_interval_ms"] == 1
    assert not any(
        thread.name == "modeldiet-peak-rss-sampler" for thread in threading.enumerate()
    )

    class BrokenModel:
        def predict(self, X):
            raise RuntimeError("prediction failed")

    with pytest.raises(RuntimeError, match="prediction failed"):
        measurement.measure(BrokenModel(), [[1]])
    assert not any(
        thread.name == "modeldiet-peak-rss-sampler" for thread in threading.enumerate()
    )


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_peak_rss_sampling_interval_validation(interval):
    with pytest.raises(ValueError, match="positive integer"):
        PeakRSSMeasurement(interval)
