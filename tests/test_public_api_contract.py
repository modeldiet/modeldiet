import os
import pickle
import subprocess
import sys
import textwrap
import warnings


def test_supported_root_imports_and_public_names():
    import modeldiet
    from modeldiet import (
        ProfileResult,
        RepeatedProfileResult,
        RepeatedProfilingError,
        profile,
        profile_repeated,
    )

    assert callable(profile)
    assert callable(profile_repeated)
    assert ProfileResult is modeldiet.ProfileResult
    assert RepeatedProfileResult is modeldiet.RepeatedProfileResult
    assert RepeatedProfilingError is modeldiet.RepeatedProfilingError
    assert modeldiet.__all__ == [
        "profile",
        "profile_repeated",
        "RepeatedProfilingError",
        "ProfileResult",
        "RepeatedProfileResult",
    ]


def test_supported_measurement_imports_and_public_names():
    import modeldiet.profiling as profiling
    from modeldiet.profiling import (
        BaseMeasurement,
        BatchInferenceMeasurement,
        LatencyMeasurement,
        PeakRSSMeasurement,
        ProcessRSSMeasurement,
        SerializationMeasurement,
        TracedAllocationMeasurement,
    )

    assert profiling.__all__ == [
        "BaseMeasurement",
        "BatchInferenceMeasurement",
        "LatencyMeasurement",
        "PeakRSSMeasurement",
        "ProcessRSSMeasurement",
        "SerializationMeasurement",
        "TracedAllocationMeasurement",
    ]
    assert all(
        isinstance(measurement(), BaseMeasurement)
        for measurement in (
            LatencyMeasurement,
            BatchInferenceMeasurement,
            ProcessRSSMeasurement,
            PeakRSSMeasurement,
            SerializationMeasurement,
            TracedAllocationMeasurement,
        )
    )


def test_supported_result_record_imports():
    from modeldiet.result import (
        AggregateMetricStatistics,
        MeasurementError,
        MeasurementResult,
        MetricDefinition,
        MetricValue,
        ProfileResult,
        RepeatedProfileResult,
        RepeatedRunResult,
    )

    assert all(
        isinstance(record, type)
        for record in (
            MetricDefinition,
            MetricValue,
            MeasurementError,
            MeasurementResult,
            ProfileResult,
            AggregateMetricStatistics,
            RepeatedProfileResult,
            RepeatedRunResult,
        )
    )


def test_optional_benchmark_import_path():
    from modeldiet.benchmark import run_repeated_benchmark as new_path
    from modeldiet.profiling.benchmark import run_repeated_benchmark

    assert callable(run_repeated_benchmark)
    assert run_repeated_benchmark is new_path


def test_legacy_model_report_and_estimator_imports_are_compatibility_aliases():
    from modeldiet.benchmark.model import Model as NewModel
    from modeldiet.benchmark.models.lr import LogisticRegressionModel as NewLR
    from modeldiet.benchmark.models.rf import RandomForestModel as NewRF
    from modeldiet.benchmark.models.xgb import XGBoostModel as NewXGB
    from modeldiet.benchmark.report import Report as NewReport
    from modeldiet.model import Model
    from modeldiet.models.lr import LogisticRegressionModel
    from modeldiet.models.rf import RandomForestModel
    from modeldiet.models.xgb import XGBoostModel
    from modeldiet.report import Report

    assert Model is NewModel
    assert LogisticRegressionModel is NewLR
    assert RandomForestModel is NewRF
    assert XGBoostModel is NewXGB
    assert Report is NewReport


def test_legacy_pickle_path_loads_through_compatibility_module():
    from modeldiet.benchmark.models.lr import LogisticRegressionModel

    model_type = LogisticRegressionModel
    original_module = model_type.__module__
    try:
        model_type.__module__ = "modeldiet.models.lr"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_pickle = pickle.dumps(model_type())
    finally:
        model_type.__module__ = original_module

    # This fixture was serialized from the known compatibility class above.
    restored = pickle.loads(legacy_pickle)  # nosec B301
    assert type(restored) is model_type


def test_ordinary_import_has_no_optional_ml_dependencies(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import sys
                import modeldiet

                assert callable(modeldiet.profile)
                assert "sklearn" not in sys.modules
                assert "xgboost" not in sys.modules
                """),
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
