"""Regression tests for visualization data boundaries and public chart behavior."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from modeldiet.visualization.data import (
    metric_number,
    metric_value,
    number,
    validate_results,
)


def sample_mapping(value: float = 2.0) -> dict:
    return {
        "accuracy": 0.9,
        "loading_time": 0.1,
        "latency_observations_ms": [1.0, 2.0, 3.0],
        "measurement_details": {
            "latency": {"mean_ms": value, "p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0},
            "process_memory": {
                "rss_before_inference_mb": 10,
                "rss_after_inference_mb": 11,
                "rss_delta_mb": 1,
            },
            "allocation": {"traced_peak_inference_allocation_mb": 4},
            "serialization": {"serialized_model_size_mb": 0.5},
        },
        "environment": {"python_version": "fixture"},
    }


def test_shared_accessors_cover_aliases_validation_and_missing_values():
    record = sample_mapping()
    assert metric_value(record, "process_rss", "rss_after_inference_mb") == 11
    assert metric_number(record, "latency", "mean_ms") == 2
    assert number(True) is None
    assert number(math.inf) is None
    assert number(float("nan")) is None
    assert number(0) == 0
    assert metric_value({}, "latency", "p99_ms") is None
    with pytest.raises(TypeError, match="mapping"):
        validate_results([])
    with pytest.raises(ValueError, match="non-empty"):
        validate_results({"": {}})


def test_all_public_figure_methods_delegate_and_report_available_metrics(tmp_path):
    pytest.importorskip("plotly")
    from modeldiet.visualization import ModelDietVisualizer

    visualizer = ModelDietVisualizer(
        {"A": sample_mapping(), "B": sample_mapping(1.5)}, theme="dark"
    )
    figures = [
        visualizer.plot_latency_comparison(),
        visualizer.plot_mean_latency(),
        visualizer.plot_latency_distribution(),
        visualizer.plot_latency_distribution(chart_type="box"),
        visualizer.plot_latency_distribution(chart_type="violin"),
        visualizer.plot_memory_comparison(),
        visualizer.plot_python_memory(),
        visualizer.plot_loading_time(),
        visualizer.plot_artifact_size(),
        visualizer.plot_accuracy_latency(),
        visualizer.plot_normalized_comparison(["p99_ms"], baseline="A"),
        visualizer.compare_models("A", "B"),
    ]
    assert all(hasattr(figure, "write_html") for figure in figures)
    assert all(figure.layout.paper_bgcolor == "#111827" for figure in figures)
    assert visualizer.available_metrics["p99_ms"] == ["A", "B"]
    report = visualizer.generate_report(tmp_path / "report.html")
    assert report.exists()


def test_selection_and_chart_arguments_are_validated():
    pytest.importorskip("plotly")
    from modeldiet.visualization import ModelDietVisualizer

    visualizer = ModelDietVisualizer({"A": sample_mapping()})
    with pytest.raises(ValueError, match="duplicate"):
        visualizer.plot_mean_latency(models=["A", "A"])
    with pytest.raises(ValueError, match="Unknown model"):
        visualizer.plot_mean_latency(models=["B"])
    with pytest.raises(ValueError, match="chart_type"):
        visualizer.plot_latency_distribution(chart_type="ridge")
    with pytest.raises(ValueError, match="baseline"):
        visualizer.plot_normalized_comparison(["p99_ms"], baseline="B")
    with pytest.raises(ValueError, match="Unknown metric"):
        visualizer.plot_normalized_comparison(["unknown"], baseline="A")


def test_normalization_keeps_zero_baselines_missing():
    pytest.importorskip("plotly")
    from modeldiet.visualization import ModelDietVisualizer

    zero = sample_mapping()
    zero["measurement_details"]["latency"]["p99_ms"] = 0
    visualizer = ModelDietVisualizer({"baseline": zero, "other": sample_mapping()})
    figure = visualizer.plot_normalized_comparison(["p99_ms"], baseline="baseline")
    assert list(figure.data[0].y) == [None, None]
    assert any("zero" in annotation.text for annotation in figure.layout.annotations)


def test_report_keeps_environment_metadata_per_model(tmp_path):
    pytest.importorskip("plotly")
    from modeldiet.visualization.html_report import write_html_report

    report = write_html_report(
        {
            "<A>": sample_mapping(),
            "B": {**sample_mapping(), "environment": {"python_version": "other"}},
        },
        tmp_path / "report.html",
        experiment_name="<script>safe</script>",
    )
    content = report.read_text(encoding="utf-8")
    assert "&lt;A&gt; Environment Python Version" in content
    assert "B Environment Python Version" in content
    assert "&lt;script&gt;safe&lt;/script&gt;" in content
    assert "<script>safe</script>" not in content


def test_cli_uses_public_facade(monkeypatch, tmp_path, capsys):
    pytest.importorskip("plotly")
    from modeldiet.visualization import cli

    source = tmp_path / "a.json"
    source.write_text(json.dumps({"model": "A", **sample_mapping()}), encoding="utf-8")
    output = tmp_path / "cli.html"
    monkeypatch.setattr(
        "sys.argv", ["modeldiet", "visualize", str(source), "--output", str(output)]
    )
    cli.main()
    assert output.exists()
    assert str(output) in capsys.readouterr().out


def test_desktop_value_accessor_uses_shared_result_accessors():
    from modeldiet.visualization.desktop import _value

    record = sample_mapping()
    assert _value(record, "accuracy") == 0.9
    assert _value(record, "loading") == 0.1
    assert _value(record, "p99") == 3
    assert _value(record, "rss_after") == 11


def test_demo_profiles_selection_without_training_benchmarks(monkeypatch):
    from modeldiet.visualization import demo

    class FakeModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [0] * len(X)

    data = [[float(index), 1.0] for index in range(12)]
    target = [0] * 6 + [1] * 6
    monkeypatch.setattr(demo, "_dataset", lambda _name: (data, target))
    monkeypatch.setattr(demo, "_model", lambda _name: FakeModel())
    values = {
        key: SimpleNamespace(value=1.0)
        for key in (
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "sample_count",
            "rss_before_inference_mb",
            "rss_after_inference_mb",
            "rss_delta_mb",
            "traced_peak_inference_allocation_mb",
            "serialized_model_size_mb",
        )
    }
    measurement = SimpleNamespace(metrics=values)
    measured = SimpleNamespace(
        measurements={
            name: measurement
            for name in (
                "latency",
                "process_rss",
                "python_traced_allocation",
                "pickle_size",
            )
        },
        environment={"python_version": "test"},
        sample_count=9,
        model_class="FakeModel",
        model_module="test",
    )
    monkeypatch.setattr(demo, "profile", lambda *args, **kwargs: measured)
    results = demo._profile_selection("fixture", ["Fake"])
    assert results["Fake"]["sample_count"] == 9
    assert results["Fake"]["model_configuration"]["class"] == "FakeModel"
