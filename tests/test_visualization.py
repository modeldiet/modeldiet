import importlib.util
import json
import subprocess
import sys

import pytest

from modeldiet.result import (
    MeasurementResult,
    MetricDefinition,
    MetricValue,
    ProfileResult,
)

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def _profile(model, p50, p95, p99, rss=20.0, traced=1.0, size=0.5):
    def measurement(name, values):
        metrics = {}
        for key, value, unit in values:
            metrics[key] = MetricValue(
                MetricDefinition(f"{name}.{key}", unit, key, "fixture"), value
            )
        return MeasurementResult(name, metrics)

    return ProfileResult(
        schema_version=1,
        measurements={
            "latency": measurement(
                "latency",
                [
                    ("p50_ms", p50, "ms"),
                    ("p95_ms", p95, "ms"),
                    ("p99_ms", p99, "ms"),
                    ("sample_count", 100, "count"),
                ],
            ),
            "process_rss": measurement(
                "process_rss",
                [
                    ("rss_before_inference_mb", rss, "MiB"),
                    ("rss_after_inference_mb", rss + 1, "MiB"),
                    ("rss_delta_mb", 1, "MiB"),
                ],
            ),
            "python_traced_allocation": measurement(
                "python_traced_allocation",
                [("traced_peak_inference_allocation_mb", traced, "MiB")],
            ),
            "pickle_size": measurement(
                "pickle_size", [("serialized_model_size_mb", size, "MiB")]
            ),
        },
        model_class=model,
        model_module="fixture",
        input_type="ndarray",
        sample_count=100,
        feature_count=4,
        environment={"python_version": "3.test", "processor": "fixture"},
    )


def test_core_package_and_visualization_namespace_do_not_import_plotly():
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, modeldiet, modeldiet.visualization; assert 'plotly' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.skipif(HAS_PLOTLY, reason="exercises missing optional dependency")
def test_visualization_error_explains_install_extra():
    from modeldiet.visualization import create_latency_percentile_chart

    with pytest.raises(ImportError, match=r"modeldiet\[viz\]"):
        create_latency_percentile_chart({})


@pytest.mark.skipif(not HAS_PLOTLY, reason="requires optional modeldiet[viz] extra")
def test_percentiles_and_missing_values_are_not_zero():
    from modeldiet.visualization import create_latency_percentile_chart

    fig = create_latency_percentile_chart(
        {"A": _profile("A", 1.2, 2.5, 4.0), "B": _profile("B", 0.8, 2, 3)}
    )
    assert list(fig.data[0].x) == ["A", "B"]
    assert list(fig.data[0].y) == [1.2, 0.8]
    assert fig.data[0].name == "P50"
    assert "Latency (ms)" in str(fig.layout.yaxis.title.text)


@pytest.mark.skipif(not HAS_PLOTLY, reason="requires optional modeldiet[viz] extra")
def test_distribution_uses_raw_samples_or_reported_percentile_fallback():
    from modeldiet.visualization import create_latency_distribution_chart

    result = _profile("A", 1, 2, 3)
    fig = create_latency_distribution_chart({"A": result})
    assert list(fig.data[0].x) == [1, 2, 3]
    assert fig.data[0].name == "Reported percentiles · A"
    assert any(
        "reported percentiles only" in item.text for item in fig.layout.annotations
    )
    fig = create_latency_distribution_chart(
        {"A": {"latency_observations_ms": [3, 1, 2]}}
    )
    assert list(fig.data[0].x) == [1, 2, 3]


@pytest.mark.skipif(not HAS_PLOTLY, reason="requires optional modeldiet[viz] extra")
def test_memory_charts_keep_process_rss_and_traced_memory_separate():
    from modeldiet.visualization import (
        create_python_memory_chart,
        create_rss_comparison_chart,
    )

    result = {"A": _profile("A", 1, 2, 3)}
    rss, traced = (
        create_rss_comparison_chart(result),
        create_python_memory_chart(result),
    )
    assert [trace.name for trace in rss.data] == [
        "Before inference",
        "After inference",
        "RSS change",
    ]
    assert list(rss.data[0].x) == ["A"]
    assert list(traced.data[0].y) == [1.0]


@pytest.mark.skipif(not HAS_PLOTLY, reason="requires optional modeldiet[viz] extra")
def test_single_empty_and_html_report(tmp_path):
    from modeldiet.visualization import (
        create_latency_percentile_chart,
        write_html_report,
    )

    assert list(
        create_latency_percentile_chart({"Solo": _profile("Solo", 1, 2, 3)}).data[0].x
    ) == ["Solo"]
    empty = create_latency_percentile_chart({})
    assert "No model results" in empty.layout.annotations[0].text
    result_path = write_html_report(
        {"Solo": _profile("Solo", 1, 2, 3)}, tmp_path / "report.html", include_json=True
    )
    content = result_path.read_text()
    assert 'role="tablist"' in content and ">Latency</button>" in content
    assert 'id="panel-memory"' in content and 'id="panel-tradeoffs"' in content
    assert "Model details" in content
    assert 'class="detail-toggle"' in content
    assert 'id="single-model"' in content
    assert 'value="compare"' in content
    assert 'id="tab-metrics"' in content
    assert content.count('class="metric-card"') == 11
    assert "Plotly.newPlot" in content
    assert 'src="https://cdn.plot.ly' not in content
    assert "python version" in content.lower()
    assert (
        json.loads((tmp_path / "results.json").read_text())["Solo"]["model_class"]
        == "Solo"
    )


@pytest.mark.skipif(not HAS_PLOTLY, reason="requires optional modeldiet[viz] extra")
def test_public_visualizer_comparison_selection_themes_and_loading(tmp_path):
    from modeldiet.visualization import ModelDietVisualizer

    payload = {
        "A": {
            "accuracy": 0.90,
            "loading_time": 0.2,
            "measurement_details": {
                "latency": {
                    "mean_ms": 1.0,
                    "p50_ms": 0.8,
                    "p95_ms": 1.5,
                    "p99_ms": 2.0,
                },
                "process_memory": {
                    "rss_after_inference_mb": 50,
                    "rss_before_inference_mb": 49,
                    "rss_delta_mb": 1,
                },
                "allocation": {"traced_peak_inference_allocation_mb": 3},
                "serialization": {"serialized_model_size_mb": 0.4},
            },
        },
        "B": {
            "accuracy": 0.92,
            "loading_time": 0.1,
            "measurement_details": {
                "latency": {
                    "mean_ms": 0.5,
                    "p50_ms": 0.4,
                    "p95_ms": 1.0,
                    "p99_ms": 1.5,
                },
                "process_memory": {
                    "rss_after_inference_mb": 45,
                    "rss_before_inference_mb": 44,
                    "rss_delta_mb": 1,
                },
                "allocation": {"traced_peak_inference_allocation_mb": 2},
                "serialization": {"serialized_model_size_mb": 0.3},
            },
        },
    }
    visualizer = ModelDietVisualizer(payload, theme="dark")
    assert visualizer.plot_latency_comparison(
        models=["A"], percentiles=["p99_ms"]
    ).data[0].x == ("A",)
    assert visualizer.plot_mean_latency().layout.paper_bgcolor == "#111827"
    assert visualizer.compare_models("A", "B").data
    normalized = visualizer.plot_normalized_comparison(["p99_ms"], baseline="A")
    assert list(normalized.data[0].y) == [100, 75]
    report = visualizer.generate_report(tmp_path / "dark.html")
    assert "#111827" in report.read_text()

    source = tmp_path / "single.json"
    source.write_text(json.dumps({"model": "Solo", **payload["A"]}))
    loaded = ModelDietVisualizer.from_file(source)
    assert list(loaded.results) == ["Solo"]
