"""Smoke-test the public ModelDiet visualization API with synthetic fixtures.

Install with ``python -m pip install -e ".[viz]"`` and run
``python examples/test_visualizer_api.py``. The numbers below are test data,
not benchmark results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modeldiet.visualization import ModelDietVisualizer


def result(
    *,
    accuracy: float,
    mean: float,
    p50: float,
    p95: float,
    p99: float,
    rss_before: float,
    rss_after: float,
    traced_peak: float,
    loading: float,
    size: float,
) -> dict[str, Any]:
    """Build one representative benchmark-style fixture."""
    return {
        "accuracy": accuracy,
        "loading_time": loading,
        "latency_observations_ms": [p50 * 0.7, p50, mean, p95, p99],
        "sample_count": 5,
        "environment": {
            "python_version": "synthetic test fixture",
            "platform": "local test",
        },
        "measurement_details": {
            "latency": {
                "mean_ms": mean,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "sample_count": 5,
            },
            "process_memory": {
                "rss_before_inference_mb": rss_before,
                "rss_after_inference_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
            },
            "allocation": {
                "traced_peak_inference_allocation_mb": traced_peak,
            },
            "serialization": {
                "serialized_model_size_mb": size,
            },
        },
    }


def main() -> None:
    output_dir = Path("visualizer-test-output")
    output_dir.mkdir(exist_ok=True)
    fixture_path = output_dir / "synthetic-results.json"
    report_path = output_dir / "visualizer-report.html"

    results = {
        "Logistic Regression": result(
            accuracy=0.94,
            mean=0.42,
            p50=0.38,
            p95=0.61,
            p99=0.75,
            rss_before=120.0,
            rss_after=121.2,
            traced_peak=2.1,
            loading=0.018,
            size=0.08,
        ),
        "Random Forest": result(
            accuracy=0.96,
            mean=1.15,
            p50=0.98,
            p95=1.82,
            p99=2.20,
            rss_before=145.0,
            rss_after=148.0,
            traced_peak=4.7,
            loading=0.044,
            size=3.2,
        ),
        "Gradient Boosting": result(
            accuracy=0.95,
            mean=0.81,
            p50=0.73,
            p95=1.23,
            p99=1.61,
            rss_before=132.0,
            rss_after=133.5,
            traced_peak=3.4,
            loading=0.031,
            size=0.9,
        ),
    }
    fixture_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    visualizer = ModelDietVisualizer(results)

    # Individual Plotly figure methods, including filtering and raw samples.
    figures = {
        "latency percentiles": visualizer.plot_latency_comparison(),
        "selected percentiles": visualizer.plot_latency_comparison(
            models=["Logistic Regression", "Random Forest"],
            percentiles=["p50_ms", "p99_ms"],
        ),
        "mean latency": visualizer.plot_mean_latency(),
        "latency ECDF": visualizer.plot_latency_distribution(),
        "latency box plot": visualizer.plot_latency_distribution(chart_type="box"),
        "latency violin plot": visualizer.plot_latency_distribution(
            chart_type="violin"
        ),
        "process RSS": visualizer.plot_memory_comparison(),
        "Python-traced memory": visualizer.plot_python_memory(),
        "loading time": visualizer.plot_loading_time(),
        "serialized size": visualizer.plot_artifact_size(),
        "accuracy vs latency": visualizer.plot_accuracy_latency(),
        "normalized metrics": visualizer.plot_normalized_comparison(
            ["p99_ms", "serialized_size_mib"], baseline="Logistic Regression"
        ),
        "pairwise comparison": visualizer.compare_models(
            "Logistic Regression", "Random Forest"
        ),
    }
    for title, figure in figures.items():
        assert figure is not None and hasattr(figure, "to_html"), title
        print(f"PASS: {title} ({len(figure.data)} Plotly trace(s))")

    visualizer.generate_report(
        report_path,
        experiment_name="Synthetic visualization API test",
        include_json=True,
    )
    report_html = report_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in report_html
    assert "height:360px" in report_html
    assert "Performance trade-offs" in report_html
    print(f"PASS: standalone HTML report: {report_path.resolve()}")

    loaded = ModelDietVisualizer.from_file(fixture_path)
    assert len(loaded.results) == 3
    dark_report = output_dir / "visualizer-report-dark.html"
    ModelDietVisualizer(loaded.results, theme="dark").generate_report(dark_report)
    assert "#111827" in dark_report.read_text(encoding="utf-8")
    print(f"PASS: JSON loading and dark theme: {dark_report.resolve()}")
    print("Done. Open the report HTML files above in a browser to inspect them.")


if __name__ == "__main__":
    main()
