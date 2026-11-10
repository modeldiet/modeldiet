"""Optional interactive reports built from existing ModelDiet results."""

from modeldiet.visualization.api import ModelDietVisualizer
from modeldiet.visualization.charts import (
    create_accuracy_latency_chart,
    create_artifact_size_chart,
    create_latency_distribution_chart,
    create_latency_percentile_chart,
    create_loading_time_chart,
    create_python_memory_chart,
    create_rss_comparison_chart,
)
from modeldiet.visualization.html_report import write_html_report

__all__ = [
    "ModelDietVisualizer",
    "create_accuracy_latency_chart",
    "create_artifact_size_chart",
    "create_latency_distribution_chart",
    "create_latency_percentile_chart",
    "create_loading_time_chart",
    "create_python_memory_chart",
    "create_rss_comparison_chart",
    "write_html_report",
]
