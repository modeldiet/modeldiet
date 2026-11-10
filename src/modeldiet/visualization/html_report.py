"""Compose standalone, offline-capable HTML profiling reports."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modeldiet.result import ProfileResult
from modeldiet.visualization import charts
from modeldiet.visualization.data import (
    Results,
    accuracy_value,
    metric_number,
    metric_value,
    validate_results,
)
from modeldiet.visualization.theme import THEMES


def _safe_json(value: Any) -> Any:
    if isinstance(value, ProfileResult):
        return value._json_value()
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _cards(results: Results) -> str:
    metrics = [
        (
            "Best test accuracy",
            accuracy_value,
            True,
            "%",
        ),
        (
            "Lowest P99 latency",
            lambda r: metric_number(r, "latency", "p99_ms"),
            False,
            " ms",
        ),
        (
            "Lowest process RSS",
            lambda r: metric_number(r, "process_rss", "rss_after_inference_mb"),
            False,
            " MiB",
        ),
        (
            "Smallest serialized size",
            lambda r: metric_number(r, "pickle_size", "serialized_model_size_mb"),
            False,
            " MiB",
        ),
    ]
    cards: list[str] = []
    for label, getter, higher, unit in metrics:
        available: list[tuple[str, float]] = []
        for name, record in results.items():
            value = getter(record)
            if value is not None:
                available.append((name, value))
        if not available:
            continue
        winner, value = (max if higher else min)(available, key=lambda item: item[1])
        formatted = f"{value * 100:.2f}" if higher else f"{value:.4g}"
        cards.append(
            f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div><strong>{formatted}{unit}</strong><div class="kpi-model">{html.escape(winner)}</div></div>'
        )
    return (
        "".join(cards)
        or '<p class="empty">No summary metrics are available in these results.</p>'
    )


def _metadata(results: Results) -> str:
    metadata: dict[str, Any] = {}
    for name, record in results.items():
        if isinstance(record, ProfileResult):
            if record.environment:
                metadata.update(
                    {
                        f"{name} environment {key}": value
                        for key, value in record.environment.items()
                    }
                )
            count = metric_value(record, "latency", "sample_count")
            if count is not None:
                metadata[f"{name} inference observations"] = count
            metadata[f"{name} model configuration"] = (
                f"{record.model_module}.{record.model_class}"
            )
        elif isinstance(record, Mapping):
            env = record.get("environment")
            if isinstance(env, Mapping):
                metadata.update(
                    {f"{name} environment {key}": value for key, value in env.items()}
                )
            for key in (
                "sample_count",
                "warmup_count",
                "repetitions",
                "benchmark_configuration",
                "model_configuration",
            ):
                if key in record:
                    metadata[f"{name} {key.replace('_', ' ')}"] = record[key]
    if not metadata:
        return '<p class="empty">Benchmark metadata was not included in the supplied results.</p>'
    rows = []
    for key, value in metadata.items():
        if value is None:
            rendered = "Unavailable"
        elif isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        rows.append(
            f"<tr><th>{html.escape(str(key).replace('_', ' ').title())}</th><td>{html.escape(rendered)}</td></tr>"
        )
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def write_html_report(
    results: Results,
    output_path: str | Path,
    *,
    experiment_name: str | None = None,
    include_json: bool = False,
    theme: str = "light",
) -> Path:
    """Write an offline standalone HTML report from collected results.

    Results may be a mapping of display names to ``ProfileResult`` records or
    the JSON mappings produced by ModelDiet's benchmark worker. The function
    never runs inference or estimates absent measurements.
    """
    if not isinstance(results, Mapping):
        raise TypeError("results must map model names to existing profiling results")
    results = validate_results(results)
    if theme not in {"light", "dark"}:
        raise ValueError("theme must be 'light' or 'dark'")
    path = Path(output_path)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"HTML report output path is a directory: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(
            f"HTML report parent directory does not exist: {path.parent}"
        )
    figures = [
        (
            "Inference performance",
            [
                (
                    "Latency percentiles",
                    charts.create_latency_percentile_chart(results, theme=theme),
                ),
                (
                    "Latency distribution",
                    charts.create_latency_distribution_chart(results, theme=theme),
                ),
            ],
        ),
        (
            "Memory characteristics",
            [
                (
                    "Process RSS",
                    charts.create_rss_comparison_chart(results, theme=theme),
                ),
                (
                    "Python-traced memory",
                    charts.create_python_memory_chart(results, theme=theme),
                ),
            ],
        ),
        (
            "Deployment characteristics",
            [
                (
                    "Model loading time",
                    charts.create_loading_time_chart(results, theme=theme),
                ),
                (
                    "Serialized model size",
                    charts.create_artifact_size_chart(results, theme=theme),
                ),
            ],
        ),
        (
            "Performance trade-offs",
            [
                (
                    "Accuracy and latency",
                    charts.create_accuracy_latency_chart(results, theme=theme),
                )
            ],
        ),
    ]
    plot_blocks: list[str] = []
    include_plotly = True
    colors = THEMES[theme]
    for section, section_figures in figures:
        blocks = []
        for title, figure in section_figures:
            rendered = figure.to_html(
                full_html=False,
                include_plotlyjs=include_plotly,
                config={"responsive": True, "displaylogo": False, "scrollZoom": True},
                default_height="360px",
                default_width="100%",
            )
            include_plotly = False
            blocks.append(
                f'<article class="chart"><h3>{html.escape(title)}</h3>{rendered}</article>'
            )
        plot_blocks.append(
            f'<section><h2>{html.escape(section)}</h2><div class="chart-grid">{"".join(blocks)}</div></section>'
        )
    generated = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    title = experiment_name or "Profiling report"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · ModelDiet</title><style>
:root{{--bg:{colors["background"]};--surface:{colors["surface"]};--text:{colors["text"]};--muted:{colors["muted"]};--border:{colors["grid"]};--accent:#60a5fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:auto;padding:36px 28px 64px}}header{{padding:26px 30px;background:var(--surface);border:1px solid var(--border);border-radius:12px}}
.brand{{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}}h1{{margin:8px 0 4px;font-size:30px}}.sub{{color:var(--muted)}}section{{margin-top:30px}}h2{{font-size:20px;margin:0 0 14px}}h3{{font-size:15px;margin:0;padding:15px 18px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px}}.kpi,.chart{{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}}
.kpi{{padding:17px 19px}}.kpi-label,.kpi-model{{color:var(--muted);font-size:12px}}.kpi strong{{display:block;font-size:24px;margin:4px 0}}.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:14px}}.chart{{min-width:0}}
table{{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}}th,td{{padding:10px 14px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}}th{{width:28%;font-weight:600}}td{{overflow-wrap:anywhere}}.empty{{color:var(--muted);background:var(--surface);padding:20px;border:1px solid var(--border);border-radius:10px}}
@media(max-width:600px){{main{{padding:18px 12px 40px}}header{{padding:20px}}h1{{font-size:24px}}}}
</style></head><body><main><header><div class="brand">ModelDiet · Interactive profile</div><h1>{html.escape(title)}</h1><div class="sub">Generated {html.escape(generated)} · {len(results)} evaluated model{"s" if len(results) != 1 else ""}</div></header>
<section><h2>Summary metrics</h2><div class="kpis">{_cards(results)}</div></section>
{"".join(plot_blocks)}
<section><h2>Benchmark metadata</h2>{_metadata(results)}</section>
<section><h2>How to read these measurements</h2><div class="kpis">
<div class="kpi"><strong>Latency percentiles</strong><div class="kpi-model">P50 is the median timed prediction. P95 and P99 show slower tail observations. They describe this local run and are affected by machine load and sample count.</div></div>
<div class="kpi"><strong>Process RSS</strong><div class="kpi-model">RSS is memory resident in the whole Python process. Before and after are snapshots; the change is not peak RSS and does not isolate the model.</div></div>
<div class="kpi"><strong>Python-traced memory</strong><div class="kpi-model">tracemalloc observes Python-tracked allocations during a separate prediction pass. Native library allocations may not be included.</div></div>
<div class="kpi"><strong>Serialized size</strong><div class="kpi-model">This is the measured pickle representation size. It is not the model’s in-memory footprint or necessarily its deployment artifact.</div></div>
</div></section>
</main></body></html>"""
    try:
        path.write_text(document, encoding="utf-8")
        if include_json:
            path.with_name("results.json").write_text(
                json.dumps(
                    _safe_json(results), indent=2, ensure_ascii=False, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
    except OSError as exc:
        raise OSError(
            f"Could not write ModelDiet HTML report to {path}: {exc}"
        ) from exc
    return path
