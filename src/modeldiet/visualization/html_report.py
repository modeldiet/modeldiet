"""Compose standalone, offline-capable HTML profiling reports."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modeldiet.result import ProfileResult, RepeatedProfileResult
from modeldiet.visualization import charts
from modeldiet.visualization.comparison import METRICS, value_for
from modeldiet.visualization.data import (
    Results,
    accuracy_value,
    metric_number,
    metric_value,
    validate_results,
)
from modeldiet.visualization.theme import THEMES

TAB_SCRIPT = """
<script>
(() => {
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
  const viewMode = document.getElementById('view-mode');
  const singleModel = document.getElementById('single-model');
  const singlePicker = document.getElementById('single-picker');
  const comparePicker = document.getElementById('compare-picker');
  const compareOptions = Array.from(document.querySelectorAll('.compare-model'));
  const selectionStatus = document.getElementById('selection-status');
  let activeModels = [singleModel.value];

  function modelForTrace(name) {
    if (name.startsWith('Observed ECDF · ')) return name.slice('Observed ECDF · '.length);
    if (name.startsWith('Reported percentiles · ')) return name.slice('Reported percentiles · '.length);
    return name;
  }

  function filterCharts(models) {
    document.querySelectorAll('.js-plotly-plot').forEach((plot) => {
      const original = plot.__modeldietOriginal || JSON.parse(JSON.stringify(plot.data));
      plot.__modeldietOriginal = original;
      const traces = [];
      original.forEach((source) => {
        const trace = JSON.parse(JSON.stringify(source));
        if (trace.type === 'bar' && trace.orientation === 'h' && Array.isArray(trace.y)) {
          const indices = trace.y.map((model, index) => models.includes(model) ? index : -1).filter((index) => index >= 0);
          trace.y = indices.map((index) => source.y[index]);
          trace.x = indices.map((index) => source.x[index]);
          if (Array.isArray(source.customdata)) trace.customdata = indices.map((index) => source.customdata[index]);
          if (trace.y.length) traces.push(trace);
        } else if (trace.type === 'bar' && Array.isArray(trace.x) && trace.x.some((model) => typeof model === 'string' && models.includes(model))) {
          const indices = trace.x.map((model, index) => models.includes(model) ? index : -1).filter((index) => index >= 0);
          trace.x = indices.map((index) => source.x[index]);
          trace.y = indices.map((index) => source.y[index]);
          if (Array.isArray(source.customdata)) trace.customdata = indices.map((index) => source.customdata[index]);
          if (trace.x.length) traces.push(trace);
        } else if (['scatter', 'box', 'violin'].includes(trace.type)) {
          if (models.includes(modelForTrace(trace.name || ''))) traces.push(trace);
        } else {
          traces.push(trace);
        }
      });
      if (window.Plotly) Plotly.react(plot, traces, plot.layout, {responsive: true, displaylogo: false, scrollZoom: true});
    });
  }

  function applyModelSelection() {
    const comparing = viewMode.value === 'compare';
    singlePicker.hidden = comparing;
    comparePicker.hidden = !comparing;
    if (comparing) {
      activeModels = compareOptions.filter((input) => input.checked).map((input) => input.value);
      selectionStatus.textContent = `Comparing ${activeModels.length} models. Select two or three.`;
    } else {
      activeModels = [singleModel.value];
      selectionStatus.textContent = 'Showing one model across the report.';
    }
    document.querySelectorAll('[data-model]').forEach((item) => {
      item.hidden = !activeModels.includes(item.dataset.model);
    });
    document.querySelectorAll('.kpi').forEach((card) => {
      const readings = card.querySelectorAll('.kpi-reading');
      card.classList.toggle('no-readings', readings.length > 0 && !Array.from(readings).some((item) => !item.hidden));
    });
    document.querySelectorAll('.metric-card').forEach((card) => {
      const readings = card.querySelectorAll('.metric-reading');
      card.classList.toggle('no-readings', readings.length > 0 && !Array.from(readings).some((item) => !item.hidden));
    });
    filterCharts(activeModels);
  }

  function activate(tab, moveFocus = false) {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.id !== tab.getAttribute('aria-controls');
    });
    if (moveFocus) tab.focus();
    requestAnimationFrame(() => {
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      if (window.Plotly && panel) {
        panel.querySelectorAll('.js-plotly-plot').forEach((plot) => Plotly.Plots.resize(plot));
      }
    });
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', (event) => {
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault();
      activate(tabs[next], true);
    });
  });
  singleModel.addEventListener('change', applyModelSelection);
  viewMode.addEventListener('change', applyModelSelection);
  compareOptions.forEach((input) => input.addEventListener('change', () => {
    const selected = compareOptions.filter((item) => item.checked);
    if (selected.length > 3) {
      input.checked = false;
      selectionStatus.textContent = 'Choose no more than three models to compare.';
      return;
    }
    if (selected.length < 2) {
      input.checked = true;
      selectionStatus.textContent = 'Choose two or three models to compare.';
      return;
    }
    applyModelSelection();
  }));
  document.querySelectorAll('.detail-toggle, .metric-toggle').forEach((button) => {
    button.addEventListener('click', () => {
      const body = document.getElementById(button.getAttribute('aria-controls'));
      const opening = button.getAttribute('aria-expanded') !== 'true';
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      button.setAttribute('aria-expanded', String(opening));
      body.setAttribute('aria-hidden', String(!opening));
      body.inert = !opening;
      if (opening) body.style.height = '0px';
      body.hidden = false;
      const height = body.scrollHeight;
      if (reducedMotion) {
        body.hidden = !opening;
        body.style.height = opening ? 'auto' : '0px';
        return;
      }
      const frames = opening
        ? [{ height: '0px', opacity: 0 }, { height: `${height}px`, opacity: 1 }]
        : [{ height: `${height}px`, opacity: 1 }, { height: '0px', opacity: 0 }];
      const animation = body.animate(frames, { duration: 260, easing: 'ease-out' });
      animation.onfinish = () => {
        body.hidden = !opening;
        body.style.height = opening ? 'auto' : '0px';
      };
    });
  });
  applyModelSelection();
  activate(tabs[0]);
})();
</script>
"""


def _safe_json(value: Any) -> Any:
    if isinstance(value, ProfileResult):
        return value._json_value()
    if isinstance(value, RepeatedProfileResult):
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
        ("Test accuracy", accuracy_value, "%"),
        ("P99 latency", lambda r: metric_number(r, "latency", "p99_ms"), " ms"),
        (
            "Process RSS after inference",
            lambda r: metric_number(r, "process_rss", "rss_after_inference_mb"),
            " MiB",
        ),
        (
            "Serialized model size",
            lambda r: metric_number(r, "pickle_size", "serialized_model_size_mb"),
            " MiB",
        ),
    ]
    cards: list[str] = []
    for label, getter, unit in metrics:
        readings = []
        for name, record in results.items():
            value = getter(record)
            formatted = (
                "Unavailable"
                if value is None
                else f"{value * 100:.2f}{unit}"
                if unit == "%"
                else f"{value:.4g}{unit}"
            )
            readings.append(
                f'<div class="kpi-reading" data-model="{html.escape(name, quote=True)}"><strong>{html.escape(formatted)}</strong><span>{html.escape(name)}</span></div>'
            )
        cards.append(
            f'<article class="kpi"><div class="kpi-label">{html.escape(label)}</div><div class="kpi-readings">{"".join(readings)}</div></article>'
        )
    return "".join(cards)


METRIC_GUIDE = {
    "accuracy": (
        "Share of evaluated predictions that were correct for this classification task.",
        "Higher is usually better. Compare on the same dataset and split; accuracy can hide poor results on rare classes.",
        "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html",
        "scikit-learn: accuracy score",
    ),
    "mean_latency_ms": (
        "Average time for one prediction in this profile.",
        "Lower is usually better for responsiveness. Results depend on hardware, input size, and machine load.",
        "https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm",
        "NIST: percentiles and ordered measurements",
    ),
    "p50_ms": (
        "Median prediction time: half of timed predictions were at or below this value.",
        "Lower is usually better for typical response time.",
        "https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm",
        "NIST: percentile definitions",
    ),
    "p95_ms": (
        "95th-percentile prediction time: about 95% of timed predictions were at or below this value.",
        "Lower is usually better for the slow tail experienced by most requests.",
        "https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm",
        "NIST: percentile definitions",
    ),
    "p99_ms": (
        "99th-percentile prediction time: about 99% of timed predictions were at or below this value.",
        "Lower is usually better when limiting rare, slow responses. Estimates are less stable with few samples.",
        "https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm",
        "NIST: percentile definitions",
    ),
    "rss_before_mib": (
        "Whole-process resident memory snapshot before the measured prediction pass.",
        "Lower means the Python process was using less physical memory. RSS includes the rest of the process, not only the model.",
        "https://psutil.readthedocs.io/stable/#psutil.Process.memory_info",
        "psutil: process memory information",
    ),
    "rss_after_mib": (
        "Whole-process resident memory snapshot after the measured prediction pass.",
        "Lower means less process memory at that moment. It is not a peak and does not isolate model memory.",
        "https://psutil.readthedocs.io/stable/#psutil.Process.memory_info",
        "psutil: process memory information",
    ),
    "rss_delta_mib": (
        "Change between the before and after process RSS snapshots.",
        "A smaller increase is generally preferable. A negative change means RSS fell during this interval; it is not proof of memory reclaimed by the model.",
        "https://psutil.readthedocs.io/stable/#psutil.Process.memory_info",
        "psutil: process memory information",
    ),
    "python_traced_peak_mib": (
        "Peak Python-tracked allocation during the profiling pass.",
        "Lower generally means fewer Python-managed allocations. Native library allocations may not be included.",
        "https://docs.python.org/3/library/tracemalloc.html",
        "Python: tracemalloc",
    ),
    "loading_time_s": (
        "Time measured to load or deserialize the model artifact.",
        "Lower usually means quicker startup. It can vary with storage, cache state, and the loading method.",
        "https://docs.python.org/3/library/pickle.html",
        "Python: pickle serialization",
    ),
    "serialized_size_mib": (
        "Size of the model's measured pickle representation.",
        "Lower reduces storage and transfer requirements. This is not in-memory size and may differ from your deployment format.",
        "https://docs.python.org/3/library/pickle.html",
        "Python: pickle serialization",
    ),
}


def _metric_cards(results: Results) -> str:
    cards = []
    for metric, (_, _, label) in METRICS.items():
        meaning, direction, url, reference = METRIC_GUIDE[metric]
        readings = []
        for model, record in results.items():
            value = value_for(record, metric)
            if value is None:
                rendered = "Unavailable"
            elif metric == "accuracy":
                rendered = f"{value:.2%}"
            elif metric == "loading_time_s":
                rendered = f"{value:.4g} s"
            else:
                rendered = f"{value:.4g}"
            readings.append(
                f'<span class="metric-reading" data-model="{html.escape(model, quote=True)}"><span>{html.escape(model)}</span><strong>{html.escape(rendered)}</strong></span>'
            )
        cards.append(
            f'<article class="metric-card" data-metric="{metric}">'
            f'<button class="metric-toggle" type="button" aria-expanded="false" aria-controls="metric-info-{metric}">'
            f'<span class="metric-main"><strong>{html.escape(label)}</strong><small>{html.escape(meaning)}</small><em>{html.escape(direction)}</em></span>'
            f'<span class="metric-values">{"".join(readings)}</span><span class="metric-chevron" aria-hidden="true">⌄</span></button>'
            f'<div class="metric-info" id="metric-info-{metric}" hidden inert aria-hidden="true">'
            f'<p>{html.escape(direction)}</p><a href="{url}" target="_blank" rel="noopener noreferrer">Learn more: {html.escape(reference)} ↗</a></div></article>'
        )
    return "".join(cards)


def _model_selector(results: Results) -> str:
    names = list(results)
    single_options = "".join(
        f'<option value="{html.escape(name, quote=True)}">{html.escape(name)}</option>'
        for name in names
    )
    compare_options = "".join(
        f'<label class="model-option"><input class="compare-model" type="checkbox" value="{html.escape(name, quote=True)}"{" checked" if index < 3 else ""}><span>{html.escape(name)}</span></label>'
        for index, name in enumerate(names)
    )
    compare_disabled = " disabled" if len(names) < 2 else ""
    return (
        '<div class="model-controls">'
        '<label class="control-field">View mode'
        f'<select id="view-mode"><option value="single">One model</option><option value="compare"{compare_disabled}>Compare 2–3 models</option></select></label>'
        '<div id="single-picker" class="control-field"><label for="single-model">Model</label>'
        f'<select id="single-model">{single_options}</select></div>'
        '<div id="compare-picker" class="control-field compare-picker" hidden>'
        '<span class="control-label">Models</span><details class="compare-menu">'
        "<summary>Choose up to three models</summary>"
        f"<fieldset><legend>Select two or three</legend>{compare_options}</fieldset></details></div>"
        '<span id="selection-status" class="selection-status" role="status" aria-live="polite"></span>'
        "</div>"
    )


def _metadata(results: Results) -> str:
    explanations = {
        "sample count": "How many prediction calls were timed.",
        "inference observations": "How many prediction calls were timed.",
        "warmup count": "Predictions run before timing to warm up the model.",
        "repetitions": "How many benchmark runs were repeated.",
        "model configuration": "The model type and configuration used for this result.",
        "benchmark configuration": "Settings used to run this benchmark.",
        "environment · python version": "The Python interpreter version used for the run.",
        "environment · platform": "The operating system and platform used for the run.",
        "environment · cpu count logical": "The number of logical CPU threads reported by the machine.",
        "environment · total ram gb": "The machine's reported installed memory, in gigabytes.",
    }
    cards = []
    for index, (name, record) in enumerate(results.items()):
        items: list[tuple[str, Any]] = []
        if isinstance(record, ProfileResult):
            if record.environment:
                items.extend(
                    (f"Environment · {key.replace('_', ' ')}", value)
                    for key, value in record.environment.items()
                )
            count = metric_value(record, "latency", "sample_count")
            if count is not None:
                items.append(("Inference observations", count))
            items.append(
                ("Model configuration", f"{record.model_module}.{record.model_class}")
            )
        elif isinstance(record, RepeatedProfileResult):
            if record.environment:
                items.extend(
                    (f"Environment · {key.replace('_', ' ')}", value)
                    for key, value in record.environment.items()
                )
            items.extend(
                [
                    (
                        "Repetitions",
                        {
                            "requested": record.repetitions_requested,
                            "completed": record.repetitions_completed,
                            "failed": len(record.failed_runs),
                        },
                    ),
                    (
                        "Model configuration",
                        f"{record.model_module}.{record.model_class}",
                    ),
                    ("Workload configuration", record.workload),
                    (
                        "Run-level summary",
                        {
                            identifier: {
                                "mean": stats.mean,
                                "std_dev": stats.std_dev,
                                "coefficient_of_variation_percent": stats.coefficient_of_variation,
                                "n": stats.n,
                                "unit": stats.definition.unit,
                            }
                            for identifier, stats in record.summary.items()
                        },
                    ),
                    (
                        "Failed repetitions",
                        [
                            {
                                "repetition_index": run.repetition_index,
                                "error_type": run.error_type,
                                "error_message": run.error_message,
                            }
                            for run in record.failed_runs
                        ],
                    ),
                ]
            )
        elif isinstance(record, Mapping):
            env = record.get("environment")
            if isinstance(env, Mapping):
                items.extend(
                    (f"Environment · {key.replace('_', ' ')}", value)
                    for key, value in env.items()
                )
            for key in (
                "sample_count",
                "warmup_count",
                "repetitions",
                "benchmark_configuration",
                "model_configuration",
            ):
                if key in record:
                    items.append((key.replace("_", " ").title(), record[key]))
        if not items:
            continue

        detail_items = []
        for label, value in items:
            if value is None:
                rendered = "Unavailable"
            elif isinstance(value, (dict, list, tuple)):
                rendered = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, indent=2
                )
            else:
                rendered = str(value)
            help_text = explanations.get(
                label.lower(), "Context recorded alongside this model's measurements."
            )
            detail_items.append(
                '<div class="detail-item">'
                f"<dt>{html.escape(label.title())}</dt>"
                f"<dd>{html.escape(rendered)}</dd>"
                f"<p>{html.escape(help_text)}</p></div>"
            )
        observation_count = metric_value(record, "latency", "sample_count")
        subtitle = (
            f"{observation_count} timed predictions"
            if observation_count is not None
            else f"{len(items)} recorded details"
        )
        body_id = f"model-details-{index}"
        cards.append(
            f'<article class="model-detail-card" data-model="{html.escape(name, quote=True)}">'
            '<div class="model-detail-heading">'
            f"<div><h3>{html.escape(name)}</h3><span>{html.escape(subtitle)}</span></div>"
            f'<button class="detail-toggle" type="button" aria-expanded="false" aria-controls="{body_id}">Explore details <span aria-hidden="true">⌄</span></button>'
            "</div>"
            f'<div class="detail-body" id="{body_id}" hidden inert aria-hidden="true"><dl class="detail-grid">{"".join(detail_items)}</dl></div>'
            "</article>"
        )
    return (
        "".join(cards)
        or '<p class="empty">No extra benchmark details were included with these results.</p>'
    )


def write_html_report(
    results: Results,
    output_path: str | Path,
    *,
    experiment_name: str | None = None,
    include_json: bool = False,
    theme: str = "light",
) -> Path:
    """Write an offline standalone tabbed HTML dashboard from collected results.

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
            "latency",
            "Latency",
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
            "memory",
            "Memory",
            [
                (
                    "Process RSS",
                    charts.create_rss_comparison_chart(results, theme=theme),
                ),
                (
                    "Python-traced memory",
                    charts.create_python_memory_chart(results, theme=theme),
                ),
                (
                    "Sampled peak process RSS",
                    charts.create_peak_rss_comparison_chart(results, theme=theme),
                ),
            ],
        ),
        (
            "deployment",
            "Deployment",
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
            "tradeoffs",
            "Trade-offs",
            [
                (
                    "Accuracy and latency",
                    charts.create_accuracy_latency_chart(results, theme=theme),
                )
            ],
        ),
    ]
    if any(isinstance(record, RepeatedProfileResult) for record in results.values()):
        figures.append(
            (
                "repeated",
                "Repeated runs",
                [
                    (
                        "Run-to-run P99 variability",
                        charts.create_repetition_variability_chart(
                            results, "latency.p99", theme=theme
                        ),
                    ),
                    (
                        "Throughput by batch size",
                        charts.create_batch_throughput_chart(results, theme=theme),
                    ),
                    (
                        "Batch-call latency by batch size",
                        charts.create_batch_latency_chart(results, theme=theme),
                    ),
                ],
            )
        )
    plot_blocks: list[str] = []
    include_plotly = True
    colors = THEMES[theme]
    for tab_id, _tab_label, section_figures in figures:
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
            blocks.append(f'<article class="chart">{rendered}</article>')
        plot_blocks.append(
            f'<section class="tab-panel" id="panel-{tab_id}" role="tabpanel" aria-labelledby="tab-{tab_id}" hidden><div class="chart-grid">{"".join(blocks)}</div></section>'
        )
    generated = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    title = experiment_name or "Profiling report"
    tab_labels = [
        ("overview", "Overview"),
        ("metrics", "Metrics"),
        *[(tab_id, label) for tab_id, label, _ in figures],
        ("details", "Model details"),
    ]
    tab_buttons = "".join(
        f'<button class="tab" id="tab-{tab_id}" type="button" role="tab" aria-controls="panel-{tab_id}" aria-selected="{str(index == 0).lower()}" tabindex="{0 if index == 0 else -1}">{html.escape(label)}</button>'
        for index, (tab_id, label) in enumerate(tab_labels)
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · ModelDiet</title><style>
:root{{--bg:{colors["background"]};--surface:{colors["surface"]};--text:{colors["text"]};--muted:{colors["muted"]};--border:{colors["grid"]};--accent:#60a5fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:auto;padding:22px 24px 36px}}header{{padding:18px 22px;background:var(--surface);border:1px solid var(--border);border-radius:14px;display:flex;align-items:center;justify-content:space-between;gap:20px}}
.brand{{color:var(--accent);font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}}h1{{margin:4px 0 2px;font-size:25px;line-height:1.2}}.sub{{color:var(--muted);font-size:12px}}.header-meta{{text-align:right;white-space:nowrap;color:var(--muted);font-size:12px}}
.toolbar-sticky{{position:sticky;top:0;z-index:20;margin:8px 0 10px;padding:6px 0;background:var(--bg)}}.model-controls{{display:flex;align-items:end;gap:10px;flex-wrap:wrap;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:12px 12px 0 0}}
.control-field{{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:11px;font-weight:700}}.control-label{{color:var(--muted);font-size:11px;font-weight:700}}.control-field select{{min-width:160px;padding:8px 30px 8px 10px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text);font-family:inherit;font-size:13px}}
.compare-picker{{position:relative}}.compare-menu{{position:relative}}.compare-menu summary{{list-style:none;cursor:pointer;padding:8px 12px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text);font-size:12px}}.compare-menu summary::-webkit-details-marker{{display:none}}.compare-menu[open] fieldset{{position:absolute;top:calc(100% + 6px);left:0;z-index:30;min-width:230px;max-height:280px;overflow:auto;padding:10px;border:1px solid var(--border);border-radius:10px;background:var(--surface);box-shadow:0 12px 28px #0002}}.compare-menu fieldset{{display:grid;gap:7px}}.compare-menu legend{{padding:0 4px;color:var(--muted);font-size:11px}}.model-option{{display:flex;align-items:center;gap:8px;padding:5px;border-radius:6px;color:var(--text);font-size:12px}}.model-option:hover{{background:var(--bg)}}.selection-status{{margin-left:auto;color:var(--muted);font-size:11px}}
.tab-bar{{display:flex;gap:5px;overflow-x:auto;margin:0;padding:7px;background:var(--surface);border:1px solid var(--border);border-top:0;border-radius:0 0 12px 12px;box-shadow:0 5px 18px #0000000d;scrollbar-width:thin}}
.tab{{flex:0 0 auto;border:0;border-radius:8px;background:transparent;color:var(--muted);padding:9px 14px;font-family:inherit;font-size:13px;font-weight:600;cursor:pointer;transition:background .16s ease,color .16s ease}}
.tab:hover{{background:var(--bg);color:var(--text)}}.tab[aria-selected="true"]{{background:var(--accent);color:#fff}}.tab:focus-visible{{outline:3px solid #60a5fa;outline-offset:2px}}
.tab-panel{{margin-top:10px}}.tab-panel[hidden]{{display:none}}.overview-intro{{color:var(--muted);margin:4px 2px 14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}.kpi,.chart{{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}}
.kpi{{padding:13px 16px}}.kpi-label{{color:var(--muted);font-size:12px}}.kpi-readings{{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}}.kpi-reading{{display:flex;flex-direction:column;min-width:100px;padding:6px 9px;background:var(--bg);border-radius:8px}}.kpi-reading strong{{font-size:17px}}.kpi-reading span{{color:var(--muted);font-size:10px;overflow-wrap:anywhere}}.chart-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.chart{{min-width:0}}
.metric-list{{display:grid;gap:9px}}.metric-card{{background:var(--surface);border:1px solid var(--border);border-radius:11px;overflow:hidden}}.metric-toggle{{display:flex;align-items:center;gap:16px;width:100%;padding:13px 15px;border:0;background:transparent;color:var(--text);text-align:left;cursor:pointer}}.metric-toggle:hover{{background:color-mix(in srgb,var(--surface),var(--accent) 5%)}}.metric-toggle:focus-visible{{outline:3px solid #60a5fa;outline-offset:-3px}}.metric-main{{flex:1;min-width:220px;display:grid;gap:3px}}.metric-main strong{{font-size:14px}}.metric-main small{{color:var(--muted);font-size:11px;line-height:1.35}}.metric-main em{{color:var(--accent);font-size:10px;font-style:normal;font-weight:700}}.metric-values{{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}}.metric-reading{{display:flex;flex-direction:column;gap:2px;padding:6px 9px;border-radius:7px;background:var(--bg);min-width:94px}}.metric-reading span{{color:var(--muted);font-size:10px;overflow-wrap:anywhere}}.metric-reading strong{{font-size:13px}}.metric-chevron{{font-size:18px;color:var(--muted);transition:transform .2s ease}}.metric-toggle[aria-expanded="true"] .metric-chevron{{transform:rotate(180deg)}}.metric-info{{padding:0 16px 13px 15px;color:var(--muted);font-size:12px}}.metric-info p{{margin:0 0 7px}}.metric-info a{{color:var(--accent);font-weight:600;text-decoration:none}}.metric-info a:hover{{text-decoration:underline}}.no-readings{{opacity:.7}}
.details-section{{margin-top:22px}}.details-section h2{{font-size:19px;margin:0 0 5px}}.details-intro{{color:var(--muted);margin:0 0 14px}}.details-section .kpi strong{{font-size:13px}}.kpi-model{{color:var(--muted);font-size:12px}}.model-details{{display:grid;gap:11px}}.model-detail-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}}
.model-detail-heading{{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 17px}}.model-detail-heading h3{{margin:0;font-size:15px}}.model-detail-heading span{{display:block;color:var(--muted);font-size:12px;margin-top:2px}}
.detail-toggle{{flex:0 0 auto;display:flex;align-items:center;gap:12px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text);padding:8px 11px;font-family:inherit;font-size:12px;font-weight:600;cursor:pointer}}.detail-toggle:hover{{border-color:var(--accent)}}.detail-toggle:focus-visible{{outline:3px solid #60a5fa;outline-offset:2px}}.detail-toggle span{{font-size:17px;transition:transform .2s ease}}.detail-toggle[aria-expanded="true"] span{{transform:rotate(180deg)}}
.detail-body{{height:0;overflow:hidden;border-top:1px solid transparent}}.detail-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;padding:14px 17px;margin:0;background:var(--bg)}}.detail-item{{min-width:0;padding:11px 12px;background:var(--surface);border:1px solid var(--border);border-radius:9px}}.detail-item dt{{font-size:12px;font-weight:700;text-transform:capitalize}}.detail-item dd{{margin:4px 0 0;font-size:13px;overflow-wrap:anywhere;white-space:pre-wrap}}.detail-item p{{margin:5px 0 0;color:var(--muted);font-size:11px;line-height:1.4}}.empty{{color:var(--muted);background:var(--surface);padding:16px;border:1px solid var(--border);border-radius:10px}}
@media(max-width:720px){{main{{padding:12px 10px 24px}}header{{padding:14px 16px}}h1{{font-size:21px}}.header-meta{{white-space:normal}}.toolbar-sticky{{margin:6px 0}}.tab{{padding:8px 11px}}.chart-grid{{grid-template-columns:1fr}}.metric-toggle{{align-items:flex-start;flex-wrap:wrap}}.metric-main{{flex-basis:75%}}.metric-values{{justify-content:flex-start}}.metric-chevron{{margin-left:auto}}}}
@media(max-width:560px){{.model-detail-heading{{align-items:flex-start;flex-direction:column}}.detail-toggle{{width:100%;justify-content:space-between}}.detail-grid{{grid-template-columns:1fr}}}}
@media(prefers-reduced-motion:reduce){{*,*::before,*::after{{scroll-behavior:auto!important;transition:none!important}}}}
</style></head><body><main><header><div><div class="brand">ModelDiet · Interactive profile</div><h1>{html.escape(title)}</h1><div class="sub">Compare measured performance, memory, and deployment characteristics.</div></div><div class="header-meta">{html.escape(generated)}<br>{len(results)} evaluated model{"s" if len(results) != 1 else ""}</div></header>
<div class="toolbar-sticky">{_model_selector(results)}<nav class="tab-bar" role="tablist" aria-label="Model comparison views">{tab_buttons}</nav></div>
<section class="tab-panel" id="panel-overview" role="tabpanel" aria-labelledby="tab-overview">
<p class="overview-intro">A quick summary of the supplied results. Open a tab to inspect each comparison in detail.</p><div class="kpis">{_cards(results)}</div></section>
<section class="tab-panel" id="panel-metrics" role="tabpanel" aria-labelledby="tab-metrics" hidden>
<p class="overview-intro">The same standard metrics are shown for one model or a comparison of two or three. Expand a metric to see how to read it and find a reference.</p><div class="metric-list">{_metric_cards(results)}</div></section>
{"".join(plot_blocks)}
<section class="tab-panel" id="panel-details" role="tabpanel" aria-labelledby="tab-details" hidden>
<section class="details-section"><h2>Model details</h2><p class="details-intro">Each card explains what was recorded for one model. Expand a card to see the details and what they mean.</p><div class="model-details">{_metadata(results)}</div></section>
<section class="details-section"><h2>How to read these measurements</h2><div class="kpis">
<div class="kpi"><strong>Latency percentiles</strong><div class="kpi-model">P50 is the median timed prediction. P95 and P99 show slower tail observations. They describe this local run and are affected by machine load and sample count.</div></div>
<div class="kpi"><strong>Process RSS</strong><div class="kpi-model">RSS is memory resident in the whole Python process. Before and after are snapshots; the change is not peak RSS and does not isolate the model.</div></div>
<div class="kpi"><strong>Python-traced memory</strong><div class="kpi-model">tracemalloc observes Python-tracked allocations during a separate prediction pass. Native library allocations may not be included.</div></div>
<div class="kpi"><strong>Serialized size</strong><div class="kpi-model">This is the measured pickle representation size. It is not the model’s in-memory footprint or necessarily its deployment artifact.</div></div>
</div></section></section>
</main>{TAB_SCRIPT}</body></html>"""
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
