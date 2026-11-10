# Interactive visualization

ModelDiet provides Plotly figures for notebooks and scripts, plus an offline,
standalone HTML dashboard for comparing existing profiling results. The report
uses results you already collected; it does not train or profile models.

## Install

Install Plotly for interactive charts and HTML reports:

```bash
python -m pip install "modeldiet[viz]"
```

For the optional guided desktop demo, install the example model dependencies:

```bash
python -m pip install "modeldiet[demo]"
modeldiet-demo
```

The desktop picker and dashboard use Tkinter and need a desktop session. They
do not need Plotly. The `show()` method opens that native dashboard for results
you already have.

## Create a dashboard

`ModelDietVisualizer` accepts either `ProfileResult` objects or supported
benchmark JSON mappings:

```python
from modeldiet.visualization import ModelDietVisualizer

visualizer = ModelDietVisualizer.from_file("results.json")
visualizer.generate_report(
    "modeldiet-report.html",
    experiment_name="April model comparison",
)
```

Open `modeldiet-report.html` in a browser. The report is self-contained and
works offline. It contains Overview, Metrics, Latency, Memory, Deployment,
Trade-offs, and Model details tabs. The tab bar stays visible while scrolling.

## Select one model or compare models

A selector above the tabs controls every report view:

- **One model** displays one selected model across the overview, metrics,
  details, and charts.
- **Compare 2–3 models** updates those same views for two or three selected
  models. The report prevents selecting more than three.

The Metrics tab keeps a consistent set of rows in either mode. Each row shows
the selected model value or `Unavailable`. Expand a metric to see its
plain-English meaning, guidance on whether higher or lower is generally
favorable, and a link to read more.

| Metric | General direction | What it tells you |
| --- | --- | --- |
| Accuracy | Higher is generally better | Share of predictions correct for this task and evaluation split. |
| Mean latency, P50, P95, P99 | Lower is generally better | Prediction time; P95 and P99 describe slower tail observations. |
| RSS before/after | Lower generally uses fewer resources | Whole-process memory snapshots, not model-only memory or peak usage. |
| RSS change | A smaller increase is generally preferable | Difference between snapshots; negative means RSS fell during the interval. |
| Python-traced peak | Lower generally uses fewer Python-managed allocations | Native library allocations may not be included. |
| Model loading time | Lower is generally better for startup | Depends on storage, cache state, and loading method. |
| Serialized model size | Lower reduces storage and transfer | Pickle representation size, not in-memory size or necessarily the deployment artifact. |

These are guidelines for like-for-like comparisons, not an overall model score.
For example, RSS includes the Python process and its other loaded data. The
report links to scikit-learn's accuracy documentation, NIST's percentile guide,
and the psutil and Python documentation for memory and serialization details.

## Use Plotly figures directly

Plot methods return figures that can be shown or exported individually:

```python
from modeldiet.visualization import ModelDietVisualizer

visualizer = ModelDietVisualizer({"Classifier A": result_a, "Classifier B": result_b})
visualizer.plot_latency_comparison().show()
visualizer.plot_memory_comparison(models=["Classifier A", "Classifier B"])
visualizer.compare_models("Classifier A", "Classifier B").show()
visualizer.plot_normalized_comparison(
    ["p99_ms", "serialized_size_mib"], baseline="Classifier A"
).show()
```

In a Jupyter notebook, call a plot method in a cell to render it inline. Use
`.write_html("chart.html")` to export one figure. Use `theme="dark"` in the
constructor for dark figures and reports.

Available public plots include latency percentiles, mean latency, latency
distribution, process RSS, Python-traced memory, loading time, serialized
size, accuracy-versus-latency, normalized metrics, and pairwise comparison.
`plot_latency_comparison()` accepts `p50_ms`, `p95_ms`, and `p99_ms`.
`plot_latency_distribution()` accepts `ecdf`, `box`, or `violin`.

The ECDF uses retained `latency_observations_ms` when supplied. If only P50,
P95, or P99 summaries are available, it shows those reported points as a
clearly labeled percentile summary; it does not invent raw observations. Box
and violin plots need raw observations and explain when they are unavailable.

`visualizer.available_metrics` maps metric identifiers to the models that
provide them. Missing and non-finite values remain unavailable rather than
being changed to zero. Supported result mappings include nested
`measurement_details` and legacy flat fields.

## Generate a report from the CLI

Pass a combined result mapping or one or more per-model benchmark JSON files:

```bash
modeldiet visualize results.json --output report.html
modeldiet-report model-a.json model-b.json --experiment "Model comparison"
```

The CLI writes the report and does not open a browser automatically. Reports
can optionally include a combined JSON sidecar with `--include-json`.

## Pairwise and normalized comparisons

`compare_models(reference, comparison)` returns a figure with both values, the
absolute difference, and percentage change relative to the reference. The
percentage is omitted if the reference is zero or either value is unavailable.
A positive change is not automatically an improvement: higher accuracy is
generally favorable, while lower latency and memory use are generally
favorable.

`plot_normalized_comparison(metrics, baseline=...)` sets the baseline to 100
and displays other values as relative indices. It is not a performance score;
raw values remain available in the individual charts and original results.

## Contributor structure

Result validation and metric aliases are centralized in
`modeldiet.visualization.data`. `VisualizationBase` handles shared result
validation, JSON loading, selection, and theme configuration. The public
`ModelDietVisualizer` is the façade; chart builders live in `charts.py`, metric
comparisons in `comparison.py`, and the standalone dashboard in `html_report.py`.

When adding a chart, use the shared data accessors and theme application, then
expose the chart through `ModelDietVisualizer`. Keep missing values unavailable
instead of substituting zero. The report currently compares models present in
the supplied results; dataset uploads and public-dataset selection are future
work.
