# Interactive visualization

Install Plotly separately from the core profiler:

```bash
python -m pip install "modeldiet[viz]"
```

For the guided desktop picker, install the example model dependencies too and
launch the demo:

```bash
python -m pip install "modeldiet[demo]"
modeldiet-demo
```

Choose a built-in dataset and one or more sample models, then select **Train,
profile, and view results**. Training and measurements run locally; the
selected models are trained once for that run. A native local dashboard
window opens when profiling finishes; it does not launch a browser or server.
The picker and dashboard use Tkinter from the Python standard library and
require a desktop session. This native demo does not require Plotly.
`visualizer.show()` opens the same native dashboard
for results you already have. The dashboard explains latency percentiles, whole-process RSS,
Python-traced allocation, and serialized size, including the interpretation
limits of each measurement.

`ModelDietVisualizer` accepts a mapping of display names to `ProfileResult`
objects or ModelDiet benchmark result mappings. It does not execute models.
Plot methods return Plotly figures, so they work in notebooks and scripts:

```python
from modeldiet.visualization import ModelDietVisualizer

visualizer = ModelDietVisualizer({"Classifier A": result_a, "Classifier B": result_b})
visualizer.plot_latency_comparison().show()
visualizer.compare_models("Classifier A", "Classifier B").show()
visualizer.plot_memory_comparison(models=["Classifier A", "Classifier B"])
visualizer.generate_report("modeldiet-report.html", experiment_name="April run")
```

Use `ModelDietVisualizer.from_file("results.json")` to load either a combined
mapping or a single benchmark-worker JSON record. Use `theme="dark"` in the
constructor for dark figures and reports. `plot_latency_comparison()` accepts
`p50_ms`, `p95_ms`, and `p99_ms`. `plot_latency_distribution()` accepts
`ecdf`, `box`, or `violin`, and renders observations only if the results
actually include `latency_observations_ms`.

`compare_models(reference, comparison)` returns a table with both raw values,
the absolute difference, and percent change relative to the reference. A
percentage change is omitted when the reference is zero or either value is
missing. `plot_normalized_comparison(metrics, baseline=...)` displays each
value as an index where the baseline is 100; it does not replace the raw charts
or indicate an overall winner.

The CLI accepts existing JSON result files and does not rerun benchmarks:

```bash
modeldiet visualize results/lr.json results/rf.json --output report.html
```

It can also be called as `modeldiet-report results/lr.json results/rf.json`.
The generated report embeds Plotly JavaScript for offline use. The current
profile schema does not contain accuracy, loading time, peak RSS, precision,
recall, F1, or independent repeated-run samples by default; corresponding
charts or values appear only when those fields are supplied by the source
result mapping. RSS snapshots and Python-traced allocation remain distinct.
# Visualization module structure

The public `ModelDietVisualizer` follows a parent/child design. Shared result
validation, JSON loading, model selection, and theme application live in
`VisualizationBase` (`modeldiet.visualization.base`). The public visualizer
inherits those common behaviors and owns the chart-specific methods exposed to
users. The lower-level chart functions in `charts.py` remain reusable by the
HTML report and direct imports.

```text
VisualizationBase
└── ModelDietVisualizer
    ├── plot_latency_comparison / plot_latency_distribution
    ├── plot_memory_comparison / plot_python_memory
    ├── plot_loading_time / plot_artifact_size
    ├── plot_accuracy_latency / plot_normalized_comparison
    └── compare_models / generate_report / show
```

When adding a chart, put measurement-specific construction in `charts.py`,
expose a user-facing method on `ModelDietVisualizer`, and use `_select()` and
`_style()` for the common behavior. Keep the returned Plotly figure interactive
and represent missing measurements as unavailable rather than as zero. New
shared behavior belongs in `VisualizationBase`, so chart methods do not each
reimplement validation, selection, or theme handling.
