"""Small optional desktop picker for a quick local profiling demonstration."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from modeldiet import profile
from modeldiet.profiling import (
    LatencyMeasurement,
    ProcessRSSMeasurement,
    SerializationMeasurement,
    TracedAllocationMeasurement,
)
from modeldiet.visualization import ModelDietVisualizer

DATASETS = ("Breast cancer", "Iris", "Wine")
MODELS = ("Logistic Regression", "Random Forest", "Gradient Boosting")


def _dataset(name: str) -> tuple[Any, Any]:
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine

    loaders = {
        "Breast cancer": load_breast_cancer,
        "Iris": load_iris,
        "Wine": load_wine,
    }
    dataset = loaders[name]()
    return dataset.data, dataset.target


def _model(name: str) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "Logistic Regression":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=42)
        )
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=60, random_state=42)
    raise ValueError(f"Unsupported example model: {name}")


def _profile_selection(
    dataset_name: str, model_names: list[str]
) -> dict[str, dict[str, Any]]:
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = _dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    output: dict[str, dict[str, Any]] = {}
    for name in model_names:
        model = _model(name)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        measured = profile(
            model,
            X_test,
            measurements=[
                LatencyMeasurement(),
                ProcessRSSMeasurement(),
                TracedAllocationMeasurement(),
                SerializationMeasurement(),
            ],
            include_environment=True,
        )
        output[name] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "loading_time": None,
            "measurement_details": {
                "latency": {
                    key: metric.value
                    for key, metric in measured.measurements["latency"].metrics.items()
                },
                "process_memory": {
                    key: metric.value
                    for key, metric in measured.measurements[
                        "process_rss"
                    ].metrics.items()
                },
                "allocation": {
                    key: metric.value
                    for key, metric in measured.measurements[
                        "python_traced_allocation"
                    ].metrics.items()
                },
                "serialization": {
                    key: metric.value
                    for key, metric in measured.measurements[
                        "pickle_size"
                    ].metrics.items()
                },
            },
            "environment": measured.environment,
            "sample_count": measured.sample_count,
            "model_configuration": {
                "class": measured.model_class,
                "module": measured.model_module,
            },
            "dataset": dataset_name,
            "training_rows": len(X_train),
        }
    return output


def main() -> None:  # pragma: no cover -- interactive Tk picker requires a display
    """Show a dataset/model picker and open one combined native dashboard."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "The ModelDiet demo window needs a desktop session. In a headless "
            "environment, use ModelDietVisualizer from Python instead."
        ) from exc

    root.title("ModelDiet · Quick comparison")
    root.geometry("460x500")
    root.minsize(420, 460)
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="ModelDiet", font=("TkDefaultFont", 20, "bold")).pack(
        anchor="w"
    )
    ttk.Label(
        frame,
        text="Choose a sample dataset and models. We’ll train, profile, and open one comparison dashboard.",
        wraplength=400,
    ).pack(anchor="w", pady=(6, 18))

    ttk.Label(frame, text="Dataset").pack(anchor="w")
    dataset_var = tk.StringVar(value=DATASETS[0])
    dataset_picker = ttk.Combobox(
        frame, textvariable=dataset_var, values=DATASETS, state="readonly"
    )
    dataset_picker.pack(fill="x", pady=(4, 16))

    ttk.Label(frame, text="Models to compare").pack(anchor="w")
    selected = {name: tk.BooleanVar(value=True) for name in MODELS}
    for name in MODELS:
        ttk.Checkbutton(frame, text=name, variable=selected[name]).pack(
            anchor="w", pady=2
        )

    status = tk.StringVar(
        value="Runs locally with small built-in scikit-learn datasets."
    )
    ttk.Label(frame, textvariable=status, wraplength=400).pack(anchor="w", pady=(18, 8))
    run_button = ttk.Button(frame, text="Train, profile, and view results")
    run_button.pack(fill="x", ipady=7, pady=(4, 0))

    def completed(results: dict[str, dict[str, Any]] | None, error: str | None) -> None:
        run_button.configure(state="normal")
        if error is not None:
            status.set("The run failed. See the error details.")
            messagebox.showerror("ModelDiet run failed", error, parent=root)
            return
        assert results is not None
        visualizer = ModelDietVisualizer(results)
        try:
            visualizer.show(experiment_name=f"{dataset_var.get()} example")
        except RuntimeError as exc:
            messagebox.showerror("Could not open dashboard", str(exc), parent=root)
            return
        status.set("Dashboard opened in a local window.")

    def run() -> None:
        models = [name for name, variable in selected.items() if variable.get()]
        if not models:
            messagebox.showwarning(
                "Choose a model", "Select at least one model to run.", parent=root
            )
            return
        run_button.configure(state="disabled")
        status.set("Training and profiling selected models…")
        dataset_name = dataset_var.get()

        def worker() -> None:
            try:
                result = _profile_selection(dataset_name, models)
            except (
                Exception
            ) as exc:  # Surface failures in the UI instead of losing the worker error.
                root.after(0, completed, None, f"{type(exc).__name__}: {exc}")
            else:
                root.after(0, completed, result, None)

        threading.Thread(target=worker, daemon=True).start()

    run_button.configure(command=run)
    root.mainloop()


if __name__ == "__main__":
    main()
