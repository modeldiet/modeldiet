"""Native, local Tkinter dashboard used by the guided demo and ``show()``."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Mapping
from tkinter import ttk
from typing import Any

from modeldiet.result import ProfileResult
from modeldiet.visualization.data import (
    accuracy_value,
    loading_time_value,
    metric_number,
)

PALETTE = ("#276EF1", "#00A884", "#F59E0B", "#9B5DE5", "#E25555")
BG = "#0B1220"
SURFACE = "#111B2E"
TEXT = "#E7EDF7"
MUTED = "#91A0B8"
GRID = "#29364D"


def _value(record: ProfileResult | Mapping[str, Any], metric: str) -> float | None:
    if metric == "accuracy":
        return accuracy_value(record)
    if metric == "loading":
        return loading_time_value(record)
    fields = {
        "mean": ("latency", "mean_ms"),
        "p50": ("latency", "p50_ms"),
        "p95": ("latency", "p95_ms"),
        "p99": ("latency", "p99_ms"),
        "rss_before": ("process_rss", "rss_before_inference_mb"),
        "rss_after": ("process_rss", "rss_after_inference_mb"),
        "rss_delta": ("process_rss", "rss_delta_mb"),
        "traced": ("python_traced_allocation", "traced_peak_inference_allocation_mb"),
        "size": ("pickle_size", "serialized_model_size_mb"),
    }
    measurement, key = fields[metric]
    return metric_number(record, measurement, key)


class _BarChart(tk.Canvas):  # pragma: no cover -- Tk rendering requires a display
    def __init__(
        self, master: tk.Misc, title: str, series: list[tuple[str, str]], **kwargs: Any
    ) -> None:
        super().__init__(master, background=SURFACE, highlightthickness=0, **kwargs)
        self.title = title
        self.series = series
        self.results: Mapping[str, ProfileResult | Mapping[str, Any]] = {}
        self.bind("<Configure>", lambda _event: self.draw())

    def set_results(
        self, results: Mapping[str, ProfileResult | Mapping[str, Any]]
    ) -> None:
        self.results = results
        self.draw()

    def draw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 400), max(self.winfo_height(), 280)
        self.create_text(
            24,
            22,
            text=self.title,
            anchor="w",
            fill=TEXT,
            font=("TkDefaultFont", 14, "bold"),
        )
        if not self.results:
            self.create_text(
                width / 2,
                height / 2,
                text="Run a profile to see measured results",
                fill=MUTED,
            )
            return
        left, right, top, bottom = 78, width - 28, 66, height - 58
        values = [
            _value(record, key)
            for record in self.results.values()
            for _, key in self.series
        ]
        finite = [item for item in values if item is not None]
        if not finite:
            self.create_text(
                width / 2,
                height / 2,
                text="This measurement was not included in the results",
                fill=MUTED,
                width=width - 80,
            )
            return
        maximum = max(finite)
        minimum = min(0.0, min(finite))
        span = maximum - minimum or 1.0
        self.create_line(left, top, left, bottom, fill=GRID)
        self.create_line(left, bottom, right, bottom, fill=GRID)
        for tick in range(5):
            fraction = tick / 4
            y = bottom - fraction * (bottom - top)
            tick_value = minimum + fraction * span
            self.create_line(left, y, right, y, fill=GRID)
            self.create_text(
                left - 8,
                y,
                text=f"{tick_value:.3g}",
                anchor="e",
                fill=MUTED,
                font=("TkDefaultFont", 8),
            )
        model_count = len(self.results)
        slot = (right - left) / max(model_count, 1)
        group_width = min(slot * 0.72, 120)
        bar_width = group_width / len(self.series)
        for model_index, (name, record) in enumerate(self.results.items()):
            center = left + slot * (model_index + 0.5)
            for series_index, (label, key) in enumerate(self.series):
                raw_value = _value(record, key)
                if raw_value is None:
                    continue
                x0 = center - group_width / 2 + series_index * bar_width + 2
                x1 = x0 + max(bar_width - 4, 3)
                y = bottom - (raw_value - minimum) / span * (bottom - top)
                self.create_rectangle(
                    x0,
                    y,
                    x1,
                    bottom,
                    fill=PALETTE[series_index % len(PALETTE)],
                    outline="",
                )
                self.create_text(
                    (x0 + x1) / 2,
                    y - 10,
                    text=f"{raw_value:.3g}",
                    fill=TEXT,
                    font=("TkDefaultFont", 8),
                )
            self.create_text(
                center,
                bottom + 18,
                text=name,
                fill=MUTED,
                width=max(slot - 6, 50),
                font=("TkDefaultFont", 8),
            )
        legend_x = left
        for index, (label, _key) in enumerate(self.series):
            x = legend_x + index * 130
            self.create_rectangle(
                x,
                height - 20,
                x + 10,
                height - 10,
                fill=PALETTE[index % len(PALETTE)],
                outline="",
            )
            self.create_text(
                x + 16,
                height - 15,
                text=label,
                anchor="w",
                fill=MUTED,
                font=("TkDefaultFont", 8),
            )


class _Tradeoff3D(
    tk.Canvas
):  # pragma: no cover -- interactive canvas requires a display
    """Small rotatable 3D view of accuracy, P99 latency, and process RSS."""

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, background=SURFACE, highlightthickness=0, **kwargs)
        self.results: Mapping[str, ProfileResult | Mapping[str, Any]] = {}
        self.yaw = 0.7
        self.pitch = 0.38
        self._drag: tuple[int, int] | None = None
        self.bind("<Configure>", lambda _event: self.draw())
        self.bind("<ButtonPress-1>", self._start_drag)
        self.bind("<B1-Motion>", self._rotate)

    def set_results(
        self, results: Mapping[str, ProfileResult | Mapping[str, Any]]
    ) -> None:
        self.results = results
        self.draw()

    def _start_drag(self, event: tk.Event) -> None:
        self._drag = (event.x, event.y)

    def _rotate(self, event: tk.Event) -> None:
        if self._drag:
            old_x, old_y = self._drag
            self.yaw += (event.x - old_x) * 0.012
            self.pitch = max(-1.1, min(1.1, self.pitch + (event.y - old_y) * 0.01))
            self._drag = (event.x, event.y)
            self.draw()

    def draw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 460), max(self.winfo_height(), 340)
        self.create_text(
            24,
            22,
            text="3D deployment trade-off",
            anchor="w",
            fill=TEXT,
            font=("TkDefaultFont", 14, "bold"),
        )
        self.create_text(
            24,
            46,
            text="Drag to rotate · X: P99 latency · Y: process RSS · Z: test accuracy",
            anchor="w",
            fill=MUTED,
            font=("TkDefaultFont", 9),
        )
        points: list[tuple[str, float, float, float]] = []
        for name, record in self.results.items():
            x, y, z = (
                _value(record, "p99"),
                _value(record, "rss_after"),
                _value(record, "accuracy"),
            )
            if x is not None and y is not None and z is not None:
                points.append((name, x, y, z))
        if not points:
            self.create_text(
                width / 2,
                height / 2,
                text="Accuracy, P99 latency, and process RSS are all needed for this view",
                fill=MUTED,
                width=width - 80,
            )
            return
        ranges: list[tuple[float, float]] = []
        axes: tuple[list[float], ...] = (
            [point[1] for point in points],
            [point[2] for point in points],
            [point[3] for point in points],
        )
        for vals in axes:
            lo, hi = min(vals), max(vals)
            ranges.append((lo, hi if hi != lo else lo + 1.0))
        center_x, center_y = width * 0.52, height * 0.60
        scale = min(width * 0.29, height * 0.33)

        def project(x: float, y: float, z: float) -> tuple[float, float]:
            xn, yn, zn = x * 2 - 1, y * 2 - 1, z * 2 - 1
            rotated_x = xn * math.cos(self.yaw) - yn * math.sin(self.yaw)
            rotated_y = xn * math.sin(self.yaw) + yn * math.cos(self.yaw)
            screen_x = center_x + (rotated_x - rotated_y) * scale * 0.58
            screen_y = (
                center_y
                + (rotated_x + rotated_y) * scale * 0.25
                - zn * scale * math.cos(self.pitch)
            )
            return screen_x, screen_y

        origin = project(0, 0, 0)
        corners = (project(1, 0, 0), project(0, 1, 0), project(0, 0, 1))
        for endpoint, label in zip(
            corners, ("P99 latency →", "Process RSS →", "Accuracy ↑")
        ):
            self.create_line(*origin, *endpoint, fill=GRID, width=2, arrow=tk.LAST)
            self.create_text(
                endpoint[0],
                endpoint[1],
                text=label,
                fill=MUTED,
                font=("TkDefaultFont", 9),
            )
        for index, (name, x, y, z) in enumerate(points):
            nx = (x - ranges[0][0]) / (ranges[0][1] - ranges[0][0])
            ny = (y - ranges[1][0]) / (ranges[1][1] - ranges[1][0])
            nz = (z - ranges[2][0]) / (ranges[2][1] - ranges[2][0])
            px, py = project(nx, ny, nz)
            color = PALETTE[index % len(PALETTE)]
            self.create_oval(
                px - 8, py - 8, px + 8, py + 8, fill=color, outline="#ffffff", width=2
            )
            self.create_text(
                px + 12,
                py - 8,
                text=name,
                anchor="w",
                fill=TEXT,
                font=("TkDefaultFont", 9, "bold"),
            )
            self.create_text(
                px + 12,
                py + 9,
                text=f"{x:.3g} ms · {y:.3g} MiB · {z:.1%}",
                anchor="w",
                fill=MUTED,
                font=("TkDefaultFont", 8),
            )


def _build_results_window(
    root: tk.Tk, results: Mapping[str, ProfileResult | Mapping[str, Any]], title: str
) -> None:  # pragma: no cover -- Tk dashboard requires a display
    root.title(f"ModelDiet · {title}")
    root.geometry("1100x760")
    root.minsize(760, 600)
    root.configure(background=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure(
        "TNotebook.Tab", background=SURFACE, foreground=TEXT, padding=(14, 9)
    )
    style.map("TNotebook.Tab", background=[("selected", "#1D2B44")])

    outer = ttk.Frame(root, padding=20)
    outer.pack(fill="both", expand=True)
    ttk.Label(
        outer,
        text="MODELDIET  /  LOCAL PROFILE",
        foreground="#67A1FF",
        font=("TkDefaultFont", 9, "bold"),
    ).pack(anchor="w")
    ttk.Label(outer, text=title, font=("TkDefaultFont", 22, "bold")).pack(
        anchor="w", pady=(2, 3)
    )
    ttk.Label(
        outer,
        text=f"{len(results)} model(s) · measured locally · drag the 3D chart to rotate",
        foreground=MUTED,
    ).pack(anchor="w", pady=(0, 14))

    notebook = ttk.Notebook(outer)
    notebook.pack(fill="both", expand=True)
    tabs: dict[str, ttk.Frame] = {}
    for tab_name in (
        "Overview",
        "Latency",
        "Process RSS",
        "Python memory",
        "Model size",
        "Loading time",
        "3D trade-offs",
        "Measurement guide",
    ):
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text=tab_name)
        tabs[tab_name] = frame

    overview = tabs["Overview"]
    for index, (name, record) in enumerate(results.items()):
        card = tk.Frame(
            overview,
            background=SURFACE,
            padx=16,
            pady=14,
            highlightbackground=GRID,
            highlightthickness=1,
        )
        card.pack(fill="x", pady=5)
        tk.Label(
            card,
            text=name,
            background=SURFACE,
            foreground=TEXT,
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")
        metrics = [
            ("Test accuracy", _value(record, "accuracy"), "%"),
            ("P99 latency", _value(record, "p99"), " ms"),
            ("Process RSS", _value(record, "rss_after"), " MiB"),
            ("Python traced peak", _value(record, "traced"), " MiB"),
            ("Serialized size", _value(record, "size"), " MiB"),
        ]
        line = "    ·    ".join(
            f"{label}: unavailable"
            if value is None
            else f"{label}: {value * 100:.1f}%"
            if unit == "%"
            else f"{label}: {value:.4g}{unit}"
            for label, value, unit in metrics
        )
        tk.Label(
            card,
            text=line,
            background=SURFACE,
            foreground=MUTED,
            wraplength=980,
            justify="left",
        ).pack(anchor="w", pady=(7, 0))

    latency = _BarChart(
        tabs["Latency"],
        "Prediction time by percentile · milliseconds",
        [("P50", "p50"), ("P95", "p95"), ("P99", "p99")],
    )
    latency.pack(fill="both", expand=True)
    rss = _BarChart(
        tabs["Process RSS"],
        "Whole-process resident memory · MiB",
        [
            ("Before inference", "rss_before"),
            ("After inference", "rss_after"),
            ("Snapshot change", "rss_delta"),
        ],
    )
    rss.pack(fill="both", expand=True)
    traced = _BarChart(
        tabs["Python memory"],
        "Python-traced peak allocation · MiB",
        [("Traced peak", "traced")],
    )
    traced.pack(fill="both", expand=True)
    size = _BarChart(
        tabs["Model size"],
        "Serialized pickle representation · MiB",
        [("Serialized size", "size")],
    )
    size.pack(fill="both", expand=True)
    loading = _BarChart(
        tabs["Loading time"],
        "Measured loading duration · seconds",
        [("Loading time", "loading")],
    )
    loading.pack(fill="both", expand=True)
    tradeoffs = _Tradeoff3D(tabs["3D trade-offs"])
    tradeoffs.pack(fill="both", expand=True)
    for chart in (latency, rss, traced, size, loading, tradeoffs):
        chart.set_results(results)

    explanations = (
        (
            "P50 / P95 / P99",
            "Percentiles say how many timed predictions completed at or below a latency. P50 is the median; P95 and P99 describe the slower tail. Tail estimates are less stable when there are few observations.",
        ),
        (
            "Process RSS",
            "Resident set size is memory used by the whole Python process. Before and after are snapshots, not a measured peak, and do not isolate the model.",
        ),
        (
            "Python-traced peak",
            "tracemalloc counts Python-tracked allocations during a separate inference pass. Native allocations used by libraries may not be included.",
        ),
        (
            "Serialized size",
            "This is the model's pickle representation size. It is not the in-memory footprint or necessarily the artifact used in deployment.",
        ),
        (
            "Accuracy vs resource use",
            "Accuracy describes this dataset split. Latency and memory describe this local run. A difference is an observation, not a causal explanation or universal winner.",
        ),
    )
    for heading, body in explanations:
        tk.Label(
            tabs["Measurement guide"],
            text=heading,
            background=BG,
            foreground=TEXT,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(10, 2))
        tk.Label(
            tabs["Measurement guide"],
            text=body,
            background=BG,
            foreground=MUTED,
            wraplength=900,
            justify="left",
        ).pack(anchor="w")


def show_results(
    results: Mapping[str, ProfileResult | Mapping[str, Any]],
    title: str = "Model comparison",
) -> None:  # pragma: no cover -- opening a native window requires a display
    """Open results in a native local window; no browser or server is used."""
    root = tk.Tk()
    _build_results_window(root, results, title)
    root.mainloop()
