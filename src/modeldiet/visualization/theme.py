"""Shared styling for ModelDiet's Plotly figures."""

from typing import Any

BACKGROUND = "#f4f6f8"
SURFACE = "#ffffff"
TEXT = "#17212b"
MUTED = "#5b6875"
GRID = "#e5e9ed"
ACCENTS = ["#2364aa", "#2a9d8f", "#e09f3e", "#9b5de5", "#e76f51"]
ACCENT_SOFT = ["#dceafe", "#d9f3ed", "#fff0d5", "#eee2ff", "#ffe3dc"]
THEMES = {
    "light": {
        "background": BACKGROUND,
        "surface": SURFACE,
        "text": TEXT,
        "muted": MUTED,
        "grid": GRID,
        "hover": "#17212b",
    },
    "dark": {
        "background": "#111827",
        "surface": "#1f2937",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "grid": "#374151",
        "hover": "#030712",
    },
}


def apply_theme(
    fig: Any, title: str | None = None, subtitle: str = "", theme: str = "light"
) -> Any:
    """Apply the shared light or dark Plotly theme and optional title."""
    if theme not in THEMES:
        raise ValueError("theme must be 'light' or 'dark'")
    colors = THEMES[theme]
    layout: dict[str, Any] = {
        "font": {
            "family": "Inter, Segoe UI, sans-serif",
            "color": colors["text"],
            "size": 13,
        },
        "title_font": {"size": 18, "color": colors["text"]},
        "paper_bgcolor": colors["background"],
        "plot_bgcolor": colors["surface"],
        "margin": {"l": 64, "r": 28, "t": 76 if subtitle else 58, "b": 60},
        "legend": {"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        "hoverlabel": {"bgcolor": colors["hover"], "font": {"color": "white"}},
        "hovermode": "closest",
        "hoverdistance": 24,
        "dragmode": "zoom",
        "bargap": 0.24,
        "bargroupgap": 0.1,
        "autosize": True,
    }
    if title is not None:
        layout["title"] = {
            "text": f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            "x": 0.02,
            "xanchor": "left",
        }
    fig.update_layout(**layout)
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=colors["grid"],
        title_font_color=colors["text"],
        tickfont_color=colors["muted"],
    )
    fig.update_yaxes(
        gridcolor=colors["grid"],
        zeroline=False,
        linecolor=colors["grid"],
        title_font_color=colors["text"],
        tickfont_color=colors["muted"],
    )
    return fig
