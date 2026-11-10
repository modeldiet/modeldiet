"""Shared styling for ModelDiet's Plotly figures."""

from typing import Any

BACKGROUND = "#f4f6f8"
SURFACE = "#ffffff"
TEXT = "#17212b"
MUTED = "#5b6875"
GRID = "#e5e9ed"
ACCENTS = ["#2364aa", "#2a9d8f", "#e09f3e", "#9b5de5", "#e76f51"]
ACCENT_SOFT = ["#dceafe", "#d9f3ed", "#fff0d5", "#eee2ff", "#ffe3dc"]


def apply_theme(fig: Any, title: str, subtitle: str = "") -> Any:
    """Apply a consistent compact engineering-report layout."""
    fig.update_layout(
        title={
            "text": f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            "x": 0.02,
            "xanchor": "left",
        },
        font={"family": "Inter, Segoe UI, sans-serif", "color": TEXT, "size": 13},
        title_font={"size": 18, "color": TEXT},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin={"l": 64, "r": 28, "t": 76 if subtitle else 58, "b": 60},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        hovermode="closest",
        hoverdistance=24,
        dragmode="zoom",
        bargap=0.24,
        bargroupgap=0.1,
        hoverlabel={"bgcolor": "#17212b", "font": {"color": "white"}},
        autosize=True,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig
