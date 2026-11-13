"""Shared plotting helpers. Uses the non-interactive Agg backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

CONTROLLER_COLOURS = {"Blend": "#1f77b4", "Switch": "#ff7f0e", "PAG": "#2ca02c"}
DPI = 300


def save_figure(fig: Figure, path: Path, **kwargs: Any) -> None:
    """Save and close a figure, reporting where it was written."""
    fig.savefig(path, dpi=DPI, **kwargs)
    plt.close(fig)
    print(f"Saved {path}")


def _wrap_header(name: object, max_len: int = 14) -> str:
    """Wrap long underscore-separated column names onto two balanced lines."""
    name_str = str(name)
    if name_str.endswith("_Cohen_d"):
        prefix = name_str[: -len("_Cohen_d")].replace("_", " ")
        return f"{prefix}\nCohen's d"
    cleaned = name_str.replace("_", " ")
    words = cleaned.split(" ")
    if len(cleaned) <= max_len or len(words) == 1:
        return cleaned
    best_split = min(
        range(1, len(words)),
        key=lambda i: max(len(" ".join(words[:i])), len(" ".join(words[i:]))),
    )
    return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])


def save_table_png(df: pd.DataFrame, path: Path, title: str = "") -> None:
    """Render a DataFrame as a presentation-ready PNG table."""
    has_index = df.index.name is not None or isinstance(df.index, pd.MultiIndex)
    display_df = df.reset_index() if has_index else df.copy()

    n_cols, n_rows = len(display_df.columns), len(display_df)
    fig, ax = plt.subplots(figsize=(max(9.0, 2.6 * n_cols), max(3.2, 0.55 * n_rows + 2.4)))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", color="#1a1a1a", pad=22)

    cell_text = [
        [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        for row in display_df.values
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=[_wrap_header(c) for c in display_df.columns],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.4)

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 1 else "#eef2f5")
            cell.set_text_props(color="#222222")

    save_figure(fig, path, bbox_inches="tight", facecolor="white")
