"""Figures for the notebook.

Only the few charts that read better than the table they replace. Everything else stays
a table: a bar chart of five numbers is decoration, not communication.

Matplotlib is an optional dependency — importing this module is what pulls it in, so the
rest of the library still installs and runs without it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..modeling import LambdaRanker

DEFAULT_TOP_N = 10


def feature_importance_chart(model: LambdaRanker, top_n: int = DEFAULT_TOP_N,
                             ax: Axes | None = None) -> Axes:
    """Horizontal bar chart of the `top_n` features by gain.

    Gain is how much a feature improved the objective across every split it was used in,
    so the axis is only meaningful in relative terms — the ordering and the size of the
    gaps are the readable part, not the absolute values.
    """
    import matplotlib.pyplot as plt

    importance = model.feature_importance().head(top_n).iloc[::-1]
    column = importance.columns[1]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.42 * len(importance) + 1))
    ax.barh(importance["feature"], importance[column], color=PRIMARY_COLOUR,
            **BAR_STYLE)
    _style_ax(ax, f"Top {len(importance)} Features by {column.title()}",
              xlabel=f"{column} (relative)")
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    plt.tight_layout()
    return ax


# the EDA's activity-segment palette, in the same order, so a reader moving between the
# two notebooks reads the same colour as the same kind of customer
SEGMENT_COLOURS = ("seagreen", "steelblue", "darkorange", "firebrick")
PRIMARY_COLOUR = "steelblue"          # the EDA's single-series colour
BAR_STYLE = dict(edgecolor="white", alpha=0.85)


def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """The EDA notebook's axis styling, so both notebooks look like one project."""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(axis="both", labelsize=8)


def segment_chart(table: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """How well the model ranks for each customer segment, as grouped bars.

    One group per metric, one bar per segment, so segments are compared within a metric
    rather than across metrics that sit on different scales.

    Who buys, and how often, is established in the EDA — this only answers how well the
    model serves each group once they do.

    Takes the frame `segment_table` returns (grouped `population` / `accuracy` columns).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    accuracy = table["accuracy"]
    segments, metrics = list(table.index), list(accuracy.columns)
    colours = [SEGMENT_COLOURS[i % len(SEGMENT_COLOURS)] for i in range(len(segments))]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))

    positions = np.arange(len(metrics))
    width = 0.8 / len(segments)
    for index, segment in enumerate(segments):
        ax.bar(positions + index * width, accuracy.loc[segment].to_numpy(),
               width, label=segment, color=colours[index], **BAR_STYLE)

    ax.set_xticks(positions + width * (len(segments) - 1) / 2)
    ax.set_xticklabels(metrics)
    _style_ax(ax, "Ranking Quality by Customer Segment", ylabel="score")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return ax
