"""Phase 3 charts: static PNGs for the repo. The CSVs in reports/ are their table view.

Encoding: nights and days are the two series each chart is about, so they
take the first two categorical slots (blue, orange; validated with the
dataviz palette checker, light surface). "Nights x days" is context, so it
is gray. The y axis is logarithmic because the legs end orders of magnitude
apart. Lines are thin, grid and axes are hairlines, and text wears ink
colors, never a series color.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; never opens a window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NIGHT = "#2a78d6"  # categorical slot 1
DAY = "#eb6834"    # categorical slot 2

SERIES: tuple[tuple[str, str, str], ...] = (
    ("night", "Nights", NIGHT),
    ("day", "Days", DAY),
    ("split_hold", "Nights × days", MUTED),
)
_LABEL_MIN_RATIO = 1.8  # end labels closer than this on the log scale would collide


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
    })


def compact(v: float) -> str:
    """A positive number for labels: 0.075, 4.16, 55.5, 14,543, 359M, 18.1B."""
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if v >= size:
            return f"{v / size:,.3g}{suffix}"
    if v >= 1e3:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.3g}"
    return f"{v:.2g}"


def multiple(v: float) -> str:
    """Growth multiple as read aloud: ×55.5, ×0.075, and below 0.01 a division (÷3.1M),
    because ×3.2e-07 is unreadable."""
    return f"÷{compact(1 / v)}" if v < 0.01 else f"×{compact(v)}"


def times(v: float, _pos: object = None) -> str:
    """Axis tick: ÷1,000, ×0.1, ×1, ×1,000, ×100M."""
    def big(x: float) -> str:
        for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if x >= size * (1 - 1e-9):  # 1 / 1e-9 is 999,999,999.99...
                return f"{x / size:.3g}{suffix}"
        return f"{x:,.0f}"

    if v < 0.01:
        return f"÷{big(1 / v)}"
    if v >= 1:
        return f"×{big(v)}"
    return f"×{v:g}"


def labels_that_fit(ends: dict[str, float], min_ratio: float = _LABEL_MIN_RATIO) -> set[str]:
    """Series whose end label clears its neighbours on a log scale.

    Converging labels are not nudged apart (that detaches them from their
    lines); they are left to the legend.
    """
    order = sorted(ends, key=lambda k: ends[k])
    ok = set()
    for i, k in enumerate(order):
        lo = order[i - 1] if i > 0 else None
        hi = order[i + 1] if i + 1 < len(order) else None
        clear_lo = lo is None or ends[k] / ends[lo] >= min_ratio
        clear_hi = hi is None or ends[hi] / ends[k] >= min_ratio
        if clear_lo and clear_hi:
            ok.add(k)
    return ok


def _log_axis(ax, numticks: int = 8) -> None:
    """Log y axis with clean multiple ticks; call after plotting (it reads the data range)."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    narrow = math.log10(hi / lo) < 2  # under two decades: add 2x and 5x ticks
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if narrow else (1.0,),
                                          numticks=numticks))
    ax.yaxis.set_major_formatter(FuncFormatter(times))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)  # where every leg starts
    ax.tick_params(which="both", length=0)


def growth_chart(
    curves: pd.DataFrame, path: Path, title: str, subtitle: str, note: str = ""
) -> Path:
    """Growth of 1 by nights, by days, and both, on one log axis."""
    _style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ends = {key: float(curves[key].iloc[-1]) for key, _, _ in SERIES}
    show = labels_that_fit(ends)
    x_end = curves.index[-1]
    for key, label, color in SERIES:
        ax.plot(curves.index, curves[key], color=color, linewidth=1.6, label=label,
                solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.scatter([x_end], [ends[key]], s=30, color=color, edgecolors=SURFACE,
                   linewidths=1.5, zorder=4, clip_on=False)
        if key in show:  # in the right margin, so the axis itself stops at the data
            ax.annotate(f"{label} {multiple(ends[key])}", (x_end, ends[key]),
                        xytext=(7, 0), textcoords="offset points", va="center",
                        color=INK_2, fontsize=8.5, annotation_clip=False)
    _log_axis(ax)
    ax.set_xlim(curves.index[0], x_end)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_2, fontsize=8.5)
    # no y-axis title: the title says "growth of 1", the subtitle "log scale",
    # and wide ticks (÷10,000) would push a rotated title off the canvas
    fig.text(0.08, 0.965, title, fontsize=12, weight="semibold", color=INK, va="top")
    fig.text(0.08, 0.915, subtitle, fontsize=8.5, color=INK_2, va="top")
    if note:
        fig.text(0.08, 0.02, note, fontsize=7.5, color=MUTED, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.8, top=0.84, bottom=0.12 if note else 0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def small_multiples(
    panels: dict[str, pd.DataFrame], path: Path, title: str, subtitle: str, ncols: int = 5
) -> Path:
    """One small panel per series, nights and days only, each on its own log scale."""
    _style()
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, 1.55 * nrows + 0.9), squeeze=False)
    for ax, (name, c) in zip(axes.flat, panels.items()):
        for key, label, color in SERIES[:2]:
            ax.plot(c.index, c[key], color=color, linewidth=1.0, label=label)
        _log_axis(ax, numticks=4)
        ax.tick_params(labelsize=6)
        ax.set_xticks([c.index[0], c.index[-1]])
        ax.set_xticklabels([str(c.index[0].year), str(c.index[-1].year)])
        ax.set_title(name, loc="left", fontsize=8, color=INK, pad=11)
        ax.text(0.0, 1.02, f"nights {multiple(c['night'].iloc[-1])} · days {multiple(c['day'].iloc[-1])}",
                transform=ax.transAxes, fontsize=6, color=MUTED, va="bottom")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2, labelcolor=INK_2, fontsize=8)
    top_in = 0.9  # inches above the grid for title, subtitle and legend
    fig_h = fig.get_size_inches()[1]
    fig.text(0.015, 1 - 0.15 / fig_h, title, fontsize=12, weight="semibold", color=INK, va="top")
    fig.text(0.015, 1 - 0.45 / fig_h, subtitle, fontsize=8, color=INK_2, va="top")
    bottom_in = 0.3  # inches for the last row's year labels, whatever the row count
    fig.subplots_adjust(left=0.04, right=0.99, bottom=bottom_in / fig_h, top=1 - top_in / fig_h,
                        hspace=0.7, wspace=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
