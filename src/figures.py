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


AQUA = "#1baf7a"  # categorical slot 3: 2.74:1 on the surface, so end labels + a table view carry it
STRATEGY_SERIES: tuple[tuple[str, str, str], ...] = (
    ("S0 hold", "Buy and hold", MUTED),
    ("S1 long night", "Long night", NIGHT),
    ("S2 short day", "Short day", DAY),
    ("S3 both", "Both", AQUA),
)


def strategy_panels(
    panels: dict[str, pd.DataFrame], path: Path, title: str, subtitle: str
) -> Path:
    """Equity curves of S0-S3 side by side, one panel per series, each on its own log scale."""
    _style()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for ax, (name, eq) in zip(axes.flat, panels.items()):
        x_end = eq.index[-1]
        ends = {k: float(eq[k].iloc[-1]) for k, _, _ in STRATEGY_SERIES}
        # 4 series over ~1.5-6 decades in a tall panel: a 1.35x gap is ~50 px, room for a label
        show = labels_that_fit(ends, min_ratio=1.35)
        for key, label, color in STRATEGY_SERIES:
            ax.plot(eq.index, eq[key], color=color, linewidth=1.4, label=label,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3)
            ax.scatter([x_end], [ends[key]], s=24, color=color, edgecolors=SURFACE,
                       linewidths=1.5, zorder=4, clip_on=False)
            if key in show:
                ax.annotate(f"{label} {multiple(ends[key])}", (x_end, ends[key]), xytext=(6, 0),
                            textcoords="offset points", va="center", color=INK_2, fontsize=7.5,
                            annotation_clip=False)
        _log_axis(ax, numticks=6)
        ax.set_xlim(eq.index[0], x_end)
        ax.set_title(name, loc="left", fontsize=9.5, color=INK, pad=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # its own row under the subtitle, so a long title never runs into it
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.012, 0.875), frameon=False,
               ncol=4, labelcolor=INK_2, fontsize=8)
    fig.text(0.02, 0.97, title, fontsize=12, weight="semibold", color=INK, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=8, color=INK_2, va="top")
    fig.subplots_adjust(left=0.06, right=0.87, top=0.76, bottom=0.08, wspace=0.55)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def edge_vs_cost(
    rows: pd.DataFrame, path: Path, title: str, subtitle: str, ncols: int = 2
) -> Path:
    """Gross edge per round trip by period (bars) against today's cost (a reference line).

    ``rows`` has columns ``panel``, ``period``, ``edge_bp`` and ``cost_bp``,
    one row per bar, periods in reading order. One x scale for every panel, so
    the panels compare.
    """
    _style()
    panels = list(dict.fromkeys(rows["panel"]))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.1 * nrows + 1.0), squeeze=False)
    lo = min(0.0, float(rows["edge_bp"].min()))
    hi = float(max(rows["edge_bp"].max(), rows["cost_bp"].max()))
    pad = 0.08 * (hi - lo)
    for ax, name in zip(axes.flat, panels):
        g = rows[rows["panel"] == name].reset_index(drop=True)
        y = list(range(len(g)))[::-1]  # first period on top
        ax.barh(y, g["edge_bp"], height=0.55, color=NIGHT, zorder=3, label="Gross edge per round trip")
        cost = float(g["cost_bp"].iloc[0])
        ax.axvline(cost, color=DAY, linewidth=2, zorder=4, label="Today's cost per round trip")
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=2)
        for yi, v in zip(y, g["edge_bp"]):
            # a label at the bar end would run into the cost line just past it, or (for a
            # negative bar) into the period names on the left: move it right of either
            if v < 0:
                at = 0.0
            elif v < cost and cost - v < 0.09 * (hi - lo):
                at = cost
            else:
                at = v
            ax.annotate(f"{v:.1f}", (at, yi), xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", color=INK_2, fontsize=7.5)
        ax.annotate(f"cost {cost:.1f}", (cost, len(g) - 0.45), xytext=(4, 0), textcoords="offset points",
                    va="bottom", color=INK_2, fontsize=7.5, annotation_clip=False)
        ax.set_yticks(y)
        ax.set_yticklabels(g["period"], fontsize=8, color=INK_2)
        ax.set_xlim(lo - pad, hi + 3 * pad)
        ax.set_ylim(-0.6, len(g) - 0.1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0, labelsize=7.5)
        ax.set_title(name, loc="left", fontsize=9.5, color=INK, pad=14)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: not labels[i].startswith("Gross"))  # bars first
    fig_h = fig.get_size_inches()[1]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.012, 1 - 0.6 / fig_h), frameon=False, ncol=2, labelcolor=INK_2, fontsize=8)
    fig.text(0.02, 1 - 0.12 / fig_h, title, fontsize=12, weight="semibold", color=INK, va="top")
    fig.text(0.02, 1 - 0.4 / fig_h, subtitle, fontsize=8, color=INK_2, va="top")
    fig.subplots_adjust(left=0.13, right=0.97, top=1 - 1.35 / fig_h, bottom=0.35 / fig_h,
                        hspace=0.75, wspace=0.45)
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
