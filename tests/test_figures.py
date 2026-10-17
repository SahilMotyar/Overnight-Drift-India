"""Chart helpers: label formats, label collisions, and that the files render."""

from __future__ import annotations

import pandas as pd

from src import decompose, figures
from tests.test_decompose import _clean


def test_compact_and_tick_formats() -> None:
    assert figures.compact(0.0749) == "0.075"
    assert figures.compact(55.54) == "55.5"
    assert figures.compact(14543.1) == "14,543"
    assert figures.compact(359_090_601) == "359M"
    assert figures.compact(1.81e10) == "18.1B"
    assert figures.times(1e10) == "×10B"
    assert figures.times(1e-9) == "÷1B"
    assert figures.times(0.1) == "×0.1"
    assert figures.times(0.001) == "÷1,000"
    assert figures.times(1e-7) == "÷10M"
    assert figures.times(1000) == "×1,000"
    assert figures.times(1e8) == "×100M"
    assert figures.multiple(0.075) == "×0.075"
    assert figures.multiple(3.2e-7) == "÷3.12M"
    assert figures.multiple(0.00052) == "÷1,923"


def test_converging_end_labels_are_left_to_the_legend() -> None:
    assert figures.labels_that_fit({"night": 55.5, "day": 0.075, "split_hold": 4.16}) == {
        "night", "day", "split_hold"}
    # night and both end close together: neither gets a label, day still does
    assert figures.labels_that_fit({"night": 5.0, "day": 0.2, "split_hold": 4.0}) == {"day"}


def test_charts_render(tmp_path) -> None:
    c = decompose.curves(_clean())
    p = figures.growth_chart(c, tmp_path / "g.png", "Title", "Subtitle", note="Note")
    assert p.exists() and p.stat().st_size > 10_000
    panels = {f"S{i}": c for i in range(7)}
    q = figures.small_multiples(panels, tmp_path / "m.png", "Title", "Subtitle", ncols=4)
    assert q.exists() and q.stat().st_size > 10_000


def test_strategy_panels_render(tmp_path) -> None:
    c = decompose.curves(_clean())
    eq = pd.DataFrame({"S0 hold": c["hold"], "S1 long night": c["night"],
                       "S2 short day": 1 / c["day"], "S3 both": c["night"] / c["day"]})
    p = figures.strategy_panels({"A": eq, "B": eq}, tmp_path / "s.png", "Title", "Subtitle")
    assert p.exists() and p.stat().st_size > 10_000


def test_edge_vs_cost_renders_with_a_negative_edge(tmp_path) -> None:
    rows = pd.DataFrame({
        "panel": ["A", "A", "B", "B"],
        "period": ["early", "late", "early", "late"],
        "edge_bp": [8.9, -2.2, 26.9, 10.4],
        "cost_bp": [8.0, 8.0, 27.8, 27.8],
    })
    p = figures.edge_vs_cost(rows, tmp_path / "e.png", "Title", "Subtitle")
    assert p.exists() and p.stat().st_size > 10_000


def test_growth_chart_survives_a_flat_leg(tmp_path) -> None:
    c = decompose.curves(_clean())
    c["day"] = 1.0
    figures.growth_chart(c, tmp_path / "flat.png", "T", "S")
    assert isinstance(c.index, pd.DatetimeIndex)
