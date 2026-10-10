"""Summary statistics on series whose answers are known in advance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import decompose, summary


def _clean(night: list[float], day: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build prices from chosen returns; row 0's night is 'first' as in Phase 2."""
    n = len(night)
    close, open_ = np.empty(n), np.empty(n)
    prev = 100.0
    for i in range(n):
        open_[i] = prev * (1 + night[i])
        close[i] = open_[i] * (1 + day[i])
        prev = close[i]
    idx = pd.bdate_range(start, periods=n, name="Date")
    return pd.DataFrame({
        "Open": open_, "Close": close, "Dividends": 0.0,
        "night_ok": [False] + [True] * (n - 1), "day_ok": True,
        "night_note": [""] * n, "reversed_open": False,
        "night_kind": ["first"] + ["weeknight"] * (n - 1), "night_days": [0] + [1] * (n - 1),
    }, index=idx)


def test_leg_stats_known_values() -> None:
    r = pd.Series([0.01, 0.0, -0.01, 0.02, np.nan])
    s = summary.leg_stats(r, years=1.0)
    assert s["n"] == 4
    assert s["mean_bp"] == pytest.approx(50.0)
    assert s["median_bp"] == pytest.approx(50.0)
    assert s["up_%"] == pytest.approx(50.0) and s["flat_%"] == pytest.approx(25.0)
    assert s["growth_x"] == pytest.approx(1.01 * 0.99 * 1.02)
    assert s["cagr_%"] == pytest.approx(100 * (1.01 * 0.99 * 1.02 - 1))


def test_float_noise_counts_as_flat() -> None:
    s = summary.leg_stats(pd.Series([1e-8, -3e-7, 0.01]), years=1.0)
    assert s["flat_%"] == pytest.approx(200 / 3)


def test_summarize_reports_both_legs_and_both_holds() -> None:
    clean = _clean([0.0] + [0.001] * 251, [-0.0005] * 252)
    row = summary.summarize(clean, decompose.leg_returns(clean))
    assert row["night_mean_bp"] == pytest.approx(10.0)
    assert row["day_mean_bp"] == pytest.approx(-5.0)
    # nothing rejected but the first night (which is 0 here): the two holds agree
    assert row["split_hold_cagr_%"] == pytest.approx(row["hold_cagr_%"])


def test_night_kinds_per_calendar_day() -> None:
    clean = _clean([0.0, 0.001, 0.003, 0.001], [0.0] * 4)
    clean["night_kind"] = ["first", "weeknight", "weekend", "weeknight"]
    clean["night_days"] = [0, 1, 3, 1]
    t = summary.night_kinds(clean, decompose.leg_returns(clean)).set_index("night_kind")
    assert t.loc["weeknight", "n"] == 2 and t.loc["weekend", "n"] == 1
    assert t.loc["weekend", "bp_per_calendar_day"] == pytest.approx(10.0)
    assert t.loc["weeknight", "bp_per_calendar_day"] == pytest.approx(10.0)


def test_sensitivity_variants_change_only_what_they_name() -> None:
    clean = _clean([0.0, 0.01, 0.02, 0.01, 0.01], [0.0, 0.0, -0.02, 0.0, 0.0])
    clean.loc[clean.index[2], "reversed_open"] = True
    clean.loc[clean.index[3], "Dividends"] = 1.0
    t = summary.sensitivity("X.NS", clean, decompose.leg_returns(clean)).set_index("variant")
    assert t.loc["main", "nights"] == 4
    assert t.loc["no reversed opens", "nights"] == 3 and t.loc["no reversed opens", "days"] == 4
    assert t.loc["no ex-dividend nights", "nights"] == 3 and t.loc["no ex-dividend nights", "days"] == 5
    assert t.loc["genuine +-20% restored"].equals(t.loc["main"])  # X.NS has none


def test_restoring_a_genuine_extreme_uses_the_real_night() -> None:
    clean = _clean([0.0, 0.01, 0.22, 0.01], [0.0] * 4)
    d = clean.index[2]
    clean.loc[d, ["night_ok", "night_note"]] = [False, summary.EXTREME_NOTE]
    legs = decompose.leg_returns(clean)
    out = summary.restore_genuine_extremes("X.NS", clean, legs, genuine=(("X.NS", str(d.date())),))
    assert np.isnan(legs.at[d, "night"]) and out.at[d, "night"] == pytest.approx(0.22)
    with pytest.raises(ValueError, match="not a night Phase 2 rejected"):
        summary.restore_genuine_extremes("X.NS", clean, legs, genuine=(("X.NS", str(clean.index[1].date())),))


def test_gap_by_year_on_common_dates_only() -> None:
    idx = pd.to_datetime(["2020-01-02", "2020-06-01", "2021-01-04", "2021-02-01"])
    a = pd.Series([0.002, 0.004, 0.001, np.nan], index=idx)
    b = pd.Series([0.001, 0.001, 0.001, 0.005], index=idx)
    g = summary.gap_by_year(a, b)
    assert g.loc[2020, "n"] == 2 and g.loc[2020, "gap_mean_bp"] == pytest.approx(20.0)
    assert g.loc[2021, "n"] == 1 and g.loc[2021, "gap_mean_bp"] == pytest.approx(0.0)
