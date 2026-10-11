"""Phase 4 tests on series whose truth is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, significance


def _legs(n: int, night_bp: float, day_bp: float, start: str = "2008-01-01") -> pd.DataFrame:
    g = config.rng("test")
    idx = pd.bdate_range(start, periods=n, name="Date")
    return pd.DataFrame({
        "night": g.normal(night_bp / 1e4, 0.008, n),
        "day": g.normal(day_bp / 1e4, 0.012, n),
    }, index=idx)


def test_period_masks_are_inclusive_and_open_ended() -> None:
    idx = pd.to_datetime(["2010-12-31", "2011-01-01", "2023-06-30", "2023-07-01"])
    m = significance.period_masks(idx)
    assert m["full"].all()
    assert m["pre_auction"].tolist() == [True, False, False, False]
    assert m["2011_2019"].tolist() == [False, True, False, False]
    assert m["covid_retail"].tolist() == [False, False, True, False]
    assert m["gift_nifty"].tolist() == [False, False, False, True]


def test_a_real_gap_is_found_and_its_ci_covers_the_truth() -> None:
    legs = _legs(6_000, night_bp=10, day_bp=-5)
    t = significance.leg_tests(legs, reps=300).set_index("period")
    full = t.loc["full"]
    assert full["night_t"] > 3 and full["day_t"] < -1.5 and full["gap_t"] > 3
    assert full["gap_ci_lo_bp"] < 15 < full["gap_ci_hi_bp"]
    assert full["gap_mean_bp"] == pytest.approx(full["night_mean_bp"] - full["day_mean_bp"])


def test_no_gap_gives_an_insignificant_t() -> None:
    t = significance.leg_tests(_legs(4_000, night_bp=0, day_bp=0), reps=200).set_index("period")
    assert abs(t.loc["full", "gap_t"]) < 3


def test_short_periods_are_not_tested() -> None:
    legs = _legs(300, night_bp=10, day_bp=-5, start="2023-06-01")  # ~20 days before gift_nifty starts
    t = significance.leg_tests(legs, reps=100).set_index("period")
    assert np.isnan(t.loc["covid_retail", "night_t"]) and t.loc["covid_retail", "night_n"] < 250
    assert t.loc["gift_nifty", "night_n"] >= 250 and not np.isnan(t.loc["gift_nifty", "night_t"])


def test_gap_needs_both_legs_on_the_same_date() -> None:
    legs = _legs(5, 10, -5)
    legs.iloc[1, 0] = np.nan
    legs.iloc[3, 1] = np.nan
    assert significance.with_gap(legs)["gap"].isna().tolist() == [False, True, False, True, False]


def test_equal_weight_averages_what_is_there() -> None:
    idx = pd.bdate_range("2020-01-01", periods=2)
    a = pd.DataFrame({"night": [0.01, np.nan], "day": [0.0, 0.02]}, index=idx)
    b = pd.DataFrame({"night": [0.03, 0.04], "day": [0.02, np.nan]}, index=idx)
    ew = significance.equal_weight({"A": a, "B": b})
    assert ew["night"].tolist() == pytest.approx([0.02, 0.04])
    assert ew["day"].tolist() == pytest.approx([0.01, 0.02])


def test_traded_value_ignores_split_adjustment() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)
    raw = pd.DataFrame({"Close": [100.0, 102.0, 101.0], "Volume": [10, 20, 0]}, index=idx)
    split = raw.assign(Close=raw["Close"] / 5, Volume=raw["Volume"] * 5)
    assert significance.traded_value(raw, "2020-01-01") == significance.traded_value(split, "2020-01-01")
    assert significance.traded_value(raw, "2020-01-01") == pytest.approx((1000 + 2040) / 2)  # zero-volume day out


def test_spearman_and_permutation() -> None:
    a = np.arange(30.0)
    assert significance.spearman(a, a ** 3) == pytest.approx(1.0)
    assert significance.spearman(a, -a) == pytest.approx(-1.0)
    assert significance.permutation_p(a, a ** 3, config.rng("crosssection"), perms=2_000) < 0.001
    noise = config.rng("test").normal(size=30)
    assert significance.permutation_p(a, noise, config.rng("crosssection"), perms=2_000) > 0.01


def test_cross_section_drops_missing_rows() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, np.nan, 5], "y": [2, 4, 6, 8, np.nan]})
    out = significance.cross_section(df, "x", "y", config.rng("crosssection"), perms=500)
    assert out["n"] == 3 and out["rho"] == pytest.approx(1.0)
