"""Equity curves, strategy combination, stats and the SIP timing tilt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest


def _idx(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, name="Date")


def test_equity_is_flat_on_missing_returns() -> None:
    r = pd.Series([0.1, np.nan, -0.5, 0.2], index=_idx(4))
    assert backtest.equity(r).tolist() == pytest.approx([1.1, 1.1, 0.55, 0.66])


def test_combine_compounds_night_then_day() -> None:
    idx = _idx(3)
    night = pd.Series([0.01, np.nan, 0.02], index=idx)
    day = pd.Series([0.03, np.nan, np.nan], index=idx)
    c = backtest.combine(night, day)
    assert c.iloc[0] == pytest.approx(1.01 * 1.03 - 1)
    assert np.isnan(c.iloc[1]) and c.iloc[2] == pytest.approx(0.02)


def test_net_table_charges_each_round_trip() -> None:
    idx = _idx(4)
    legs = pd.DataFrame({"night": [np.nan, 0.003, 0.003, 0.003], "day": [-0.002] * 4}, index=idx)
    hold = (1 + legs["night"].fillna(0)) * (1 + legs["day"]) - 1
    cost = {t: 10.0 for t in backtest.config.TRADE_TYPES}  # 10 bp per round trip, any trade
    gross = backtest.strategy_returns_table(legs, hold, "stock", net=False, trade_cost_bp=cost)
    net = backtest.strategy_returns_table(legs, hold, "stock", net=True, trade_cost_bp=cost)
    assert net["S1 long night"].iloc[1] == pytest.approx(0.003 - 0.001)
    assert net["S2 short day"].iloc[1] == pytest.approx(0.002 - 0.001)
    assert net["S3 both"].iloc[1] == pytest.approx((1 + 0.002) * (1 + 0.001) - 1)
    assert gross["S0 hold"].equals(net["S0 hold"])  # buy-and-hold pays no daily round trips


def test_stats_known_path() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    r = pd.Series([0.0, -0.5, 1.0], index=idx)        # 1 -> 0.5 -> 1.0
    s = backtest.stats(r, idx[0], idx[-1])
    assert s["final_x"] == pytest.approx(1.0)
    assert s["max_drawdown_%"] == pytest.approx(-50.0)


def test_sip_timing_with_a_known_night_and_day() -> None:
    """Every night +1%, every day -1%: the close is the cheapest of the three."""
    n = 60
    idx = _idx(n)
    open_, close = np.empty(n), np.empty(n)
    prev = 100.0
    for i in range(n):
        open_[i] = prev * 1.01
        close[i] = open_[i] * 0.99
        prev = close[i]
    clean = pd.DataFrame({"Open": open_, "Close": close, "day_ok": True, "night_ok": True}, index=idx)
    dates = backtest.sip_dates(clean)
    assert len(dates) == 3                             # Jan, Feb, Mar 2020
    out = backtest.sip(clean, dates)
    assert out["edge_vs_open_bp"] == pytest.approx(1e4 * (1 / 0.99 - 1))
    assert out["edge_vs_next_open_bp"] == pytest.approx(100.0)
    assert out["value_close"] > out["value_open"] and out["value_close"] > out["value_next open"]


def test_sip_skips_untrusted_sessions() -> None:
    idx = _idx(30)
    clean = pd.DataFrame({"Open": 100.0, "Close": 100.0, "day_ok": True, "night_ok": True}, index=idx)
    clean.loc[idx[0], "day_ok"] = False               # first session of January not trusted
    assert backtest.sip_dates(clean)[0] == idx[1]
