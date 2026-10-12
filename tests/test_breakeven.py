"""Break-even solvers on returns whose answers are known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import breakeven, costs
from src.config import rng


def test_constant_edge_breaks_even_at_the_edge() -> None:
    """10 bp every round trip: at a 10 bp cost the strategy neither gains nor loses."""
    r = pd.Series(np.full(500, 0.001))
    out = breakeven.breakeven(r, years=2.0, hold_cagr=None, cost_bp=4.0)
    assert out["be_mean_bp"] == pytest.approx(10.0)
    assert out["be_cagr_bp"] == pytest.approx(10.0, abs=1e-6)
    assert out["room_bp"] == pytest.approx(6.0, abs=1e-6)
    assert out["net_cagr_%"] == pytest.approx(100 * (1.0006 ** 250 - 1))
    assert out["cost_drag_%_per_year"] == pytest.approx(4.0 * 250 / 100)


def test_beating_buy_and_hold_needs_a_cheaper_trade_than_breaking_even() -> None:
    r = np.full(250, 0.001)                      # +28.4%/yr gross over 1 year
    be_zero = breakeven.solve_cost(r, 1.0, 0.0)
    be_hold = breakeven.solve_cost(r, 1.0, 0.10)  # buy-and-hold does 10%
    assert be_hold < be_zero
    assert breakeven.net_cagr(r, be_hold, 1.0) == pytest.approx(0.10, abs=1e-9)


def test_never_beating_hold_is_nan() -> None:
    r = np.full(250, 0.0001)                     # ~2.5%/yr gross
    assert np.isnan(breakeven.solve_cost(r, 1.0, 0.10))


def test_a_wipeout_session_is_minus_100_percent() -> None:
    assert breakeven.net_cagr(np.array([0.01, -0.995]), 0.01, 1.0) == -1.0


def test_volatility_drags_cagr_below_the_mean_edge() -> None:
    """Same 10 bp mean, but noisy: compounding makes the CAGR break-even lower."""
    r = pd.Series(0.001 + rng("test").normal(0, 0.01, 5_000))
    out = breakeven.breakeven(r, years=20.0, hold_cagr=None, cost_bp=0.0)
    assert out["be_cagr_bp"] < out["be_mean_bp"]


def test_short_day_is_minus_the_day_and_futures_pay_carry() -> None:
    idx = pd.bdate_range("2026-01-05", periods=3)
    legs = pd.DataFrame({"night": [np.nan, 0.002, 0.003], "day": [-0.001, 0.004, np.nan]}, index=idx)
    nights = pd.Series([0, 1, 3], index=idx)
    stock_short = breakeven.strategy_returns(legs, "stock", "short day")
    assert stock_short.tolist() == pytest.approx([0.001, -0.004])
    fut_long = breakeven.strategy_returns(legs, "index", "long night", night_days=nights)
    assert fut_long.tolist() == pytest.approx(
        [0.002 - costs.futures_carry(nights).iloc[1], 0.003 - costs.futures_carry(nights).iloc[2]])
    fut_short = breakeven.strategy_returns(legs, "index", "short day")
    assert fut_short.iloc[0] == pytest.approx(0.001 + costs.day_carry())
    with pytest.raises(ValueError, match="night_days"):
        breakeven.strategy_returns(legs, "index", "long night")


def test_every_kind_has_a_trade_type_for_both_strategies() -> None:
    for kind, m in breakeven.TRADE_TYPE.items():
        assert set(m) == set(breakeven.STRATEGIES), kind
