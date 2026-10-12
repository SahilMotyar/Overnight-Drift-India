"""Round-trip costs, worked by hand from the published rates."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, costs


def test_stock_delivery_round_trip_by_hand() -> None:
    """Rs 1 lakh bought at a close, sold next open (BTST = delivery STT both legs)."""
    c = costs.round_trip("stock_delivery", order_value=100_000, slippage_bp_per_side=0)
    assert c["stt"] == pytest.approx(20.0)            # 0.1% + 0.1%
    assert c["stamp"] == pytest.approx(1.5)           # 0.015% on the buy
    assert c["exchange"] == pytest.approx(0.614)      # 2 x 0.00307%
    assert c["sebi"] == pytest.approx(0.02)           # 2 x Rs 10 / crore
    assert c["brokerage"] == 0
    assert c["gst"] == pytest.approx(0.18 * (0.614 + 0.02))
    assert c["dp"] == pytest.approx(1.534)            # Rs 15.34 on Rs 1 lakh
    assert c["total"] == pytest.approx(20 + 1.5 + 0.614 + 0.02 + 0.18 * 0.634 + 1.534)


def test_intraday_brokerage_hits_the_rs_20_cap() -> None:
    """0.03% of Rs 1 lakh is Rs 30, so the Rs 20 cap binds: 2 bp per order."""
    c = costs.round_trip("stock_intraday", order_value=100_000, slippage_bp_per_side=0)
    assert c["stt"] == pytest.approx(2.5) and c["stamp"] == pytest.approx(0.3)
    assert c["brokerage"] == pytest.approx(4.0)
    small = costs.round_trip("stock_intraday", order_value=20_000, slippage_bp_per_side=0)
    assert small["brokerage"] == pytest.approx(6.0)   # 0.03% binds below Rs 66,667


def test_etf_delivery_is_cheap_on_stt() -> None:
    c = costs.round_trip("etf_delivery", order_value=100_000, slippage_bp_per_side=0)
    assert c["stt"] == pytest.approx(0.1)             # 0.001%, seller only
    assert c["total"] < 4


def test_futures_round_trip_after_the_2026_stt_rise() -> None:
    c = costs.round_trip("index_futures", order_value=1_500_000, slippage_bp_per_side=0)
    assert c["stt"] == pytest.approx(5.0)             # 0.05% on the sell
    assert c["brokerage"] == pytest.approx(2 * 20 / 1_500_000 * 1e4)


def test_slippage_counts_both_sides_and_dp_shrinks_with_size() -> None:
    assert costs.round_trip("stock_delivery", slippage_bp_per_side=3)["slippage"] == 6
    big = costs.round_trip("stock_delivery", order_value=1_000_000, slippage_bp_per_side=0)
    assert big["dp"] == pytest.approx(0.1534)


def test_cost_table_has_every_trade_type() -> None:
    t = costs.cost_table()
    assert set(t.index) == set(config.TRADE_TYPES)
    assert (t["total"] > 0).all()
    assert t.loc["stock_delivery", "total"] > t.loc["stock_intraday", "total"] > t.loc["etf_delivery", "total"]


def test_futures_carry_by_night_length() -> None:
    nights = pd.Series([1, 3])
    carry = costs.futures_carry(nights, annual=0.05)
    assert carry.iloc[0] == pytest.approx(0.05 * 17.75 / 8760)   # ~1.0 bp
    assert carry.iloc[1] == pytest.approx(0.05 * 65.75 / 8760)   # ~3.8 bp
    assert costs.day_carry(0.05) == pytest.approx(0.05 * 6.25 / 8760)
