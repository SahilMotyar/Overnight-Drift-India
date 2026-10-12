"""Phase 5: what one round trip (a buy and a sell of the same value) costs in India.

All rates live in ``config.TRADE_TYPES`` with their sources; this module only
does the arithmetic. Per side, on traded value V:

* STT: ``stt_buy`` on the buy, ``stt_sell`` on the sell
* stamp duty: ``stamp_buy``, buy side only
* exchange transaction charge and SEBI fee: both sides
* brokerage: ``min(brokerage_pct x V, brokerage_cap)`` per order
* GST: 18% of (brokerage + exchange charge + SEBI fee)
* DP charge: a flat rupee amount when shares leave demat (delivery sells)
* slippage: an assumed bp per side (``config.SLIPPAGE_BP_PER_SIDE``)

Futures also lose (long) or earn (short) the cost of carry while held; that is
a property of the instrument's return, not a fee, so it is handled by
:func:`futures_carry` and applied to returns in src/breakeven.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

BP = 1e4
COMPONENTS = ["stt", "stamp", "exchange", "sebi", "brokerage", "gst", "dp", "slippage"]


def round_trip(
    trade_type: str,
    order_value: float | None = None,
    slippage_bp_per_side: float | None = None,
) -> dict[str, float]:
    """Cost of one round trip by component, in bp of the traded value, plus ``total``."""
    r = config.TRADE_TYPES[trade_type]
    v = config.ORDER_VALUE_INR[trade_type] if order_value is None else order_value
    slip = config.SLIPPAGE_BP_PER_SIDE[trade_type] if slippage_bp_per_side is None else slippage_bp_per_side

    brokerage_per_order = min(r["brokerage_pct"] * v, r["brokerage_cap"]) / v if r["brokerage_pct"] else 0.0
    exchange = 2 * r["exchange"]
    sebi = 2 * config.SEBI_FEE
    brokerage = 2 * brokerage_per_order
    parts = {
        "stt": r["stt_buy"] + r["stt_sell"],
        "stamp": r["stamp_buy"],
        "exchange": exchange,
        "sebi": sebi,
        "brokerage": brokerage,
        "gst": config.GST * (brokerage + exchange + sebi),
        "dp": r["dp_per_sell"] / v,
    }
    out = {k: BP * x for k, x in parts.items()}
    out["slippage"] = 2 * slip
    out["total"] = sum(out[k] for k in COMPONENTS)
    return out


def cost_table(order_values: dict[str, float] | None = None) -> pd.DataFrame:
    """Every trade type's round-trip cost by component (bp), at the configured order values."""
    rows = {t: round_trip(t, None if order_values is None else order_values.get(t))
            for t in config.TRADE_TYPES}
    return pd.DataFrame.from_dict(rows, orient="index")[[*COMPONENTS, "total"]].rename_axis("trade_type")


def futures_carry(night_days: pd.Series, annual: float = config.FUTURES_CARRY_ANNUAL) -> pd.Series:
    """Carry a long futures position gives up over each night, as a return (positive = cost).

    A night spans ``night_days * 24 - 6.25`` hours (close to next open):
    17.75 h on a weeknight, 65.75 h over a weekend.
    """
    hours = np.asarray(night_days, dtype=float) * 24 - config.SESSION_HOURS
    return pd.Series(annual * hours / (365 * 24), index=night_days.index)


def day_carry(annual: float = config.FUTURES_CARRY_ANNUAL) -> float:
    """Carry a short futures position earns over one session, as a return."""
    return annual * config.SESSION_HOURS / (365 * 24)
