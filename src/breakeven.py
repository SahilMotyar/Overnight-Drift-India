"""Phase 5: how much a round trip could cost before each strategy stops paying.

Two strategies, one round trip per session:

* **long night**: buy at the close, sell at the next open. Gross return = the night.
* **short day**: short at the open, cover at the close. Gross return = -day
  exactly: (Open - Close) / Open.

Each series is traded the way it can be: stocks and the ETF in the cash
market (delivery for the night, intraday for the day), indices through index
futures, whose overnight return also loses the cost of carry (src.costs).

With a round-trip cost c, each session nets ``r_t - c`` and the strategy
compounds those. Break-evens, in bp per round trip:

* ``be_mean_bp``: the c at which the mean net return is zero (the mean gross)
* ``be_cagr_bp``: the c at which the strategy stops growing money (net CAGR 0)
* ``be_hold_bp``: the c at which its CAGR falls to buy-and-hold's over the
  same period (NaN if it never beats buy-and-hold even at zero cost)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import costs

BP = 1e4
STRATEGIES: dict[str, tuple[str, int]] = {"long night": ("night", +1), "short day": ("day", -1)}

# How each kind of series is traded, per strategy.
TRADE_TYPE: dict[str, dict[str, str]] = {
    "index": {"long night": "index_futures", "short day": "index_futures"},
    "etf": {"long night": "etf_delivery", "short day": "etf_intraday"},
    "stock": {"long night": "stock_delivery", "short day": "stock_intraday"},
    "portfolio": {"long night": "stock_delivery", "short day": "stock_intraday"},
}


def strategy_returns(
    legs: pd.DataFrame, kind: str, strategy: str, night_days: pd.Series | None = None
) -> pd.Series:
    """Gross return per round trip, before trading costs; NaN rows dropped.

    For an index (traded as futures) the long night gives up the carry over
    each night's length and the short day earns one session of it.
    """
    leg, sign = STRATEGIES[strategy]
    r = sign * legs[leg]
    if kind == "index":
        if strategy == "long night":
            if night_days is None:
                raise ValueError("futures carry needs each night's length (night_days)")
            r = r - costs.futures_carry(night_days.reindex(r.index))
        else:
            r = r + costs.day_carry()
    return r.dropna()


def net_cagr(r: np.ndarray, cost: float, years: float) -> float:
    """CAGR of compounding ``r - cost`` every round trip; -100% if any session wipes it out."""
    net = np.asarray(r, dtype=float) - cost
    if (net <= -1).any():
        return -1.0
    return float(np.exp(np.log1p(net).sum() / years) - 1)


def solve_cost(r: np.ndarray, years: float, target_cagr: float, hi: float = 0.05) -> float:
    """The round-trip cost at which net CAGR equals ``target_cagr`` (bisection; net CAGR
    falls as cost rises). NaN if the strategy is below the target even at zero cost."""
    if net_cagr(r, 0.0, years) < target_cagr:
        return float("nan")
    lo = 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if net_cagr(r, mid, years) > target_cagr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def breakeven(r: pd.Series, years: float, hold_cagr: float | None, cost_bp: float) -> dict[str, float]:
    """Break-evens for one strategy on one series, and what is left at ``cost_bp``."""
    x = r.to_numpy()
    c = cost_bp / BP
    out = {
        "round_trips": len(x),
        "per_year": len(x) / years,
        "gross_mean_bp": BP * x.mean(),
        "gross_cagr_%": 100 * net_cagr(x, 0.0, years),
        "be_mean_bp": BP * x.mean(),
        "be_cagr_bp": BP * solve_cost(x, years, 0.0),
        "be_hold_bp": BP * solve_cost(x, years, hold_cagr) if hold_cagr is not None else np.nan,
        "cost_bp": cost_bp,
        "cost_drag_%_per_year": cost_bp * len(x) / years / 100,
        "net_mean_bp": BP * (x.mean() - c),
        "net_cagr_%": 100 * net_cagr(x, c, years),
        "hold_cagr_%": 100 * hold_cagr if hold_cagr is not None else np.nan,
    }
    out["room_bp"] = out["be_cagr_bp"] - cost_bp
    return out
