"""Phase 6: equity curves for the night/day trades after costs, and the free timing tilt.

Strategies on one series (returns and trade types from src.breakeven, costs
from src.costs):

* **S0** buy-and-hold: close to close across consecutive sessions
* **S1** long night: the night, less one round trip per night
* **S2** short day: minus the day, less one round trip per day
* **S3** both: S1 then S2 every session, so two round trips a day

A session where a strategy has no accepted return leaves it flat, as in
Phase 3. ``r - c`` is the net return of a round trip costing ``c``.

**S4**, the timing tilt: a monthly SIP that buys on the same sessions at the
open, at the close, or at the next session's open (an order placed after the
close, e.g. an after-market order). Same trades, same costs; only the timing
differs, so whatever it is worth, it is free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, costs
from .breakeven import TRADE_TYPE, strategy_returns

BP = 1e4
STRATEGY_COLUMNS = ["S0 hold", "S1 long night", "S2 short day", "S3 both"]


def equity(r: pd.Series) -> pd.Series:
    """Growth of 1; a missing return leaves the equity where it was."""
    return (1 + r.fillna(0.0)).cumprod()


def combine(first: pd.Series, second: pd.Series) -> pd.Series:
    """Two returns taken back to back on the same date (night, then day)."""
    idx = first.index.union(second.index)
    a, b = first.reindex(idx), second.reindex(idx)
    both_missing = a.isna() & b.isna()
    return ((1 + a.fillna(0.0)) * (1 + b.fillna(0.0)) - 1).mask(both_missing)


def strategy_returns_table(
    legs: pd.DataFrame,
    hold: pd.Series,
    kind: str,
    night_days: pd.Series | None = None,
    net: bool = True,
    trade_cost_bp: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-date returns of S0-S3 for one series, after costs if ``net``."""
    cost = trade_cost_bp or {t: costs.round_trip(t)["total"] for t in config.TRADE_TYPES}
    out = {"S0 hold": hold}
    legs_by_strategy = {}
    for strategy, column in (("long night", "S1 long night"), ("short day", "S2 short day")):
        r = strategy_returns(legs, kind, strategy, night_days=night_days)
        if net:
            r = r - cost[TRADE_TYPE[kind][strategy]] / BP
        legs_by_strategy[strategy] = r
        out[column] = r
    out["S3 both"] = combine(legs_by_strategy["long night"], legs_by_strategy["short day"])
    idx = hold.index
    for r in out.values():
        idx = idx.union(r.index)
    return pd.DataFrame({k: v.reindex(idx) for k, v in out.items()})[STRATEGY_COLUMNS]


def stats(r: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    """CAGR, volatility, return / volatility (risk-free rate 0), max drawdown, final multiple."""
    x = r.dropna()
    years = (end - start).days / 365.25
    eq = equity(r.loc[start:end])
    per_year = len(x) / years if years > 0 else np.nan
    vol = x.std() * np.sqrt(per_year)
    return {
        "cagr_%": 100 * (eq.iloc[-1] ** (1 / years) - 1),
        "vol_%": 100 * vol,
        "ret_per_vol": (x.mean() * per_year) / vol if vol > 0 else np.nan,
        "max_drawdown_%": 100 * float((eq / eq.cummax() - 1).min()),
        "final_x": float(eq.iloc[-1]),
    }


SIP_TIMINGS = ("open", "close", "next open")


def sip_dates(clean: pd.DataFrame) -> pd.DatetimeIndex:
    """First session of each month whose open and close are both trusted (``day_ok``)
    and whose next session's open is trusted too (``night_ok`` on the next row)."""
    ok = clean["day_ok"].astype(bool) & clean["night_ok"].astype(bool).shift(-1, fill_value=False)
    days = clean.index[ok.to_numpy()]
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby([days.year, days.month]).first().to_numpy())


def sip(clean: pd.DataFrame, dates: pd.DatetimeIndex, amount: float = 1.0) -> dict[str, float]:
    """Final value of buying ``amount`` on each of ``dates`` at each timing, valued at the last close.

    Dividend-adjusted prices, so dividends are reinvested. Also returns the
    mean per-instalment edge of buying at the close: ``open / close - 1`` (vs
    that morning's open) and ``next open / close - 1`` (vs the order that
    waits for the next morning), in bp.
    """
    pos = clean.index.get_indexer(dates)
    nxt = clean.index[pos + 1]
    price = {
        "open": clean["Open"].to_numpy()[pos],
        "close": clean["Close"].to_numpy()[pos],
        "next open": clean.loc[nxt, "Open"].to_numpy(),
    }
    last_close = float(clean["Close"].iloc[-1])
    out: dict[str, float] = {"instalments": len(dates), "invested": amount * len(dates)}
    for t in SIP_TIMINGS:
        out[f"value_{t}"] = float((amount / price[t]).sum() * last_close)
    out["close_vs_open_%"] = 100 * (out["value_close"] / out["value_open"] - 1)
    out["close_vs_next_open_%"] = 100 * (out["value_close"] / out["value_next open"] - 1)
    out["edge_vs_open_bp"] = BP * float(np.mean(price["open"] / price["close"] - 1))
    out["edge_vs_next_open_bp"] = BP * float(np.mean(price["next open"] / price["close"] - 1))
    return out
