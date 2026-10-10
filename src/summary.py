"""Phase 3 summary statistics of the night and day legs.

Descriptive only. Whether any of it is more than noise is Phase 4's question;
these functions report sizes, not significance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .decompose import curves as chain

BP = 1e4
NIGHT_KINDS = ("weeknight", "weekend", "holiday")
EXTREME_NOTE = "night beyond +-20%"


def years_spanned(index: pd.DatetimeIndex) -> float:
    return (index[-1] - index[0]).days / 365.25


def leg_stats(r: pd.Series, years: float) -> dict[str, float]:
    """Size of one leg: growth of 1, CAGR, mean / median per return, hit rate, volatility."""
    r = r.dropna()
    if r.empty:
        return {"n": 0}
    growth = float(np.exp(np.log1p(r).sum()))
    return {
        "n": len(r),
        "growth_x": growth,
        "cagr_%": 100 * (growth ** (1 / years) - 1),
        "mean_bp": BP * r.mean(),
        "median_bp": BP * r.median(),
        "up_%": 100 * (r > config.FLAT_TOLERANCE).mean(),
        "flat_%": 100 * (r.abs() <= config.FLAT_TOLERANCE).mean(),
        "vol_%": 100 * r.std() * np.sqrt(len(r) / years),
    }


def summarize(clean: pd.DataFrame, legs: pd.DataFrame) -> dict[str, object]:
    """One row of the Phase 3 summary table."""
    years = years_spanned(clean.index)
    c = chain(clean, legs)
    row: dict[str, object] = {
        "first": clean.index[0].date(),
        "last": clean.index[-1].date(),
        "years": round(years, 1),
    }
    for leg in ("night", "day"):
        row.update({f"{leg}_{k}": v for k, v in leg_stats(legs[leg], years).items()})
    row["split_hold_cagr_%"] = 100 * (c["split_hold"].iloc[-1] ** (1 / years) - 1)
    row["hold_cagr_%"] = 100 * (c["hold"].iloc[-1] ** (1 / years) - 1)
    return row


def night_kinds(clean: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    """Accepted nights by what they span. If the drift paid for bearing risk
    while the market is shut, longer nights should earn more per night."""
    df = pd.DataFrame({"night": legs["night"], "kind": clean["night_kind"], "days": clean["night_days"]}).dropna()
    rows = []
    for kind in NIGHT_KINDS:
        g = df[df["kind"] == kind]
        if g.empty:
            continue
        rows.append({
            "night_kind": kind,
            "n": len(g),
            "calendar_days": round(g["days"].mean(), 2),
            "mean_bp": BP * g["night"].mean(),
            "median_bp": BP * g["night"].median(),
            "bp_per_calendar_day": BP * g["night"].mean() / g["days"].mean(),
        })
    return pd.DataFrame(rows)


def restore_genuine_extremes(
    ticker: str,
    clean: pd.DataFrame,
    legs: pd.DataFrame,
    genuine: tuple[tuple[str, str], ...] = config.GENUINE_EXTREME_NIGHTS,
) -> pd.DataFrame:
    """``legs`` with this ticker's NSE-confirmed +-20% nights put back."""
    out = legs.copy()
    for t, day in genuine:
        if t != ticker:
            continue
        d = pd.Timestamp(day)
        if d not in clean.index or clean.at[d, "night_note"] != EXTREME_NOTE:
            raise ValueError(f"{ticker} {day} is not a night Phase 2 rejected as beyond +-20%")
        prev = clean.index[clean.index.get_loc(d) - 1]
        out.at[d, "night"] = clean.at[d, "Open"] / clean.at[prev, "Close"] - 1
    return out


def sensitivity(ticker: str, clean: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    """The main numbers under each robustness variant Phase 2 set up.

    * ``main``: Phase 2's accepted returns
    * ``no reversed opens``: freak opening prints out (night and day)
    * ``no ex-dividend nights``: so the night premium is not just dividends
    * ``genuine +-20% restored``: the 7 NSE-confirmed big nights put back
    """
    years = years_spanned(clean.index)
    reversed_open = clean["reversed_open"].astype(bool)
    variants = {
        "main": legs,
        "no reversed opens": legs.mask(np.repeat(reversed_open.to_numpy()[:, None], 2, axis=1)),
        "no ex-dividend nights": legs.assign(night=legs["night"].mask(clean["Dividends"] > 0)),
        "genuine +-20% restored": restore_genuine_extremes(ticker, clean, legs),
    }
    rows = []
    for name, v in variants.items():
        n, d = leg_stats(v["night"], years), leg_stats(v["day"], years)
        rows.append({
            "variant": name,
            "nights": n["n"], "night_mean_bp": n["mean_bp"], "night_cagr_%": n["cagr_%"],
            "days": d["n"], "day_mean_bp": d["mean_bp"], "day_cagr_%": d["cagr_%"],
        })
    return pd.DataFrame(rows)


def gap_by_year(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Per year, on dates both have: mean of ``a`` and ``b`` and the a - b gap, in bp.

    Used to compare the ETF's and the stocks' nights with the index's: the
    index is computed from its constituents' opens, so a tracker that opens
    far from it is printing a price the index does not.
    """
    both = pd.DataFrame({"a": a, "b": b}).dropna()
    both["gap"] = both["a"] - both["b"]
    g = both.groupby(both.index.year)
    return pd.DataFrame({
        "n": g.size(),
        "a_mean_bp": BP * g["a"].mean(),
        "b_mean_bp": BP * g["b"].mean(),
        "gap_mean_bp": BP * g["gap"].mean(),
        "gap_median_bp": BP * g["gap"].median(),
    }).rename_axis("year")
