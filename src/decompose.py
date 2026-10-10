"""Phase 3: split each series into its nights and its days, and chain each.

For trading day t, on dividend-adjusted prices (src.data):

    night_t = Open_t / Close_{t-1} - 1          close -> next open
    day_t   = Close_t / Open_t - 1              open -> close
    (1 + night_t) * (1 + day_t) = Close_t / Close_{t-1}

Only returns Phase 2 accepted (``night_ok`` / ``day_ok``) are chained. A
rejected return adds nothing to its leg, so the two chained legs multiply to
``split_hold`` exactly. ``hold`` is plain buy-and-hold from the first open.
The gap between ``hold`` and ``split_hold`` is what the rejected returns
carried: mostly nights across sessions the series lacks, which contain a
whole trading day and so belong to neither leg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import cache_path

CURVE_COLUMNS = ["night", "day", "split_hold", "hold"]


def load_clean(ticker: str, clean_dir: Path = config.DATA_CLEAN) -> pd.DataFrame:
    """A Phase 2 output frame. Phase 3 onward reads only these."""
    path = cache_path(ticker, clean_dir)
    if not path.exists():
        raise FileNotFoundError(f"no clean data for {ticker!r} at {path}; run `python -m scripts.phase2_validate`")
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])


def leg_returns(clean: pd.DataFrame) -> pd.DataFrame:
    """Simple ``night`` and ``day`` returns; NaN wherever Phase 2 rejected them."""
    night = clean["Open"] / clean["Close"].shift(1) - 1
    day = clean["Close"] / clean["Open"] - 1
    return pd.DataFrame({
        "night": night.where(clean["night_ok"].astype(bool)),
        "day": day.where(clean["day_ok"].astype(bool)),
    })


def identity_error(clean: pd.DataFrame) -> float:
    """Largest ``|(1 + night)(1 + day) - Close / prev Close|`` over days with both legs accepted.

    The Open cancels, so for legs taken from one frame this is zero up to
    float noise by construction. It guards the implementation: a change that
    computed the two legs from differently filtered rows would break it.
    """
    legs = leg_returns(clean)
    both = legs.notna().all(axis=1)
    lhs = (1 + legs["night"]) * (1 + legs["day"])
    rhs = clean["Close"] / clean["Close"].shift(1)
    return float((lhs - rhs)[both].abs().max()) if both.any() else 0.0


def curves(clean: pd.DataFrame, legs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Growth of 1 in each leg, plus the two holds.

    * ``night``, ``day``: chained accepted returns; flat where rejected
    * ``split_hold``: ``night * day``, exactly
    * ``hold``: ``Close / first Open``, everything included
    """
    legs = leg_returns(clean) if legs is None else legs
    log_night = np.log1p(legs["night"]).fillna(0.0).cumsum()
    log_day = np.log1p(legs["day"]).fillna(0.0).cumsum()
    return pd.DataFrame({
        "night": np.exp(log_night),
        "day": np.exp(log_day),
        "split_hold": np.exp(log_night + log_day),
        "hold": clean["Close"] / clean["Open"].iloc[0],
    })[CURVE_COLUMNS]
