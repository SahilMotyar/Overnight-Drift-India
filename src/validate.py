"""Phase 2 checks on one price series.

Each function takes a frame of consecutive rows (so "previous row" means the
row before it in that frame) and returns a boolean mask. None of them drops or
edits anything: src/clean.py applies them in order and records every removed
row and every rejected return in the drop table.

Two kinds of check:

* **row checks**: the row is wrong as a whole and is removed. Its own night
  and day go with it, and the next row's night is then flagged as spanning a
  missing session.
* **return checks**: the prices stay, but the row's night and/or day return
  is marked unusable.

All checks run on Yahoo's raw (split-adjusted, not dividend-adjusted) prices,
so a placeholder row still equals the previous close exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
_RTOL = 1e-6  # float noise in Yahoo's split-adjusted prices


def _close(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(np.isclose(a, b, rtol=_RTOL, atol=0.0), index=a.index)


def _flat(df: pd.DataFrame) -> pd.Series:
    """Open == High == Low == Close."""
    return _close(df["Open"], df["High"]) & _close(df["High"], df["Low"]) & _close(df["Low"], df["Close"])


# --------------------------------------------------------------------------
# Row checks
# --------------------------------------------------------------------------

def nonpositive(df: pd.DataFrame) -> pd.Series:
    """Any price <= 0 (e.g. NIFTYBEES opened at 0.0 on 2009-07-07)."""
    return (df[PRICE_COLUMNS] <= 0).any(axis=1)


def ohlc_inconsistent(df: pd.DataFrame) -> pd.Series:
    """Open or Close outside [Low, High], or High below Low."""
    hi = df["High"] * (1 + _RTOL)
    lo = df["Low"] * (1 - _RTOL)
    return (
        (df["Open"] > hi) | (df["Open"] < lo)
        | (df["Close"] > hi) | (df["Close"] < lo)
        | (df["High"] < lo)
    )


def placeholder(df: pd.DataFrame) -> pd.Series:
    """Yahoo's filler for a day it has no data: flat bar at the previous close, zero volume."""
    return _flat(df) & _close(df["Close"], df["Close"].shift(1)) & (df["Volume"] == 0)


def no_trades(df: pd.DataFrame) -> pd.Series:
    """Flat bar with zero volume at any price: nothing traded, so no real open or close.

    Superset of :func:`placeholder`. Only meaningful for stocks and ETFs; Yahoo's
    index volume is unreliable (often zero on real sessions).
    """
    return _flat(df) & (df["Volume"] == 0)


def scale_glitches(
    close: pd.Series,
    min_jump: float = config.GLITCH_MIN_JUMP,
    max_rows: int = config.GLITCH_MAX_ROWS,
) -> pd.Series:
    """Rows inside a price excursion that jumps away and comes straight back.

    A close-to-close jump beyond ``min_jump`` starts a candidate excursion. If,
    within ``max_rows`` rows, an opposite jump brings the price back to within a
    quarter of the original jump, every row from the first jump up to (not
    including) the return row is flagged.
    """
    lc = np.log(close.to_numpy(dtype=float))
    jump = np.log1p(min_jump)
    bad = np.zeros(len(lc), dtype=bool)
    i = 1
    while i < len(lc):
        end = _excursion_end(lc, i, jump, max_rows)
        if end is None:
            i += 1
        else:
            bad[i:end] = True
            i = end
    return pd.Series(bad, index=close.index)


def _excursion_end(lc: np.ndarray, i: int, jump: float, max_rows: int) -> int | None:
    """If the jump into row ``i`` reverts within ``max_rows`` rows, the first row back."""
    step = lc[i] - lc[i - 1]
    if abs(step) <= jump:
        return None
    for j in range(i + 1, min(i + max_rows, len(lc) - 1) + 1):
        back = lc[j] - lc[j - 1]
        if (
            abs(back) > jump
            and np.sign(back) != np.sign(step)
            and abs(lc[j] - lc[i - 1]) < 0.25 * abs(step)
        ):
            return j
    return None


# --------------------------------------------------------------------------
# Return checks
# --------------------------------------------------------------------------

def stale_open(df: pd.DataFrame) -> pd.Series:
    """Open equal to the previous close.

    For an index this is a data error: a value computed from 50 opening prices
    does not land exactly on yesterday's close. For stocks it is real (NSE's
    official OPEN equals PREVCLOSE on these days), so callers apply it to
    indices only.
    """
    return _close(df["Open"], df["Close"].shift(1))


def extreme_night(df: pd.DataFrame, limit: float = config.EXTREME_NIGHT) -> pd.Series:
    """Overnight simple return beyond +-``limit``."""
    night = df["Open"] / df["Close"].shift(1) - 1
    return night.abs() > limit


def bad_open(df: pd.DataFrame, min_jump: float = config.BAD_OPEN_MIN_JUMP) -> pd.Series:
    """Night and day both large, opposite, and nearly cancelling: the Open is wrong."""
    n = np.log(df["Open"] / df["Close"].shift(1))
    d = np.log(df["Close"] / df["Open"])
    jump = np.log1p(min_jump)
    return (
        (n.abs() > jump) & (d.abs() > jump)
        & (np.sign(n) != np.sign(d))
        & ((n + d).abs() < 0.25 * np.minimum(n.abs(), d.abs()))
    )
