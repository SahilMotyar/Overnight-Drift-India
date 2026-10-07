"""The NSE trading calendar, special sessions, and what each night spans.

Yahoo has no calendar of its own: it invents rows on holidays and misses real
sessions (every stock lacks 2025-03-18). So the calendar is built from the
data and settled against NSE:

1. **Consensus.** A date is a candidate session if at least
   ``config.CONSENSUS_SHARE`` of the stocks listed at the time have a real row
   on it, after the row checks.
2. **Disputes.** Where consensus and the index calendar disagree, or the
   stocks are split, NSE decides: a session if and only if it published a
   bhavcopy that day (:func:`src.nse.verify_sessions`).

Then every row's night is labelled by what it spans. A night that spans a
session the series has no row for covers a whole trading day as well, so it
is not a night return at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def consensus_share(
    frames: dict[str, pd.DataFrame], min_series: int = config.CONSENSUS_MIN_SERIES
) -> pd.Series:
    """For every date any series has, the share of series alive then that have a row.

    A series is alive between its own first and last row, so a stock listed in
    2017 does not count against 2005 dates. Where fewer than ``min_series`` are
    alive the share is NaN: too few to vote (before 1996 only TMPV is alive).
    """
    dates = pd.DatetimeIndex([])
    for df in frames.values():
        dates = dates.union(pd.DatetimeIndex(df.index))
    alive = np.zeros(len(dates))
    has = np.zeros(len(dates))
    for df in frames.values():
        idx = pd.DatetimeIndex(df.index)
        alive += (dates >= idx[0]) & (dates <= idx[-1])
        has += dates.isin(idx)
    share = np.where(alive >= min_series, has / np.maximum(alive, 1), np.nan)
    return pd.Series(share, index=dates, name="share")


def disputed_dates(
    share: pd.Series,
    index_dates: pd.DatetimeIndex,
    threshold: float = config.CONSENSUS_SHARE,
    split: tuple[float, float] = config.DISPUTED_SHARE,
) -> pd.DatetimeIndex:
    """Dates the data alone cannot settle.

    * consensus and the index calendar disagree (within the indices' span)
    * the stocks are split: between ``split[0]`` and ``split[1]`` have a row
    """
    consensus = share.index[share >= threshold]
    lo, hi = index_dates.min(), index_dates.max()
    disagree = consensus.symmetric_difference(index_dates)
    disagree = disagree[(disagree >= lo) & (disagree <= hi)]
    divided = share.index[(share > split[0]) & (share < split[1])]
    return disagree.union(divided)


def build_calendar(
    share: pd.Series,
    verified: pd.Series,
    threshold: float = config.CONSENSUS_SHARE,
) -> pd.DatetimeIndex:
    """Consensus sessions, overridden by NSE's verdict on every verified date."""
    sessions = set(share.index[share >= threshold])
    sessions |= set(verified.index[verified])
    sessions -= set(verified.index[~verified])
    return pd.DatetimeIndex(sorted(sessions), name="Date")


def session_kinds(
    calendar: pd.DatetimeIndex, muhurat: tuple[str, ...] = config.MUHURAT_SESSIONS
) -> pd.Series:
    """``regular``, ``muhurat`` (Diwali evening session) or ``weekend`` (Saturday/Sunday)."""
    kinds = pd.Series("regular", index=calendar, name="session_kind")
    kinds[calendar.dayofweek >= 5] = "weekend"
    kinds[calendar.isin(pd.DatetimeIndex(muhurat))] = "muhurat"
    return kinds


def tag_nights(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Label the night ending at each of a series' rows.

    * ``first``: the series' first row; there is no previous close
    * ``gap``: a session lies between the previous row and this one, so the
      "night" also contains at least one whole trading day
    * ``weeknight``: consecutive calendar days
    * ``weekend``: only Saturday/Sunday in between
    * ``holiday``: a weekday non-session in between

    ``night_days`` is the calendar length of the night (0 for ``first``).
    """
    dates = pd.DatetimeIndex(dates)
    pos = calendar.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError(f"{(pos < 0).sum()} row(s) are not sessions; drop them first")

    prev = dates[:-1]
    cur = dates[1:]
    skipped = np.diff(pos) > 1
    days = (cur - prev).days.to_numpy()
    weekdays_between = np.busday_count(
        (prev + pd.Timedelta(days=1)).to_numpy().astype("datetime64[D]"),
        cur.to_numpy().astype("datetime64[D]"),
    )
    kind = np.select(
        [skipped, days == 1, weekdays_between == 0],
        ["gap", "weeknight", "weekend"],
        default="holiday",
    )
    return pd.DataFrame(
        {"night_kind": np.concatenate([["first"], kind]),
         "night_days": np.concatenate([[0], days])},
        index=dates,
    )


def volume_ratio(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, window: int = 20) -> pd.Series:
    """Cross-sectional median of volume / trailing ``window``-session median volume.

    A short session (Muhurat, a market-wide halt) shows up as a collapse; the
    Phase 2 checkpoint lists them to cross-check ``config.MUHURAT_SESSIONS``.
    """
    vol = pd.DataFrame({t: df["Volume"] for t, df in frames.items()}).reindex(calendar)
    trailing = vol.rolling(window, min_periods=window // 2).median().shift(1)
    return (vol / trailing).median(axis=1).rename("volume_ratio")
