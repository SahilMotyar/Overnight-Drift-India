"""Phase 1 checkpoint: which days each series has, against a reference calendar."""

from __future__ import annotations

import pandas as pd

from . import config


def reference_calendar(
    frames: dict[str, pd.DataFrame], tickers: tuple[str, ...] = config.CALENDAR_TICKERS
) -> pd.DatetimeIndex:
    """Union of the trading days of ``tickers``, the stand-in NSE calendar."""
    absent = [t for t in tickers if t not in frames]
    if absent:
        raise KeyError(f"reference calendar needs {absent}, which failed to download")
    dates = pd.DatetimeIndex([])
    for t in tickers:
        dates = dates.union(pd.DatetimeIndex(frames[t].index))
    return dates


def coverage_row(dates: pd.DatetimeIndex, ref: pd.DatetimeIndex) -> dict[str, object]:
    """Date coverage of one series, measured only within its own first..last span."""
    dates = pd.DatetimeIndex(dates)
    unique = dates.unique().sort_values()
    first, last = unique[0], unique[-1]
    in_span = ref[(ref >= first) & (ref <= last)]
    missing = int((~in_span.isin(unique)).sum())
    checkable = unique[unique >= ref[0]]
    return {
        "first": first.date(),
        "last": last.date(),
        "rows": len(dates),
        "before_ref": int((unique < ref[0]).sum()),
        "missing": missing,
        "missing_%": round(100 * missing / len(in_span), 2) if len(in_span) else 0.0,
        "extra": int((~checkable.isin(ref)).sum()),
        "behind": int((ref > last).sum()),
        "dup_dates": int(dates.duplicated().sum()),
        "weekend": int((unique.dayofweek >= 5).sum()),
        "max_gap_days": int(unique.to_series().diff().dt.days.max()) if len(unique) > 1 else 0,
    }


def coverage_table(frames: dict[str, pd.DataFrame], ref: pd.DatetimeIndex) -> pd.DataFrame:
    """One :func:`coverage_row` per ticker, indexed by ticker.

    * ``before_ref``: rows before the reference calendar starts (uncheckable)
    * ``missing``: reference trading days inside the span with no row
    * ``extra``: rows on days the reference calendar does not have
    * ``behind``: reference trading days after the series' last row
    * ``weekend``: rows on a Saturday/Sunday (special sessions, or bad data)
    * ``max_gap_days``: longest calendar-day gap between consecutive rows
    """
    rows = {t: coverage_row(df.index, ref) for t, df in frames.items()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("ticker")
