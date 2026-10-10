"""Phase 2 pipeline: a raw Yahoo frame -> a validated frame with return flags.

Two passes, because the calendar needs every series:

1. :func:`row_stage` per series: the NSE start floor and the row checks that
   need no calendar.
2. The calendar, from all row-stage survivors (src.sessions, src.nse).
3. :func:`finish` per series: off-calendar rows, start date, scale glitches,
   then a verdict on every night and every day, the ``reversed_open`` flag,
   and the dividend adjustment.

Every removed row and every rejected return is counted in a :class:`DropLog`.
A rejected return is counted once, under the first reason that applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config, sessions, validate
from .data import adjust_for_dividends

CLEAN_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
    "session_kind", "night_kind", "night_days", "night_ok", "night_note", "day_ok", "day_note",
    "reversed_open",
]


@dataclass
class DropLog:
    records: list[dict] = field(default_factory=list)

    def add(self, ticker: str, stage: str, reason: str, count: int) -> None:
        if count:
            self.records.append({"ticker": ticker, "stage": stage, "reason": reason, "count": int(count)})

    def table(self) -> pd.DataFrame:
        """``ticker x stage x reason -> count``; stage is ``row``, ``night`` or ``day``."""
        if not self.records:
            return pd.DataFrame(columns=["ticker", "stage", "reason", "count"])
        df = pd.DataFrame(self.records)
        return df.groupby(["ticker", "stage", "reason"], sort=False, as_index=False)["count"].sum()


def _drop(df: pd.DataFrame, mask: pd.Series, ticker: str, reason: str, log: DropLog) -> pd.DataFrame:
    log.add(ticker, "row", reason, mask.sum())
    return df.loc[~mask]


def row_stage(raw: pd.DataFrame, ticker: str, is_index: bool, log: DropLog) -> pd.DataFrame:
    """Row checks that need no calendar."""
    df = _drop(raw, raw.index < pd.Timestamp(config.NSE_EQUITY_START), ticker,
               "before NSE's equity segment opened (1994-11-03)", log)
    df = _drop(df, validate.nonpositive(df), ticker, "price <= 0", log)
    df = _drop(df, validate.ohlc_inconsistent(df), ticker, "Open/Close outside [Low, High]", log)
    df = _drop(df, validate.placeholder(df), ticker, "placeholder: flat at previous close, no volume", log)
    if not is_index:
        df = _drop(df, validate.no_trades(df), ticker, "no trades: flat bar, zero volume", log)
    return df


def start_date(
    dates: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    window: int = config.COVERAGE_WINDOW,
    min_coverage: float = config.MIN_COVERAGE,
) -> pd.Timestamp:
    """First row after the last missing session inside the last under-covered window."""
    span = calendar[(calendar >= dates[0]) & (calendar <= dates[-1])]
    present = pd.Series(span.isin(dates), index=span)
    coverage = present.rolling(window, min_periods=window).mean()
    weak = coverage.index[coverage < min_coverage]
    if not len(weak):
        return dates[0]
    last_missing = present.index[~present.to_numpy() & (present.index <= weak[-1])][-1]
    return dates[dates > last_missing][0]


def finish(
    rows: pd.DataFrame,
    ticker: str,
    is_index: bool,
    calendar: pd.DatetimeIndex,
    kinds: pd.Series,
    log: DropLog,
    coverage_calendar: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calendar-dependent checks and return flags. Returns the clean frame and a summary row.

    ``coverage_calendar`` is what the start-date rule measures coverage against.
    Pass the sessions at least one series has: a session no Yahoo series has
    (e.g. Saturday 2004-10-09) is a market-wide gap, not evidence that this
    series is unreliable, and would otherwise drag start dates past it.
    """
    df = _drop(rows, ~rows.index.isin(calendar), ticker, "not an NSE session", log)

    rule = start_date(df.index, calendar if coverage_calendar is None else coverage_calendar)
    override = config.START_OVERRIDES.get(ticker)
    start, start_by, why = rule, "coverage rule", ""
    if override and pd.Timestamp(override[0]) > start:
        start, start_by, why = pd.Timestamp(override[0]), "override", override[1]
    bad_dividends = df.index[validate.implausible_dividend(df)]
    if len(bad_dividends) and bad_dividends[-1] > start:
        start, start_by = bad_dividends[-1], "implausible dividend"
        why = f"Yahoo dividend > {config.MAX_DIVIDEND_YIELD:.0%} of price on {bad_dividends[-1].date()}"
    df = _drop(df, df.index < start, ticker, "before series start date", log)

    while (glitch := validate.scale_glitches(df["Close"])).any():
        df = _drop(df, glitch, ticker, "scale glitch: jump that reverts within 5 rows", log)

    nights = sessions.tag_nights(df.index, calendar)
    session = kinds.reindex(df.index)
    night_note = pd.Series("", index=df.index)
    day_note = pd.Series("", index=df.index)

    def reject(note: pd.Series, mask: pd.Series, reason: str, stage: str) -> None:
        new = mask.to_numpy() & (note == "").to_numpy()
        note[new] = reason
        log.add(ticker, stage, reason, new.sum())

    special = session != "regular"
    reject(night_note, nights["night_kind"] == "first", "first row: no previous close", "night")
    reject(night_note, nights["night_kind"] == "gap", "spans a session the series lacks", "night")
    reject(night_note, special | special.shift(1, fill_value=False),
           "special session at either end", "night")
    reject(day_note, special, "special session (Muhurat / weekend)", "day")
    if is_index:
        stale = validate.stale_open(df)
        reject(night_note, stale, "stale index open (= previous close)", "night")
        reject(day_note, stale, "stale index open (= previous close)", "day")
    reject(night_note, validate.extreme_night(df), "night beyond +-20%", "night")

    out = adjust_for_dividends(df).join(nights)
    nonpositive = (out[validate.PRICE_COLUMNS] <= 0).any(axis=1)
    if nonpositive.any():
        raise ValueError(f"{ticker}: {nonpositive.sum()} dividend-adjusted price(s) <= 0 from "
                         f"{out.index[nonpositive][0].date()}; a dividend Yahoo mis-scaled?")
    out["session_kind"] = session
    out["night_ok"] = night_note == ""
    out["night_note"] = night_note
    out["day_ok"] = day_note == ""
    out["day_note"] = day_note
    out["reversed_open"] = validate.reversed_open(df) & out["night_ok"] & out["day_ok"]
    out = out[CLEAN_COLUMNS]

    summary = {
        "ticker": ticker,
        "start": start.date(),
        "start_by": start_by,
        "start_reason": why,
        "first_raw": rows.index[0].date(),
        "last": out.index[-1].date(),
        "rows": len(out),
        "nights_ok": int(out["night_ok"].sum()),
        "days_ok": int(out["day_ok"].sum()),
        "reversed_opens": int(out["reversed_open"].sum()),
    }
    return out, summary
