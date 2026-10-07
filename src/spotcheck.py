"""Phase 2 spot-check: cleaned Yahoo returns against NSE's official prices.

Compares returns, not price levels. Yahoo's prices are split-adjusted and
NSE's are not, but a night (Open / previous close) and a day (Close / Open) are
the same in both. Raw Yahoo prices are used, not dividend-adjusted ones,
because NSE's PREVCLOSE is not dividend-adjusted either.

Only returns the cleaned data accepts (``night_ok`` / ``day_ok``) count
towards the match rates: the question is whether what survives Phase 2 is
right. Sensex is a BSE index, so NSE cannot check it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from . import config, nse

# NSE renamed its indices over the years ("S&P CNX Nifty" -> "CNX Nifty" -> "Nifty 50").
INDEX_NAMES: dict[str, tuple[str, ...]] = {
    "^NSEI": ("Nifty 50", "CNX Nifty", "S&P CNX Nifty"),
    "^NSEBANK": ("Nifty Bank", "Bank Nifty", "CNX Bank Index", "CNX Bank"),
}


def sample_sessions(
    calendar: pd.DatetimeIndex, kinds: pd.Series, n: int, start: str, rng: np.random.Generator
) -> pd.DatetimeIndex:
    regular = (kinds.reindex(calendar) == "regular").to_numpy()
    pool = calendar[(calendar >= pd.Timestamp(start)) & regular]
    return pd.DatetimeIndex(np.sort(rng.choice(pool.to_numpy(), size=n, replace=False)))


def _stock_row(bhav: pd.DataFrame, symbol: str, isin: str | None) -> pd.Series | None:
    """Match by ISIN where the file has one (survives renames), else by symbol."""
    eq = bhav[bhav["SERIES"] == "EQ"]
    if isin:
        hit = eq[eq["ISIN"] == isin]
        if len(hit):
            return hit.iloc[0]
    hit = eq[eq["SYMBOL"] == symbol]
    return hit.iloc[0] if len(hit) else None


def _reference(
    day: pd.Timestamp, ticker: str, ids: dict[str, tuple[str, str | None]], cache_dir
) -> tuple[bool, dict[str, float] | None]:
    """``(checkable, NSE prices)``. Not checkable: Sensex (BSE), or no NSE file that day."""
    if ticker in INDEX_NAMES:
        indices = nse.index_close(day, cache_dir)
        if indices is None:
            return False, None
        hit = indices[indices["Index Name"].isin(INDEX_NAMES[ticker])]
        if hit.empty:
            return True, None
        h = hit.iloc[0]
        return True, {"open": h["Open Index Value"], "close": h["Closing Index Value"], "last": np.nan,
                      "prev": h["Closing Index Value"] - h["Points Change"]}
    if ticker in ids:
        bhav = nse.bhavcopy(day, cache_dir)
        if bhav is None:
            return False, None
        row = _stock_row(bhav, *ids[ticker])
        if row is None:
            return True, None
        return True, {"open": row["OPEN"], "close": row["CLOSE"], "last": row["LAST"], "prev": row["PREVCLOSE"]}
    return False, None


def _record(day: pd.Timestamp, ticker: str, cf: pd.DataFrame, y: pd.DataFrame,
            ref: dict[str, float] | None) -> dict[str, object]:
    rec: dict[str, object] = {"date": day.date(), "ticker": ticker, "found": ref is not None}
    if ref is not None:
        prev_day = cf.index[cf.index.get_loc(day) - 1]
        y_open, y_close, y_prev = y.at[day, "Open"], y.at[day, "Close"], y.at[prev_day, "Close"]
        rec.update({
            "night_ok": bool(cf.at[day, "night_ok"]),
            "day_ok": bool(cf.at[day, "day_ok"]),
            "night_yahoo": y_open / y_prev - 1,
            "night_nse": ref["open"] / ref["prev"] - 1,
            "day_yahoo": y_close / y_open - 1,
            "day_nse": ref["close"] / ref["open"] - 1,
            "day_nse_last": ref["last"] / ref["open"] - 1,
        })
    return rec


def compare_session(
    day: pd.Timestamp,
    raw: dict[str, pd.DataFrame],
    clean: dict[str, pd.DataFrame],
    ids: dict[str, tuple[str, str | None]],
    cache_dir=config.DATA_NSE,
) -> list[dict[str, object]]:
    """One record per series NSE can check on ``day``.

    ``ids`` maps stock/ETF tickers to ``(NSE symbol, ISIN or None)``.
    """
    records = []
    for ticker, cf in clean.items():
        if day not in cf.index or cf.index.get_loc(day) == 0:
            continue
        checkable, ref = _reference(day, ticker, ids, cache_dir)
        if checkable:
            records.append(_record(day, ticker, cf, raw[ticker], ref))
    return records


SAME, DIFFERENT = "NSE shows the same move", "NSE differs from Yahoo"


def check_against_nse(
    raw: dict[str, pd.DataFrame],
    clean: dict[str, pd.DataFrame],
    ids: dict[str, tuple[str, str | None]],
    pick: Callable[[pd.DataFrame], pd.Series],
    label: str,
    tol: float = config.SPOTCHECK_TOLERANCE,
    cache_dir=config.DATA_NSE,
) -> pd.DataFrame:
    """Check the nights ``pick(clean_frame)`` selects against NSE's own prices.

    ``verdict`` is :data:`SAME` or :data:`DIFFERENT` (or not found / not
    checkable). SAME does not by itself mean an economic return: NSE leaves its
    PREVCLOSE unadjusted for demergers and bonus issues too (TMPV 2025-10-14,
    RELIANCE 1997-11-05), so a corporate action looks the same in both.
    """
    recs = []
    for ticker, cf in clean.items():
        for day in cf.index[pick(cf).to_numpy()]:
            if cf.index.get_loc(day) == 0:
                continue
            checkable, ref = _reference(day, ticker, ids, cache_dir)
            rec: dict[str, object] = {"ticker": ticker, "date": day.date(), "check": label}
            if not checkable:
                rec["verdict"] = "not checkable"
            elif ref is None:
                rec["verdict"] = "not found at NSE"
            else:
                rec.update(_record(day, ticker, cf, raw[ticker], ref))
                gap = abs(np.log1p(rec["night_yahoo"]) - np.log1p(rec["night_nse"]))
                rec["verdict"] = SAME if gap < tol else DIFFERENT
            recs.append(rec)
    return pd.DataFrame(recs)


def _log_gap(a: pd.Series, b: pd.Series) -> pd.Series:
    return (np.log1p(a) - np.log1p(b)).abs()


def summarize(records: pd.DataFrame, kinds: dict[str, str], tol: float = config.SPOTCHECK_TOLERANCE) -> pd.DataFrame:
    """Match rates per instrument kind, on the returns the cleaned data accepts.

    ``close_is_official_%``: on days where NSE's CLOSE and LAST differ by more
    than ``tol``, how often Yahoo's day return matches the official CLOSE.
    """
    df = records.assign(kind=records["ticker"].map(kinds))
    rows = []
    for kind, g in df.groupby("kind", sort=False):
        found = g[g["found"].astype(bool)]
        nights = found[found["night_ok"].astype(bool)]
        days = found[found["day_ok"].astype(bool)]
        night_match = _log_gap(nights["night_yahoo"], nights["night_nse"]) < tol
        day_match = _log_gap(days["day_yahoo"], days["day_nse"]) < tol
        split = days[_log_gap(days["day_nse"], days["day_nse_last"]) > tol]
        official = _log_gap(split["day_yahoo"], split["day_nse"]) < tol
        rows.append({
            "kind": kind,
            "checked": len(g),
            "not_found": int((~g["found"].astype(bool)).sum()),
            "nights": len(nights),
            "nights_match_%": round(100 * night_match.mean(), 1) if len(nights) else np.nan,
            "days": len(days),
            "days_match_%": round(100 * day_match.mean(), 1) if len(days) else np.nan,
            "close_vs_last_days": len(split),
            "close_is_official_%": round(100 * official.mean(), 1) if len(split) else np.nan,
        })
    return pd.DataFrame(rows).set_index("kind")
