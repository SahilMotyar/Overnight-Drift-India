"""Yahoo Finance download, daily on-disk cache, and dividend adjustment.

Only Phase 1 touches the network. Every later phase reads the cache through
:func:`load` and never downloads.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
RAW_COLUMNS = [*PRICE_COLUMNS, "Adj Close", "Volume", "Dividends", "Stock Splits"]


def ist_today() -> date:
    return datetime.now(IST).date()


def cache_path(ticker: str, cache_dir: Path = config.DATA_RAW) -> Path:
    return cache_dir / f"{ticker.replace('^', '')}.csv"


def is_fresh(path: Path) -> bool:
    """True if ``path`` was written earlier on today's IST date."""
    return path.exists() and datetime.fromtimestamp(path.stat().st_mtime, IST).date() >= ist_today()


def tidy(history: pd.DataFrame, today: date) -> pd.DataFrame:
    """Shape a raw ``Ticker.history`` frame for the cache.

    * index becomes tz-naive trading dates (Yahoo stamps them midnight IST)
    * keeps ``RAW_COLUMNS``; fails if any is missing
    * drops rows dated ``today`` (IST) or later: the run's own session may
      still be open, and the plan stops at the last *full* trading day

    Nothing else is changed. Data-quality fixes belong to Phase 2, where every
    dropped row is counted in the drop table.
    """
    missing = [c for c in RAW_COLUMNS if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing columns {missing}")
    extra = [c for c in history.columns if c not in RAW_COLUMNS]
    if extra:
        logger.info("Ignoring extra columns %s", extra)

    df = history[RAW_COLUMNS].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    df.index = idx.normalize().rename("Date")

    partial = df.index >= pd.Timestamp(today)
    if partial.any():
        logger.info("Dropping %d row(s) dated %s or later (session may be partial)", partial.sum(), today)
    return df.loc[~partial].sort_index()


def read_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])


def download(ticker: str, refresh: bool = False, cache_dir: Path = config.DATA_RAW) -> pd.DataFrame:
    """Return the cached raw frame for ``ticker``, downloading it if stale or ``refresh``."""
    path = cache_path(ticker, cache_dir)
    if not refresh and is_fresh(path):
        logger.info("Using today's cache for %s", ticker)
        return read_cache(path)

    import yfinance as yf  # imported lazily: later phases only read the cache

    logger.info("Downloading %s from Yahoo Finance", ticker)
    history = yf.Ticker(ticker).history(
        period=config.YF_PERIOD, interval="1d", auto_adjust=False, actions=True
    )
    if history is None or history.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker!r}")

    df = tidy(history, today=ist_today())
    if df.empty:
        raise RuntimeError(f"no complete sessions for {ticker!r} before {ist_today()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def adjust_for_dividends(raw: pd.DataFrame) -> pd.DataFrame:
    """Rescale Open/High/Low/Close by ``Adj Close / Close``, as auto_adjust=True does.

    Yahoo's OHLC are already split-adjusted, so the only effect is dividends.
    On an ex-date the night return becomes ``Open / (prev Close - D) - 1``
    instead of ``Open / prev Close - 1``: the dividend is credited to whoever
    held overnight. Intraday returns are unchanged, because both prices on a
    day share one factor.
    """
    factor = raw["Adj Close"] / raw["Close"]
    out = raw.drop(columns="Adj Close")
    out[PRICE_COLUMNS] = raw[PRICE_COLUMNS].mul(factor, axis=0)
    return out


def load(ticker: str, adjusted: bool = True, cache_dir: Path = config.DATA_RAW) -> pd.DataFrame:
    """Read ``ticker`` from the cache (never the network); dividend-adjusted by default."""
    path = cache_path(ticker, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached prices for {ticker!r} at {path}; run `python -m scripts.phase1_acquire`"
        )
    raw = read_cache(path)
    return adjust_for_dividends(raw) if adjusted else raw
