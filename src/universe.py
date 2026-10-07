"""The stock universe: Nifty 50 constituents as published by NSE, frozen on disk.

The list is fetched once and written to ``config.NIFTY50_SNAPSHOT``, which is
committed. Later runs read the snapshot and never re-fetch unless asked, so the
sample cannot change under the study when NSE rebalances the index.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["Company Name", "Industry", "Symbol", "Series", "ISIN Code"]
N_CONSTITUENTS = 50

# NSE's servers reject requests that don't look like they come from a browser.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}


def parse_constituents(text: str, source: str, fetched: str) -> pd.DataFrame:
    """Validate NSE's constituent CSV and add the Yahoo ticker for each stock.

    Fails loudly on any surprise (columns, row count, duplicate symbols) rather
    than freezing a malformed universe.
    """
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = df.columns.str.strip()
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns from {source}: {list(df.columns)}")
    if len(df) != N_CONSTITUENTS:
        raise ValueError(f"expected {N_CONSTITUENTS} constituents from {source}, got {len(df)}")

    df = df.apply(lambda col: col.str.strip())
    dupes = df.loc[df["Symbol"].duplicated(), "Symbol"].tolist()
    if dupes:
        raise ValueError(f"duplicate symbols from {source}: {dupes}")

    df["Ticker"] = df["Symbol"] + ".NS"
    df["Source"] = source
    df["Fetched"] = fetched
    return df


def fetch_nifty50(urls: tuple[str, ...] = config.NIFTY50_LIST_URLS) -> pd.DataFrame:
    """Download the current constituent list, trying each official source in turn.

    A network failure moves on to the next source; a malformed file does not,
    because a second copy of the same file would not fix it.
    """
    errors: list[str] = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8-sig")
        except OSError as e:  # URLError, HTTPError and timeouts all subclass OSError
            errors.append(f"{url}: {e}")
            continue
        fetched = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return parse_constituents(text, source=url, fetched=fetched)
    raise RuntimeError("could not fetch the Nifty 50 list from any source:\n  " + "\n  ".join(errors))


def load_universe(path: Path = config.NIFTY50_SNAPSHOT) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python -m scripts.phase1_acquire` to freeze the universe"
        )
    return pd.read_csv(path, dtype=str)


def freeze_universe(refresh: bool = False, path: Path = config.NIFTY50_SNAPSHOT) -> pd.DataFrame:
    """Return the frozen universe, fetching and writing it only if absent or ``refresh``."""
    if path.exists() and not refresh:
        return load_universe(path)
    df = fetch_nifty50()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Froze %d constituents from %s to %s", len(df), df["Source"].iat[0], path)
    return df


def stock_tickers(universe: pd.DataFrame | None = None) -> dict[str, str]:
    """``{yahoo_ticker: company_name}`` for the frozen universe."""
    u = load_universe() if universe is None else universe
    return dict(zip(u["Ticker"], u["Company Name"]))
