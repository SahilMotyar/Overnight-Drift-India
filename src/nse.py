"""NSE's own archives: daily bhavcopies and index closes.

Phase 2 uses them for two things:

* settling disputed trading days: NSE publishes a bhavcopy for every session
  and only for sessions, so a missing file means NSE did not trade;
* spot-checking Yahoo's prices against NSE's official OPEN / CLOSE / PREVCLOSE.

Every file is cached under ``config.DATA_NSE``. A day with no NSE file is
cached as an empty marker, so it is never requested twice.
"""

from __future__ import annotations

import http.client
import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

# strftime's %b depends on the OS locale; NSE's paths want English, upper case.
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

BHAV_OLD = "https://nsearchives.nseindia.com/content/historical/EQUITIES/{y}/{mon}/cm{dd}{mon}{y}bhav.csv.zip"
BHAV_NEW = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
BHAV_NEW_FROM = pd.Timestamp("2024-07-08")  # the old path stops on 2024-07-05

INDEX_CLOSE = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{dmy}.csv"
INDEX_CLOSE_FROM = pd.Timestamp("2012-01-01")  # the archive has no earlier files

BHAV_COLUMNS = ["SYMBOL", "SERIES", "ISIN", "OPEN", "HIGH", "LOW", "CLOSE", "LAST", "PREVCLOSE", "TOTTRDQTY"]
_NEW_TO_OLD = {
    "TckrSymb": "SYMBOL", "SctySrs": "SERIES", "OpnPric": "OPEN", "HghPric": "HIGH",
    "LwPric": "LOW", "ClsPric": "CLOSE", "LastPric": "LAST", "PrvsClsgPric": "PREVCLOSE",
    "TtlTradgVol": "TOTTRDQTY",
}
INDEX_COLUMNS = ["Index Name", "Open Index Value", "High Index Value", "Low Index Value",
                 "Closing Index Value", "Points Change"]


def bhav_url(day: pd.Timestamp) -> str:
    day = pd.Timestamp(day)
    if day >= BHAV_NEW_FROM:
        return BHAV_NEW.format(ymd=f"{day:%Y%m%d}")
    return BHAV_OLD.format(y=day.year, mon=_MONTHS[day.month - 1], dd=f"{day:%d}")


def index_close_url(day: pd.Timestamp) -> str:
    return INDEX_CLOSE.format(dmy=f"{pd.Timestamp(day):%d%m%Y}")


# Seconds to wait before each retry of a transient failure (dropped connection,
# timeout, 5xx). A 404 is an answer, not a failure, and is never retried.
RETRY_WAITS: tuple[float, ...] = (2, 10, 30)
_TRANSIENT = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)


def _get(url: str) -> bytes | None:
    """Fetch ``url``; ``None`` if NSE has no such file (404). Other failures raise
    once the retries are used up."""
    for wait in (*RETRY_WAITS, None):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code < 500 or wait is None:
                raise
            error: Exception = e
        except _TRANSIENT as e:
            if wait is None:
                raise
            error = e
        finally:
            time.sleep(config.NSE_REQUEST_DELAY_SECONDS)
        logger.warning("NSE request failed (%s); retrying in %ss: %s", error, wait, url)
        time.sleep(wait)
    raise AssertionError("unreachable")


def parse_bhavcopy(raw_zip: bytes) -> pd.DataFrame:
    """Either bhavcopy format -> ``BHAV_COLUMNS`` (old-format names)."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype={"ISIN": str})
    df.columns = df.columns.str.strip()
    df = df.rename(columns=_NEW_TO_OLD)
    if "ISIN" not in df.columns:  # the old format only added ISIN around 2011
        df["ISIN"] = pd.NA
    df["SYMBOL"] = df["SYMBOL"].str.strip()
    df["SERIES"] = df["SERIES"].str.strip()
    return df[BHAV_COLUMNS]


_PRICE_FIELDS = ["OPEN", "HIGH", "LOW", "CLOSE", "LAST", "PREVCLOSE", "TOTTRDQTY",
                 *INDEX_COLUMNS[1:]]


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """NSE writes "-" for a value it does not have (e.g. an index with no open)."""
    for col in df.columns.intersection(_PRICE_FIELDS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _cached(path: Path, fetch, parse) -> pd.DataFrame | None:
    if path.exists():
        return None if path.stat().st_size == 0 else _numeric(pd.read_csv(path, dtype={"ISIN": str}))
    raw = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        path.write_bytes(b"")  # marker: NSE has no file for this day
        return None
    df = _numeric(parse(raw))
    df.to_csv(path, index=False)
    return df


def bhavcopy(day: pd.Timestamp, cache_dir: Path = config.DATA_NSE) -> pd.DataFrame | None:
    """NSE's equity bhavcopy for ``day``, or ``None`` if NSE did not trade."""
    day = pd.Timestamp(day)
    return _cached(cache_dir / f"bhav_{day:%Y%m%d}.csv", lambda: _get(bhav_url(day)), parse_bhavcopy)


def index_close(day: pd.Timestamp, cache_dir: Path = config.DATA_NSE) -> pd.DataFrame | None:
    """NSE's all-indices close file for ``day`` (2012 onward), or ``None``."""
    day = pd.Timestamp(day)
    if day < INDEX_CLOSE_FROM:
        return None

    def parse(raw: bytes) -> pd.DataFrame:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = df.columns.str.strip()
        df["Index Name"] = df["Index Name"].str.strip()
        return df[INDEX_COLUMNS]

    return _cached(cache_dir / f"indices_{day:%Y%m%d}.csv", lambda: _get(index_close_url(day)), parse)


def verify_sessions(
    days: pd.DatetimeIndex,
    path: Path = config.NSE_SESSION_CHECKS,
    cache_dir: Path = config.DATA_NSE,
) -> pd.Series:
    """Whether NSE traded on each of ``days``, by whether it published a bhavcopy.

    Answers are appended to ``path`` (committed), so the calendar can be
    rebuilt offline and every disputed date's verdict is on record.
    """
    known = (
        pd.read_csv(path, parse_dates=["date"]).set_index("date")["nse_session"]
        if path.exists() else pd.Series(dtype=bool, name="nse_session")
    )
    todo = pd.DatetimeIndex(days).difference(known.index)
    if len(todo):
        logger.info("Asking NSE about %d disputed date(s)", len(todo))
        new = pd.Series({d: bhavcopy(d, cache_dir) is not None for d in todo}, name="nse_session")
        known = pd.concat([known, new]).sort_index()
        known.rename_axis("date").reset_index().to_csv(path, index=False)
    return known.reindex(pd.DatetimeIndex(days)).astype(bool)
