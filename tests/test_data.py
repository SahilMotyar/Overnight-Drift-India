"""Cache shaping, cache freshness, and the dividend adjustment."""

from __future__ import annotations

import os
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import data


def _history(days: list[str]) -> pd.DataFrame:
    """A frame shaped like ``yf.Ticker.history``: midnight-IST tz-aware index."""
    idx = pd.DatetimeIndex(days).tz_localize("Asia/Kolkata").rename("Date")
    n = len(days)
    return pd.DataFrame({
        "Open": np.full(n, 100.0), "High": np.full(n, 101.0), "Low": np.full(n, 99.0),
        "Close": np.full(n, 100.0), "Adj Close": np.full(n, 100.0), "Volume": np.full(n, 1000),
        "Dividends": np.zeros(n), "Stock Splits": np.zeros(n),
    }, index=idx)


# --------------------------------------------------------------------------
# tidy
# --------------------------------------------------------------------------

def test_tidy_drops_todays_partial_session_and_tz() -> None:
    h = _history(["2026-10-05", "2026-10-06", "2026-10-07"])
    df = data.tidy(h, today=date(2026, 10, 7))
    assert list(df.index) == [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")]
    assert df.index.tz is None and df.index.name == "Date"


def test_tidy_keeps_only_raw_columns() -> None:
    h = _history(["2026-10-05"])
    h["Capital Gains"] = 0.0
    assert list(data.tidy(h, today=date(2026, 10, 7)).columns) == data.RAW_COLUMNS


def test_tidy_fails_on_missing_column() -> None:
    h = _history(["2026-10-05"]).drop(columns="Adj Close")
    with pytest.raises(ValueError, match="Adj Close"):
        data.tidy(h, today=date(2026, 10, 7))


# --------------------------------------------------------------------------
# dividend adjustment
# --------------------------------------------------------------------------

def _dividend_case() -> pd.DataFrame:
    """Dividend of 2 going ex on day 2, with Yahoo's factor 1 - D / prev Close."""
    h = _history(["2026-01-01", "2026-01-02", "2026-01-05"])
    h["Open"] = [99.0, 97.5, 99.5]
    h["High"] = [100.5, 99.5, 100.5]
    h["Low"] = [98.5, 97.0, 99.0]
    h["Close"] = [100.0, 99.0, 100.0]
    h["Dividends"] = [0.0, 2.0, 0.0]
    h["Adj Close"] = h["Close"] * np.array([1 - 2.0 / 100.0, 1.0, 1.0])
    return data.tidy(h, today=date(2026, 10, 7))


def test_adjusted_close_matches_yahoo_adj_close() -> None:
    raw = _dividend_case()
    adj = data.adjust_for_dividends(raw)
    np.testing.assert_allclose(adj["Close"], raw["Adj Close"])
    assert "Adj Close" not in adj.columns


def test_ex_date_dividend_is_credited_to_the_night() -> None:
    raw = _dividend_case()
    adj = data.adjust_for_dividends(raw)
    night = adj["Open"] / adj["Close"].shift(1) - 1
    expected = raw["Open"].iat[1] / (raw["Close"].iat[0] - raw["Dividends"].iat[1]) - 1
    assert night.iat[1] == pytest.approx(expected)
    # Away from the ex-date the night is untouched.
    raw_night = raw["Open"] / raw["Close"].shift(1) - 1
    assert night.iat[2] == pytest.approx(raw_night.iat[2])


def test_intraday_returns_are_unchanged() -> None:
    raw = _dividend_case()
    adj = data.adjust_for_dividends(raw)
    np.testing.assert_allclose(adj["Close"] / adj["Open"], raw["Close"] / raw["Open"])


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def test_cache_round_trip_and_load(tmp_path) -> None:
    raw = _dividend_case()
    raw.to_csv(data.cache_path("^TEST", tmp_path))
    pd.testing.assert_frame_equal(data.load("^TEST", adjusted=False, cache_dir=tmp_path), raw,
                                  check_freq=False)
    assert data.cache_path("^TEST", tmp_path).name == "TEST.csv"


def test_load_never_downloads(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="phase1_acquire"):
        data.load("ABSENT.NS", cache_dir=tmp_path)


def test_freshness_follows_the_ist_date(tmp_path) -> None:
    path = tmp_path / "X.csv"
    path.write_text("Date\n")
    assert data.is_fresh(path)
    two_days_ago = time.time() - 2 * 86400
    os.utime(path, (two_days_ago, two_days_ago))
    assert not data.is_fresh(path)


def test_fresh_cache_skips_the_network(tmp_path, monkeypatch) -> None:
    _dividend_case().to_csv(data.cache_path("FRESH.NS", tmp_path))

    import yfinance

    def boom(*_args, **_kwargs):
        raise AssertionError("downloaded despite a fresh cache")

    monkeypatch.setattr(yfinance, "Ticker", boom)
    assert len(data.download("FRESH.NS", cache_dir=tmp_path)) == 3
