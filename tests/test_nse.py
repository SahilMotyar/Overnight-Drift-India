"""NSE archive URLs, parsing, and caching -- all offline."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from src import nse


def _zip(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("bhav.csv", csv_text)
    return buf.getvalue()


def test_bhav_url_switches_format_on_2024_07_08() -> None:
    assert nse.bhav_url(pd.Timestamp("2024-07-05")).endswith("/2024/JUL/cm05JUL2024bhav.csv.zip")
    assert nse.bhav_url(pd.Timestamp("2024-07-08")).endswith("BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip")


def test_old_format_parses_and_gets_an_empty_isin() -> None:
    text = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,\n"
            "RELIANCE,EQ,510.6,511,483.65,490.25,483.65,519.2,7292002,1.0,11-MAY-2004,\n")
    df = nse.parse_bhavcopy(_zip(text))
    assert list(df.columns) == nse.BHAV_COLUMNS
    assert df.loc[0, "CLOSE"] == 490.25 and pd.isna(df.loc[0, "ISIN"])


def test_new_format_is_renamed_to_old_names() -> None:
    text = ("TradDt,TckrSymb,SctySrs,ISIN,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,TtlTradgVol\n"
            "2026-10-05,RELIANCE,EQ,INE002A01018,1190,1220,1188,1218,1217.5,1189,100\n")
    df = nse.parse_bhavcopy(_zip(text))
    row = df.iloc[0]
    assert (row.SYMBOL, row.OPEN, row.PREVCLOSE, row.ISIN) == ("RELIANCE", 1190, 1189, "INE002A01018")


def test_missing_day_is_cached_and_never_refetched(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(nse, "_get", lambda url: calls.append(url))  # returns None: 404
    assert nse.bhavcopy(pd.Timestamp("2000-01-26"), cache_dir=tmp_path) is None
    assert nse.bhavcopy(pd.Timestamp("2000-01-26"), cache_dir=tmp_path) is None
    assert len(calls) == 1


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_dropped_connection_is_retried_but_404_is_an_answer(monkeypatch) -> None:
    import http.client
    import urllib.error

    monkeypatch.setattr(nse.config, "NSE_REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(nse, "RETRY_WAITS", (0, 0))
    monkeypatch.setattr(nse.time, "sleep", lambda s: None)

    calls = []

    def flaky(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return _Resp(b"data")

    monkeypatch.setattr(nse.urllib.request, "urlopen", flaky)
    assert nse._get("https://x/file.zip") == b"data" and len(calls) == 2

    def missing(req, timeout):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    calls.clear()
    monkeypatch.setattr(nse.urllib.request, "urlopen", missing)
    assert nse._get("https://x/none.zip") is None and len(calls) == 1

    def always_down(req, timeout):
        calls.append(req.full_url)
        raise ConnectionResetError("reset")

    calls.clear()
    monkeypatch.setattr(nse.urllib.request, "urlopen", always_down)
    with pytest.raises(ConnectionResetError):
        nse._get("https://x/down.zip")
    assert len(calls) == 3  # first try + two retries


def test_verify_sessions_records_answers_and_reuses_them(tmp_path, monkeypatch) -> None:
    answers = {pd.Timestamp("2025-03-18"): True, pd.Timestamp("2000-01-26"): False}
    asked = []

    def fake_bhav(day, cache_dir):
        asked.append(day)
        return pd.DataFrame() if answers[day] else None

    monkeypatch.setattr(nse, "bhavcopy", fake_bhav)
    path = tmp_path / "checks.csv"
    days = pd.DatetimeIndex(list(answers))
    got = nse.verify_sessions(days, path=path, cache_dir=tmp_path)
    assert got.to_dict() == answers
    nse.verify_sessions(days, path=path, cache_dir=tmp_path)
    assert len(asked) == 2  # second call answered from the file


def test_index_file_not_requested_before_the_archive_starts(monkeypatch) -> None:
    monkeypatch.setattr(nse, "_get", lambda url: pytest.fail("requested a pre-2012 index file"))
    assert nse.index_close(pd.Timestamp("2008-01-15")) is None
