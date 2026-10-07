"""Spot-check matching and scoring, against fake NSE files (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import nse, spotcheck

D0, D1 = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")


def _yahoo(open_, close, prev_close, scale=1.0) -> pd.DataFrame:
    """Two rows; ``scale`` mimics Yahoo's split adjustment of the price level."""
    return pd.DataFrame(
        {"Open": [prev_close * 0.99 * scale, open_ * scale], "Close": [prev_close * scale, close * scale]},
        index=pd.DatetimeIndex([D0, D1]),
    )


def _clean(night_ok=True, day_ok=True) -> pd.DataFrame:
    return pd.DataFrame({"night_ok": [False, night_ok], "day_ok": [True, day_ok]},
                        index=pd.DatetimeIndex([D0, D1]))


def _bhav(**rows) -> pd.DataFrame:
    recs = [{"SYMBOL": s, "SERIES": "EQ", "ISIN": isin, "OPEN": o, "HIGH": np.nan, "LOW": np.nan,
             "CLOSE": c, "LAST": last, "PREVCLOSE": p, "TOTTRDQTY": 1}
            for s, (isin, o, c, last, p) in rows.items()]
    return pd.DataFrame(recs, columns=nse.BHAV_COLUMNS)


def test_returns_match_despite_split_adjusted_levels(monkeypatch) -> None:
    # NSE: prev 200, open 202, close 204 (last 205). Yahoo: same, at 1/10 scale.
    monkeypatch.setattr(nse, "bhavcopy", lambda day, cache_dir: _bhav(ABC=("INE1", 202, 204, 205, 200)))
    monkeypatch.setattr(nse, "index_close", lambda day, cache_dir: None)
    recs = spotcheck.compare_session(
        D1, {"ABC.NS": _yahoo(202, 204, 200, scale=0.1)}, {"ABC.NS": _clean()}, {"ABC.NS": ("ABC", "INE1")}
    )
    assert len(recs) == 1 and recs[0]["found"]
    assert np.isclose(recs[0]["night_yahoo"], recs[0]["night_nse"])
    assert np.isclose(recs[0]["day_yahoo"], recs[0]["day_nse"])
    summary = spotcheck.summarize(pd.DataFrame(recs), {"ABC.NS": "stock"})
    assert summary.loc["stock", "nights_match_%"] == 100.0
    assert summary.loc["stock", "close_is_official_%"] == 100.0  # 204, not the 205 last trade


def test_isin_match_survives_a_rename() -> None:
    bhav = _bhav(TATAGLOBAL=("INE192A01025", 1, 1, 1, 1))
    assert spotcheck._stock_row(bhav, "TATACONSUM", "INE192A01025")["SYMBOL"] == "TATAGLOBAL"
    assert spotcheck._stock_row(bhav, "TATACONSUM", None) is None


def test_rejected_returns_do_not_count(monkeypatch) -> None:
    monkeypatch.setattr(nse, "bhavcopy", lambda day, cache_dir: _bhav(ABC=("INE1", 150, 204, 204, 200)))
    monkeypatch.setattr(nse, "index_close", lambda day, cache_dir: None)
    # Yahoo's open is wrong (202 vs NSE 150), but the cleaned data already rejected it.
    recs = spotcheck.compare_session(
        D1, {"ABC.NS": _yahoo(202, 204, 200)}, {"ABC.NS": _clean(night_ok=False, day_ok=False)},
        {"ABC.NS": ("ABC", None)},
    )
    summary = spotcheck.summarize(pd.DataFrame(recs), {"ABC.NS": "stock"})
    assert summary.loc["stock", "nights"] == 0 and summary.loc["stock", "days"] == 0


def test_index_prev_close_comes_from_points_change(monkeypatch) -> None:
    idx = pd.DataFrame([{"Index Name": "Nifty 50", "Open Index Value": 101.0, "High Index Value": 0,
                         "Low Index Value": 0, "Closing Index Value": 102.0, "Points Change": 2.0}])
    monkeypatch.setattr(nse, "bhavcopy", lambda day, cache_dir: None)
    monkeypatch.setattr(nse, "index_close", lambda day, cache_dir: idx)
    recs = spotcheck.compare_session(D1, {"^NSEI": _yahoo(101, 102, 100)}, {"^NSEI": _clean()}, {})
    assert np.isclose(recs[0]["night_nse"], 0.01) and np.isclose(recs[0]["night_yahoo"], 0.01)


def test_rejected_nights_are_judged_against_nse(monkeypatch) -> None:
    # Yahoo night for both: 150/200 - 1 = -25%. NSE: ABC opened at 150 too
    # (a real gap); XYZ opened at 199 (Yahoo's open was wrong).
    monkeypatch.setattr(nse, "bhavcopy", lambda day, cache_dir: _bhav(
        ABC=("INE1", 150, 152, 152, 200), XYZ=("INE2", 199, 152, 152, 200)))
    monkeypatch.setattr(nse, "index_close", lambda day, cache_dir: None)
    cf = _clean(night_ok=False).assign(night_note=["first row: no previous close", "night beyond +-20%"])
    out = spotcheck.check_against_nse(
        {"ABC.NS": _yahoo(150, 152, 200), "XYZ.NS": _yahoo(150, 152, 200), "^BSESN": _yahoo(150, 152, 200)},
        {"ABC.NS": cf, "XYZ.NS": cf, "^BSESN": cf},
        {"ABC.NS": ("ABC", "INE1"), "XYZ.NS": ("XYZ", "INE2")},
        pick=lambda f: f["night_note"] == "night beyond +-20%", label="extreme",
    ).set_index("ticker")["verdict"]
    assert out.to_dict() == {"ABC.NS": spotcheck.SAME, "XYZ.NS": spotcheck.DIFFERENT,
                             "^BSESN": "not checkable"}


def test_dash_for_missing_values_reads_as_nan() -> None:
    df = nse._numeric(pd.DataFrame({"Index Name": ["A", "B"], "Open Index Value": ["5217.65", "-"]}))
    assert df["Open Index Value"].iloc[0] == 5217.65 and pd.isna(df["Open Index Value"].iloc[1])


def test_sampling_is_reproducible_and_skips_special_sessions() -> None:
    from src.config import rng
    cal = pd.bdate_range("2020-01-01", periods=200)
    kinds = pd.Series("regular", index=cal)
    kinds.iloc[::2] = "muhurat"
    a = spotcheck.sample_sessions(cal, kinds, 10, "2020-01-01", rng("test"))
    b = spotcheck.sample_sessions(cal, kinds, 10, "2020-01-01", rng("test"))
    assert a.equals(b) and (kinds[a] == "regular").all()
