"""The Phase 2 pipeline: what it removes, what it rejects, and that it logs all of it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clean, config, sessions


def _raw(dates, close=None, **overrides) -> pd.DataFrame:
    """Raw-cache-shaped frame; each row a normal bar unless overridden by date."""
    idx = pd.DatetimeIndex(dates, name="Date")
    c = np.asarray(close if close is not None else np.linspace(100, 110, len(idx)), dtype=float)
    df = pd.DataFrame({
        "Open": c * 0.999, "High": c * 1.01, "Low": c * 0.99, "Close": c,
        "Adj Close": c, "Volume": 1000, "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=idx)
    for day, row in overrides.items():
        df.loc[pd.Timestamp(day.replace("_", "-")), list(row)] = list(row.values())
    return df


CAL = pd.bdate_range("2026-01-05", periods=15, name="Date")
KINDS = sessions.session_kinds(CAL, muhurat=())


def test_row_stage_removes_and_logs_each_defect() -> None:
    raw = _raw(CAL[:6])
    raw.iloc[1, :4] = [0.0, 101, 99, 100]              # price <= 0
    raw.iloc[3, :4] = [raw.iloc[2, 3]] * 4             # placeholder at prev close
    raw.iloc[3, raw.columns.get_loc("Volume")] = 0
    log = clean.DropLog()
    out = clean.row_stage(raw, "X.NS", is_index=False, log=log)
    assert len(out) == 4
    reasons = dict(zip(log.table()["reason"], log.table()["count"]))
    assert reasons["price <= 0"] == 1
    assert reasons["placeholder: flat at previous close, no volume"] == 1


def test_start_date_skips_a_sparse_early_stretch() -> None:
    cal = pd.bdate_range("2026-01-05", periods=30)
    dates = cal[[0, 3, 6, 9]].append(cal[10:])         # sparse, then complete from cal[9]
    assert clean.start_date(dates, cal, window=5, min_coverage=0.9) == cal[9]
    assert clean.start_date(cal, cal, window=5) == cal[0]


def test_finish_flags_gap_special_and_extreme_nights() -> None:
    cal = CAL
    kinds = KINDS.copy()
    kinds[cal[10]] = "muhurat"
    close = np.full(len(cal), 100.0)
    close[6:] = 75.0                                    # a -25% level shift that stays
    raw = _raw(cal.drop(cal[3]), close=np.delete(close, 3))
    log = clean.DropLog()
    out, summary = clean.finish(raw, "X.NS", False, cal, kinds, log)

    note = out["night_note"]
    assert note[cal[0]] == "first row: no previous close"
    assert note[cal[4]] == "spans a session the series lacks"
    assert note[cal[6]] == "night beyond +-20%"
    assert note[cal[10]] == note[cal[11]] == "special session at either end"
    assert not out.loc[cal[10], "day_ok"]
    assert out["night_ok"].sum() == len(out) - 5
    assert summary["nights_ok"] == out["night_ok"].sum()

    # every rejection is in the log exactly once
    t = log.table()
    assert t.loc[t.stage == "night", "count"].sum() == (~out["night_ok"]).sum()
    assert t.loc[t.stage == "day", "count"].sum() == (~out["day_ok"]).sum()


def test_off_calendar_rows_and_glitches_are_removed() -> None:
    dates = CAL[:8].append(pd.DatetimeIndex(["2026-01-10"]))  # a Saturday nobody traded
    close = np.full(len(dates), 100.0)
    close[4] = 1000.0                                          # one row at 10x
    raw = _raw(dates, close=close).sort_index()
    log = clean.DropLog()
    out, _ = clean.finish(raw, "X.NS", False, CAL, KINDS, log)
    assert pd.Timestamp("2026-01-10") not in out.index
    assert dates[4] not in out.index
    assert out.loc[dates[5], "night_kind"] == "gap"            # the hole the glitch left
    rows = log.table().query("stage == 'row'").set_index("reason")["count"]
    assert rows["not an NSE session"] == 1
    assert rows["scale glitch: jump that reverts within 5 rows"] == 1


def test_stale_open_rejected_for_indices_only() -> None:
    raw = _raw(CAL[:4])
    raw.loc[CAL[2], "Open"] = raw.loc[CAL[1], "Close"]
    idx, _ = clean.finish(raw, "^X", True, CAL, KINDS, clean.DropLog())
    stock, _ = clean.finish(raw, "X.NS", False, CAL, KINDS, clean.DropLog())
    assert not idx.loc[CAL[2], "night_ok"] and not idx.loc[CAL[2], "day_ok"]
    assert stock.loc[CAL[2], "night_ok"] and stock.loc[CAL[2], "day_ok"]


def test_reversed_open_is_flagged_not_rejected() -> None:
    raw = _raw(CAL[:4], close=[100.0, 100.0, 100.0, 100.0])
    raw.loc[CAL[2], ["Open", "Low"]] = [80.0, 79.0]      # opens -20%, closes back at 100
    out, summary = clean.finish(raw, "X.NS", False, CAL, KINDS, clean.DropLog())
    assert out.loc[CAL[2], "reversed_open"]
    assert out.loc[CAL[2], "night_ok"] and out.loc[CAL[2], "day_ok"]
    assert summary["reversed_opens"] == 1


def test_closed_days_include_saturdays_and_diwali_sundays() -> None:
    cal = pd.bdate_range("2026-10-01", "2026-11-03").drop(pd.Timestamp("2026-10-02"))
    got = sessions.closed_days_to_check(cal)
    assert pd.Timestamp("2026-10-02") in got           # weekday holiday
    assert pd.Timestamp("2026-10-03") in got           # Saturday
    assert pd.Timestamp("2026-10-04") in got           # Sunday in October
    assert pd.DatetimeIndex(got).isin(cal).sum() == 0


def test_sessions_no_series_has_do_not_move_the_start() -> None:
    # 30 sessions NSE traded but Yahoo has nothing for: enough to sink a
    # 250-session coverage window if they were counted against the series.
    cal = pd.bdate_range("2020-01-06", periods=400)
    hidden = cal[100:130]
    observed = cal.drop(hidden)
    rows = _raw(observed)
    kinds = sessions.session_kinds(cal, muhurat=())
    _, naive = clean.finish(rows, "X.NS", False, cal, kinds, clean.DropLog())
    out, fixed = clean.finish(rows, "X.NS", False, cal, kinds, clean.DropLog(), coverage_calendar=observed)
    assert naive["start"] == cal[130].date()
    assert fixed["start"] == cal[0].date()
    assert out.loc[cal[130], "night_kind"] == "gap"  # the night across them is still rejected


def test_start_override_is_a_floor(monkeypatch) -> None:
    monkeypatch.setitem(config.START_OVERRIDES, "X.NS", (str(CAL[5].date()), "test"))
    out, summary = clean.finish(_raw(CAL), "X.NS", False, CAL, KINDS, clean.DropLog())
    assert out.index[0] == CAL[5] and summary["start_by"] == "override"


def test_output_is_dividend_adjusted() -> None:
    raw = _raw(CAL[:3], close=[100.0, 99.0, 100.0])
    raw["Dividends"] = [0.0, 2.0, 0.0]
    raw["Adj Close"] = raw["Close"] * np.array([0.98, 1.0, 1.0])
    out, _ = clean.finish(raw, "X.NS", False, CAL, KINDS, clean.DropLog())
    np.testing.assert_allclose(out["Close"], raw["Adj Close"])
    assert list(out.columns) == clean.CLEAN_COLUMNS


def test_rows_not_on_the_calendar_never_reach_tag_nights() -> None:
    with pytest.raises(ValueError):
        sessions.tag_nights(pd.DatetimeIndex(["2026-01-10"]), CAL)
