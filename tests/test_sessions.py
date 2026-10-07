"""Calendar construction and night labelling."""

from __future__ import annotations

import pandas as pd
import pytest

from src import sessions


def _frame(dates) -> pd.DataFrame:
    return pd.DataFrame({"Volume": 1}, index=pd.DatetimeIndex(dates))


def test_consensus_counts_only_series_alive_on_the_date() -> None:
    old = _frame(["2026-01-05", "2026-01-06", "2026-01-07"])
    new = _frame(["2026-01-06", "2026-01-07"])  # listed on the 6th
    junk = _frame(["2026-01-05", "2026-01-07"])  # missing the 6th
    share = sessions.consensus_share({"old": old, "new": new, "junk": junk}, min_series=2)
    # 5th: old+junk alive and present (new not yet listed) -> 1.0
    assert share[pd.Timestamp("2026-01-05")] == 1.0
    # 6th: all three alive, junk missing -> 2/3
    assert share[pd.Timestamp("2026-01-06")] == pytest.approx(2 / 3)


def test_too_few_live_series_cannot_vote() -> None:
    lone = _frame(["1995-01-02", "1995-01-03"])
    share = sessions.consensus_share({"lone": lone}, min_series=5)
    assert share.isna().all()
    assert len(sessions.build_calendar(share, pd.Series(dtype=bool))) == 0


def test_disputes_are_disagreements_and_split_dates() -> None:
    share = pd.Series(
        [1.0, 0.95, 0.6, 0.02, 1.0],
        index=pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]),
    )
    index_dates = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-09", "2026-01-12"])
    got = sessions.disputed_dates(share, index_dates)
    # 7th: stocks split (0.6). 12th: index only. 8th: 2% is too few to be a dispute.
    assert list(got) == [pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-12")]


def test_nse_verdict_overrides_consensus_both_ways() -> None:
    share = pd.Series([1.0, 1.0], index=pd.DatetimeIndex(["2026-01-05", "2026-01-26"]))
    verified = pd.Series([False, True], index=pd.DatetimeIndex(["2026-01-26", "2026-03-18"]))
    cal = sessions.build_calendar(share, verified)
    assert list(cal) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-18")]


def test_session_kinds() -> None:
    cal = pd.DatetimeIndex(["2025-01-31", "2025-02-01", "2025-10-21"])
    kinds = sessions.session_kinds(cal, muhurat=("2025-10-21",))
    assert kinds.tolist() == ["regular", "weekend", "muhurat"]


def test_night_labels() -> None:
    # Mon 5 .. Fri 9, Mon 12, Tue 13 (holiday Wed 14), Thu 15, Fri 16
    cal = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
                            "2026-01-12", "2026-01-13", "2026-01-15", "2026-01-16"])
    # the series is missing the 7th
    rows = cal.drop(pd.Timestamp("2026-01-07"))
    tags = sessions.tag_nights(rows, cal)
    assert tags["night_kind"].tolist() == [
        "first", "weeknight", "gap", "weeknight", "weekend", "weeknight", "holiday", "weeknight",
    ]
    assert tags["night_days"].tolist() == [0, 1, 2, 1, 3, 1, 2, 1]


def test_rows_off_the_calendar_must_be_dropped_first() -> None:
    cal = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    with pytest.raises(ValueError, match="not sessions"):
        sessions.tag_nights(pd.DatetimeIndex(["2026-01-05", "2026-01-10"]), cal)
