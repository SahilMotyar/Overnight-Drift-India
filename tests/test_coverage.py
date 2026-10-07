"""The coverage counts the Phase 1 checkpoint reports."""

from __future__ import annotations

import pandas as pd
import pytest

from src.coverage import coverage_row, coverage_table, reference_calendar

REF = pd.bdate_range("2026-01-05", "2026-01-16")  # two full weeks, 10 days


def test_complete_series_has_nothing_missing() -> None:
    row = coverage_row(REF, REF)
    assert row["rows"] == 10 and row["missing"] == 0 and row["extra"] == 0
    assert row["max_gap_days"] == 3  # Friday -> Monday


def test_counts_missing_extra_duplicate_and_weekend_rows() -> None:
    dates = (
        REF.drop(pd.Timestamp("2026-01-08"))           # one missing weekday
        .append(pd.DatetimeIndex(["2026-01-10"]))      # a Saturday session
        .append(pd.DatetimeIndex(["2026-01-12"]))      # a duplicated row
    )
    row = coverage_row(dates, REF)
    assert row["missing"] == 1
    assert row["missing_%"] == pytest.approx(10.0)
    assert row["extra"] == 1 and row["weekend"] == 1
    assert row["dup_dates"] == 1
    assert row["rows"] == 11


def test_missing_counts_only_inside_the_series_own_span() -> None:
    """A stock listed mid-period is not 'missing' the days before it listed."""
    row = coverage_row(REF[5:], REF)
    assert row["missing"] == 0
    assert str(row["first"]) == "2026-01-12"


def test_rows_before_the_reference_are_not_called_extra() -> None:
    early = pd.bdate_range("2025-12-29", "2026-01-02")  # 5 days before REF starts
    row = coverage_row(early.append(REF), REF)
    assert row["before_ref"] == 5 and row["extra"] == 0


def test_a_series_that_stops_early_is_behind() -> None:
    row = coverage_row(REF[:-2], REF)
    assert row["behind"] == 2 and row["missing"] == 0


def test_reference_calendar_is_a_union_and_needs_both_indices() -> None:
    a = pd.DataFrame(index=REF[:6])
    b = pd.DataFrame(index=REF[4:])
    assert reference_calendar({"A": a, "B": b}, tickers=("A", "B")).equals(REF)
    with pytest.raises(KeyError, match="failed to download"):
        reference_calendar({"A": a}, tickers=("A", "B"))


def test_table_is_indexed_by_ticker() -> None:
    table = coverage_table({"X.NS": pd.DataFrame(index=REF)}, REF)
    assert table.index.tolist() == ["X.NS"] and table.index.name == "ticker"
