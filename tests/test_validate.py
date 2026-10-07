"""Each Phase 2 check flags exactly the pattern it is named for."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import validate


def _frame(rows: list[tuple[float, float, float, float, int]]) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-05", periods=len(rows), name="Date")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)


NORMAL = (100.0, 101.0, 99.0, 100.5, 1000)


def test_nonpositive_and_inconsistent_bars() -> None:
    df = _frame([NORMAL, (0.0, 101, 99, 100, 10), (98.0, 101, 99, 100, 10), (100, 99, 101, 100, 10)])
    assert validate.nonpositive(df).tolist() == [False, True, False, False]
    # open below low; high below low
    assert validate.ohlc_inconsistent(df).tolist() == [False, True, True, True]


def test_placeholder_is_flat_at_previous_close_with_no_volume() -> None:
    df = _frame([
        NORMAL,
        (100.5, 100.5, 100.5, 100.5, 0),   # placeholder
        (101.0, 101.0, 101.0, 101.0, 0),   # flat, zero volume, but a new price
        (101.0, 101.0, 101.0, 101.0, 50),  # flat but traded
    ])
    assert validate.placeholder(df).tolist() == [False, True, False, False]
    assert validate.no_trades(df).tolist() == [False, True, True, False]


def test_single_row_scale_glitch() -> None:
    """TATASTEEL-style: one row at ~10x, then straight back."""
    close = pd.Series([100, 101, 1050, 102, 103], dtype=float)
    assert validate.scale_glitches(close).tolist() == [False, False, True, False, False]


def test_multi_row_scale_glitch() -> None:
    """NIFTYBEES 2019-12-19/20 style: two rows at 1/10 scale."""
    close = pd.Series([120, 121, 12.1, 12.2, 122, 123], dtype=float)
    assert validate.scale_glitches(close).tolist() == [False, False, True, True, False, False]


def test_a_real_level_shift_is_not_a_glitch() -> None:
    """A big move that does not come back (e.g. a demerger) is left to other checks."""
    close = pd.Series([100, 101, 60, 61, 60, 62, 61, 63], dtype=float)
    assert not validate.scale_glitches(close).any()


def test_excursion_longer_than_max_rows_is_not_a_glitch() -> None:
    close = pd.Series([100] + [200] * 7 + [100], dtype=float)
    assert not validate.scale_glitches(close, max_rows=5).any()


def test_stale_open() -> None:
    df = _frame([NORMAL, (100.5, 101, 99, 100, 10), (100.4, 101, 99, 100, 10)])
    assert validate.stale_open(df).tolist() == [False, True, False]


def test_extreme_night_is_symmetric_in_simple_returns() -> None:
    df = _frame([NORMAL, (79.0, 80, 78, 79, 10), (100.0, 101, 99, 100, 10), (100.0, 101, 99, 100, 10)])
    # 100.5 -> 79 is -21%; 79 -> 100 is +27%; 100 -> 100 is 0
    assert validate.extreme_night(df).tolist() == [False, True, True, False]


def test_reversed_open_needs_opposite_cancelling_moves() -> None:
    df = _frame([
        NORMAL,
        (80.0, 101, 79, 100.0, 10),   # open -20% then day +25%: back to ~prev close
        (80.0, 81, 79, 80.0, 10),     # a -20% gap that holds
    ])
    flags = validate.reversed_open(df).tolist()
    assert flags == [False, True, False]


def test_checks_never_modify_their_input() -> None:
    df = _frame([NORMAL, (100.5, 100.5, 100.5, 100.5, 0)])
    before = df.copy()
    for check in (validate.nonpositive, validate.ohlc_inconsistent, validate.placeholder,
                  validate.no_trades, validate.stale_open, validate.extreme_night, validate.reversed_open):
        check(df)
    validate.scale_glitches(df["Close"])
    pd.testing.assert_frame_equal(df, before)
    assert np.isfinite(df.to_numpy()).all()
