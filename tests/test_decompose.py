"""The night/day split: the identity, the chaining, and what rejected returns do."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, decompose
from src.universe import instruments


def _clean(n: int = 300, seed_stream: str = "test") -> pd.DataFrame:
    """A random-walk series shaped like a Phase 2 output, every return accepted."""
    g = config.rng(seed_stream)
    night = g.normal(0.0008, 0.006, n)
    day = g.normal(-0.0004, 0.010, n)
    close = 100 * np.cumprod((1 + night) * (1 + day))
    open_ = np.concatenate([[100 * (1 + night[0])], close[:-1] * (1 + night[1:])])
    return pd.DataFrame({
        "Open": open_, "Close": close, "night_ok": True, "day_ok": True,
    }, index=pd.bdate_range("2020-01-01", periods=n, name="Date"))


def test_identity_holds_to_float_precision() -> None:
    assert decompose.identity_error(_clean()) < 1e-12


def test_split_hold_is_exactly_night_times_day() -> None:
    c = decompose.curves(_clean())
    np.testing.assert_allclose(c["split_hold"], c["night"] * c["day"], rtol=1e-12)


def test_with_nothing_rejected_split_hold_is_buy_and_hold() -> None:
    clean = _clean()
    clean.iloc[0, clean.columns.get_loc("night_ok")] = False  # the first night has no previous close
    c = decompose.curves(clean)
    np.testing.assert_allclose(c["split_hold"], c["hold"], rtol=1e-10)


def test_a_rejected_return_leaves_its_leg_flat() -> None:
    clean = _clean()
    clean.loc[clean.index[[0, 50]], "night_ok"] = False
    clean.loc[clean.index[80], "day_ok"] = False
    legs = decompose.leg_returns(clean)
    assert legs["night"].isna().sum() == 2 and legs["day"].isna().sum() == 1
    c = decompose.curves(clean, legs)
    assert c["night"].iloc[50] == c["night"].iloc[49]
    assert c["day"].iloc[80] == c["day"].iloc[79]
    # buy-and-hold still has those returns, the split does not
    assert not np.isclose(c["hold"].iloc[-1], c["split_hold"].iloc[-1])


def test_the_open_cancels_so_only_misaligned_closes_break_it() -> None:
    """What the identity can and cannot see."""
    clean = _clean()
    late_opens = clean.assign(Open=clean["Open"].shift(1).bfill())
    assert decompose.identity_error(late_opens) < 1e-12  # wrong opens: invisible

    legs = decompose.leg_returns(clean)
    # night measured from a close two rows back instead of one
    bad_night = clean["Open"] / clean["Close"].shift(2) - 1
    lhs = (1 + bad_night) * (1 + legs["day"])
    rhs = clean["Close"] / clean["Close"].shift(1)
    assert (lhs - rhs).abs().max() > 1e-4


def test_identity_on_every_cleaned_series() -> None:
    """The real data, if Phase 2 has been run on this machine."""
    tickers = [t for t in instruments() if decompose.cache_path(t, config.DATA_CLEAN).exists()]
    if not tickers:
        pytest.skip("no data/clean/ yet; run scripts.phase2_validate")
    worst = {t: decompose.identity_error(decompose.load_clean(t)) for t in tickers}
    assert max(worst.values()) < 1e-10, {t: e for t, e in worst.items() if e >= 1e-10}
