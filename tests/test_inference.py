"""HAC and bootstrap checked against cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from src import inference
from src.config import rng


def _ar1(phi: float, n: int, seed_stream: str = "test") -> np.ndarray:
    g = rng(seed_stream)
    e = g.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_lag_rule() -> None:
    assert inference.nw_lag(100) == 4
    assert inference.nw_lag(7600) == 10


def test_iid_hac_se_matches_the_textbook_se() -> None:
    x = rng("test").normal(size=20_000)
    naive = x.std(ddof=0) / np.sqrt(len(x))
    assert inference.newey_west_se(x) == pytest.approx(naive, rel=0.05)


def test_hac_se_widens_for_autocorrelated_data() -> None:
    """AR(1), phi = 0.5: long-run variance is sigma^2 / (1 - phi)^2 = 4x the iid one."""
    x = _ar1(0.5, 50_000)
    true_se = np.sqrt(1 / (1 - 0.5) ** 2 / len(x))
    naive = x.std() / np.sqrt(len(x))
    hac = inference.newey_west_se(x, lag=40)
    assert hac == pytest.approx(true_se, rel=0.15)
    assert hac > 1.6 * naive


def test_mean_test_drops_nans() -> None:
    m = inference.hac_mean_test([0.01, np.nan, 0.03, 0.02, np.nan, 0.02])
    assert m.n == 4 and m.mean == pytest.approx(0.02)


def test_bootstrap_is_reproducible_and_centred() -> None:
    x = rng("test").normal(0.001, 0.01, size=3_000)
    a = inference.stationary_bootstrap_means(x, 500, rng("bootstrap"))
    b = inference.stationary_bootstrap_means(x, 500, rng("bootstrap"))
    assert np.array_equal(a, b)
    assert a.mean() == pytest.approx(x.mean(), abs=0.2 * x.std() / np.sqrt(len(x)))


def test_block_one_is_the_iid_bootstrap() -> None:
    x = rng("test").normal(size=4_000)
    means = inference.stationary_bootstrap_means(x, 2_000, rng("bootstrap"), block=1)
    assert means.std() == pytest.approx(x.std() / np.sqrt(len(x)), rel=0.1)


def test_one_endless_block_is_a_rotation() -> None:
    """With no jumps after the first step, each resample is x rotated: same mean."""
    x = rng("test").normal(size=500)
    means = inference.stationary_bootstrap_means(x, 50, rng("bootstrap"), block=1e12)
    np.testing.assert_allclose(means, x.mean(), rtol=1e-12)


def test_blocks_widen_the_ci_for_autocorrelated_data() -> None:
    x = _ar1(0.5, 6_000)
    iid = inference.bootstrap_ci(x, 0.95, 1_000, rng("bootstrap"), block=1)
    blocks = inference.bootstrap_ci(x, 0.95, 1_000, rng("bootstrap"), block=30)
    assert (blocks[1] - blocks[0]) > 1.5 * (iid[1] - iid[0])
    assert blocks[0] < x.mean() < blocks[1]
