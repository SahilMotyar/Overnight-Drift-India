"""Phase 4 inference primitives: HAC t-statistics and a stationary block bootstrap.

numpy only (statsmodels was left out in Phase 0). Daily returns are
autocorrelated and their volatility clusters, so a plain t-statistic
overstates certainty:

* :func:`newey_west_se` corrects the standard error of a mean for both
  (Bartlett kernel, Newey-West 1994 lag rule).
* :func:`stationary_bootstrap_means` resamples random-length blocks
  (Politis & Romano 1994), so a bootstrap CI keeps the same dependence.

Missing values are dropped and the rest treated as one contiguous sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def nw_lag(n: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 (n / 100)^(2/9)). 10 for ~7,600 sessions."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_se(x, lag: int | None = None) -> float:
    """HAC standard error of the mean of ``x`` (Bartlett weights ``1 - l / (L + 1)``)."""
    x = _clean(x)
    n = len(x)
    if n < 2:
        return float("nan")
    lag = nw_lag(n) if lag is None else lag
    e = x - x.mean()
    s = e @ e / n
    for l in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return float(np.sqrt(s / n))


@dataclass(frozen=True)
class MeanTest:
    n: int
    mean: float
    se: float
    t: float
    lag: int


def hac_mean_test(x, lag: int | None = None) -> MeanTest:
    """Mean of ``x`` with its Newey-West standard error and t-statistic."""
    x = _clean(x)
    lag = nw_lag(len(x)) if lag is None else lag
    se = newey_west_se(x, lag)
    mean = float(x.mean()) if len(x) else float("nan")
    return MeanTest(n=len(x), mean=mean, se=se, t=mean / se if se > 0 else float("nan"), lag=lag)


def block_length(n: int) -> int:
    """Expected block length: n^(1/3), about 20 sessions for ~7,600."""
    return max(1, round(n ** (1 / 3)))


def stationary_bootstrap_means(
    x, reps: int, rng: np.random.Generator, block: float | None = None, chunk: int = 250
) -> np.ndarray:
    """Means of ``reps`` stationary-bootstrap resamples of ``x``.

    Each resample walks through ``x`` circularly and jumps to a fresh random
    position with probability ``1 / block`` at every step, so block lengths
    are geometric with mean ``block``. Built without a Python loop over time:
    a block's start time is the running maximum of jump times, and each
    position is its block's random start plus the steps since.
    """
    x = _clean(x)
    n = len(x)
    block = block_length(n) if block is None else block
    t = np.arange(n)
    out = []
    for done in range(0, reps, chunk):
        b = min(chunk, reps - done)
        jump = rng.random((b, n)) < 1.0 / block
        jump[:, 0] = True
        start_time = np.maximum.accumulate(np.where(jump, t, 0), axis=1)
        start_pos = np.take_along_axis(rng.integers(0, n, size=(b, n)), start_time, axis=1)
        idx = (start_pos + (t - start_time)) % n
        out.append(x[idx].mean(axis=1))
    return np.concatenate(out)


def bootstrap_ci(
    x, level: float, reps: int, rng: np.random.Generator, block: float | None = None
) -> tuple[float, float]:
    """Percentile CI for the mean of ``x`` from the stationary bootstrap."""
    means = stationary_bootstrap_means(x, reps, rng, block)
    tail = (1 - level) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return float(lo), float(hi)
