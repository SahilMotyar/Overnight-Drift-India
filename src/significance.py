"""Phase 4: is the night/day split more than noise?

Every test is on returns Phase 2 accepted, in bp per session:

* the night mean and the day mean, each with a Newey-West t-statistic
* the gap: night - day on the same date (both accepted), with its
  t-statistic and a stationary-bootstrap CI
* in the full sample and in each subperiod fixed in ``config.SUBPERIODS``
  before any result was seen

The equal-weight stock portfolio (each date: the mean over the stocks that
have that return) is the one aggregate test for the stocks, so the headline
does not rest on 50 separate tests.

Bootstrap draws come from a fresh ``rng("bootstrap")`` for every test, so
each CI is reproducible on its own, whatever order the tests run in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .inference import bootstrap_ci, hac_mean_test

BP = 1e4
FULL = "full"
LEGS = ("night", "day", "gap")


def with_gap(legs: pd.DataFrame) -> pd.DataFrame:
    """``legs`` plus ``gap = night - day``, NaN unless both are accepted."""
    return legs.assign(gap=legs["night"] - legs["day"])


def period_masks(
    index: pd.DatetimeIndex, subperiods: dict[str, tuple[str | None, str | None]] = config.SUBPERIODS
) -> dict[str, np.ndarray]:
    """``{"full": all rows, name: rows inside that subperiod}`` (inclusive, open-ended on None)."""
    masks = {FULL: np.ones(len(index), dtype=bool)}
    for name, (start, end) in subperiods.items():
        m = np.ones(len(index), dtype=bool)
        if start:
            m &= index >= pd.Timestamp(start)
        if end:
            m &= index <= pd.Timestamp(end)
        masks[name] = m
    return masks


def leg_tests(
    legs: pd.DataFrame,
    reps: int = config.BOOTSTRAP_REPS,
    level: float = config.CI_LEVEL,
    min_n: int = config.MIN_TEST_SESSIONS,
) -> pd.DataFrame:
    """One row per period: n, mean bp and HAC t for each leg and the gap, plus the gap's CI.

    A period with fewer than ``min_n`` accepted returns in a leg is left NaN
    for that leg rather than tested on a handful of days.
    """
    frame = with_gap(legs)
    rows = []
    for name, mask in period_masks(frame.index).items():
        f = frame[mask]
        have = f.dropna(how="all").index
        row: dict[str, object] = {
            "period": name,
            "first": have[0].date() if len(have) else None,
            "last": have[-1].date() if len(have) else None,
        }
        for leg in LEGS:
            x = f[leg].dropna().to_numpy()
            row[f"{leg}_n"] = len(x)
            if len(x) < min_n:
                row[f"{leg}_mean_bp"] = row[f"{leg}_t"] = np.nan
                continue
            m = hac_mean_test(x)
            row[f"{leg}_mean_bp"] = BP * m.mean
            row[f"{leg}_t"] = m.t
        gap = f["gap"].dropna().to_numpy()
        if len(gap) >= min_n:
            lo, hi = bootstrap_ci(gap, level, reps, config.rng("bootstrap"))
            row["gap_ci_lo_bp"], row["gap_ci_hi_bp"] = BP * lo, BP * hi
        else:
            row["gap_ci_lo_bp"] = row["gap_ci_hi_bp"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def equal_weight(legs_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per date, the mean night and mean day over the series that have them."""
    return pd.DataFrame({
        leg: pd.DataFrame({t: lg[leg] for t, lg in legs_by_ticker.items()}).mean(axis=1)
        for leg in ("night", "day")
    })


def traded_value(raw: pd.DataFrame, start: str) -> float:
    """Median daily traded value (Rs) from ``start``: Close x Volume on Yahoo's raw prices.

    Yahoo split-adjusts both (price / k, volume x k), so the product is the
    real rupee value whatever splits came later.
    """
    r = raw.loc[raw.index >= pd.Timestamp(start)]
    value = r["Close"] * r["Volume"]
    return float(value[value > 0].median())


def _ranks(x) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()


def spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def permutation_p(a, b, rng: np.random.Generator, perms: int = config.PERMUTATIONS) -> float:
    """Two-sided permutation p-value for Spearman's rho: how often shuffled pairs
    correlate at least as strongly. (count + 1) / (perms + 1), so never 0."""
    ra, rb = _ranks(a), _ranks(b)
    za = (ra - ra.mean()) / ra.std()
    zb = (rb - rb.mean()) / rb.std()
    observed = abs(za @ zb / len(za))
    shuffled = rng.permuted(np.tile(zb, (perms, 1)), axis=1)
    null = np.abs(shuffled @ za / len(za))
    return float((np.sum(null >= observed - 1e-12) + 1) / (perms + 1))


def cross_section(per_stock: pd.DataFrame, x: str, y: str, rng: np.random.Generator,
                  perms: int = config.PERMUTATIONS) -> dict[str, float]:
    """Spearman rho between two per-stock columns, with a permutation p-value."""
    d = per_stock[[x, y]].dropna()
    return {"n": len(d), "rho": spearman(d[x], d[y]), "p": permutation_p(d[x], d[y], rng, perms)}
