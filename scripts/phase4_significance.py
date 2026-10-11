"""Phase 4 checkpoint: is the night/day split more than noise?

    python -m scripts.phase4_significance

Reads data/clean/ (Phase 2 output) and the Phase 1 cache (for volume).
Writes:
  reports/phase4_tests.csv          every series x period: means, Newey-West t, gap CI
  reports/phase4_cross_section.csv  per stock, 2011 on: night, day, gap, traded value
"""

from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import config, decompose, significance
from src.data import load
from src.universe import instruments

OUT = {
    "tests": config.REPORTS / "phase4_tests.csv",
    "xs": config.REPORTS / "phase4_cross_section.csv",
}
EW = "EW stocks"
HEADLINE = ["^NSEI", "^BSESN", "^NSEBANK", EW, "NIFTYBEES.NS"]
PERIODS = [significance.FULL, *config.SUBPERIODS]


def _heading(text: str) -> None:
    print(f"\n{'=' * 8} {text} {'=' * max(0, 70 - len(text))}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", None)

    inst = instruments()
    kind_of = {t: k for t, (_, k) in inst.items()}
    stocks = [t for t in inst if kind_of[t] == "stock"]
    clean = {t: decompose.load_clean(t) for t in inst}
    legs = {t: decompose.leg_returns(c) for t, c in clean.items()}
    legs[EW] = significance.equal_weight({t: legs[t] for t in stocks})
    kind_of[EW] = "portfolio"

    # Every series x period -----------------------------------------------
    tests = pd.concat(
        [significance.leg_tests(legs[s]).assign(series=s, kind=kind_of[s]) for s in legs],
        ignore_index=True,
    )
    tests = tests[["series", "kind", *tests.columns.drop(["series", "kind"])]]
    tests.to_csv(OUT["tests"], index=False)

    _heading("Indices, the ETF and the equal-weight stock portfolio (bp per session)")
    cols = ["first", "last", "night_n", "night_mean_bp", "night_t", "day_mean_bp", "day_t",
            "gap_mean_bp", "gap_t", "gap_ci_lo_bp", "gap_ci_hi_bp"]
    head = tests[tests["series"].isin(HEADLINE)].set_index(["series", "period"]).loc[HEADLINE][cols]
    print(head.round(2).to_string())
    print("\nNIFTYBEES is shown for completeness only: Phase 3 found its night/day split comes from its "
          "official opening price (far above the index's open from 2019), not the market.")

    # Stocks: how many pass, and at what bar ----------------------------
    n_tests = len(stocks) * len(PERIODS)
    bonferroni = NormalDist().inv_cdf(1 - 0.05 / (2 * n_tests))
    _heading(f"Stocks: how many clear |t| > 2, and the Bonferroni bar |t| > {bonferroni:.2f}")
    st = tests[tests["kind"] == "stock"]
    rows = []
    for p in PERIODS:
        g = st[(st["period"] == p) & st["gap_t"].notna()]
        rows.append({
            "period": p,
            "stocks_tested": len(g),
            "night_t>2": int((g["night_t"] > 2).sum()),
            "night_t<-2": int((g["night_t"] < -2).sum()),
            "day_t<-2": int((g["day_t"] < -2).sum()),
            "day_t>2": int((g["day_t"] > 2).sum()),
            "gap_t>2": int((g["gap_t"] > 2).sum()),
            "gap_t<-2": int((g["gap_t"] < -2).sum()),
            f"gap_t>{bonferroni:.2f}": int((g["gap_t"] > bonferroni).sum()),
            "gap_CI_above_0": int((g["gap_ci_lo_bp"] > 0).sum()),
            "median_gap_bp": round(g["gap_mean_bp"].median(), 1),
        })
    print(pd.DataFrame(rows).set_index("period").to_string())
    print(f"\n{n_tests} stock tests in all. At |t| > 2, chance alone gives about "
          f"{0.025 * n_tests:.0f} false positives per side; the Bonferroni bar holds the family-wide "
          f"false-positive rate at 5%.")

    # Cross-section, 2011 on ----------------------------------------------
    start = config.CROSS_SECTION_FROM
    xs_rows = []
    for t in stocks:
        lg = legs[t].loc[legs[t].index >= pd.Timestamp(start)]
        have = lg.dropna(how="all").index
        if len(have) == 0:
            continue
        years = (have[-1] - have[0]).days / 365.25
        gap = significance.with_gap(lg)["gap"]
        xs_rows.append({
            "ticker": t,
            "years": round(years, 1),
            "night_mean_bp": significance.BP * lg["night"].mean(),
            "day_mean_bp": significance.BP * lg["day"].mean(),
            "gap_mean_bp": significance.BP * gap.mean(),
            "traded_value_cr": significance.traded_value(load(t, adjusted=False).loc[clean[t].index], start) / 1e7,
        })
    xs = pd.DataFrame(xs_rows).set_index("ticker")
    xs.to_csv(OUT["xs"])
    eligible = xs[xs["years"] >= config.CROSS_SECTION_MIN_YEARS]
    _heading(f"Cross-section from {start}: does the gap scale with liquidity? ({len(eligible)} stocks, "
             f">= {config.CROSS_SECTION_MIN_YEARS:g} years)")
    for y in ("gap_mean_bp", "night_mean_bp", "day_mean_bp"):
        r = significance.cross_section(eligible, "traded_value_cr", y, config.rng("crosssection"))
        print(f"Spearman rho(traded value, {y:13s}) = {r['rho']:+.2f}   permutation p = {r['p']:.3f}   n = {r['n']}")
    q = pd.qcut(eligible["traded_value_cr"], 3, labels=["least liquid third", "middle third", "most liquid third"])
    print("\nby liquidity tercile (median of per-stock means, bp):")
    print(eligible.groupby(q, observed=True)[["traded_value_cr", "night_mean_bp", "day_mean_bp", "gap_mean_bp"]]
          .median().round(1).to_string())
    print("\nNot tested: size (no market-cap history) and F&O inclusion (every current constituent is in F&O).")
    print("Caveat: survivorship. These are today's Nifty 50, so buy-and-hold is flattered. The night/day "
          "split is a different question, but it is still only measured on survivors.")

    print("\nWrote:", ", ".join(str(p.relative_to(config.ROOT)) for p in OUT.values()))


if __name__ == "__main__":
    main()
