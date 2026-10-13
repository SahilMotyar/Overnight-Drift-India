"""Phase 6 checkpoint: equity curves after costs, and what the pattern is worth for free.

    python -m scripts.phase6_backtest

Reads data/clean/ (Phase 2 output). Writes:
  reports/phase6_strategies.csv       S0-S3 per series x period, gross and net of costs
  reports/phase6_sip.csv              monthly SIP bought at the open / close / next open, per series
  reports/figures/phase6_net_equity.png
"""

from __future__ import annotations

import sys

import pandas as pd

from src import backtest, config, decompose, figures, significance
from src.universe import instruments

FIG = config.FIGURES / "phase6_net_equity.png"
OUT = {
    "strategies": config.REPORTS / "phase6_strategies.csv",
    "sip": config.REPORTS / "phase6_sip.csv",
}
EW = "EW stocks"
SERIES = ["^NSEI", "^NSEBANK", "^BSESN", EW]
LATEST = list(config.SUBPERIODS)[-1]


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
    hold = {t: (c["Close"] / c["Close"].shift(1) - 1).where(~c["night_kind"].isin(["first", "gap"]))
            for t, c in clean.items()}
    legs[EW] = significance.equal_weight({t: legs[t] for t in stocks})
    hold[EW] = pd.DataFrame({t: hold[t] for t in stocks}).mean(axis=1)
    kind_of[EW] = "portfolio"

    # S0-S3, gross and net --------------------------------------------------
    rows, net_equity = [], {}
    for s in SERIES:
        nights = clean[s]["night_days"] if kind_of[s] == "index" else None
        for basis in ("gross", "net"):
            table = backtest.strategy_returns_table(legs[s], hold[s], kind_of[s], night_days=nights,
                                                    net=basis == "net")
            for period, mask in significance.period_masks(table.index).items():
                if period not in (significance.FULL, LATEST):
                    continue
                t = table[mask]
                start, end = t.dropna(how="all").index[[0, -1]]
                years = (end - start).days / 365.25
                for col in backtest.STRATEGY_COLUMNS:
                    trips = 0 if col == "S0 hold" else t[col].notna().sum() * (2 if col == "S3 both" else 1)
                    rows.append({"series": s, "period": period, "basis": basis, "strategy": col,
                                 **backtest.stats(t[col], start, end),
                                 "round_trips_per_year": trips / years})
            if basis == "net":
                net_equity[s] = pd.DataFrame({c: backtest.equity(table[c]) for c in table})
    strat = pd.DataFrame(rows)
    strat.to_csv(OUT["strategies"], index=False)

    # S4: SIP timing ---------------------------------------------------------
    sip_rows = []
    for t in [s for s in SERIES if s != EW] + stocks:
        c = clean[t]
        sip_rows.append({"series": t, "kind": kind_of[t], "first": c.index[0].date(),
                         **backtest.sip(c, backtest.sip_dates(c))})
    sip = pd.DataFrame(sip_rows).set_index("series")
    sip.to_csv(OUT["sip"])

    # Chart ------------------------------------------------------------------
    names = {"^NSEI": "Nifty 50 via futures", EW: "Equal-weight Nifty 50 stocks"}
    figures.strategy_panels(
        {names[s]: net_equity[s] for s in names}, FIG,
        "After today's costs, trading the night or the day loses to doing nothing",
        "Growth of ₹1 after round-trip costs (Phase 5); buy-and-hold pays none. Log scale, each "
        "panel on its own. Every number: reports/phase6_strategies.csv.")

    # Checkpoint print -------------------------------------------------------
    cols = ["cagr_%", "vol_%", "ret_per_vol", "max_drawdown_%", "final_x", "round_trips_per_year"]
    for period in (significance.FULL, LATEST):
        _heading(f"S0-S3, gross vs net of costs ({period})")
        t = strat[strat["period"] == period].set_index(["series", "strategy", "basis"])[cols]
        t = t.unstack("basis")
        t.columns = [f"{a} {b}" for a, b in t.columns]
        keep = [f"{c} {b}" for c in ("cagr_%", "ret_per_vol", "max_drawdown_%") for b in ("gross", "net")]
        keep.append("round_trips_per_year net")
        print(t.reindex([(s, c) for s in SERIES for c in backtest.STRATEGY_COLUMNS])[keep].round(2).to_string())
    print("\nret_per_vol: annualised mean / annualised volatility, risk-free rate 0. S3 pays two round trips a day.")

    _heading("S4: a monthly SIP bought at the close instead of the open (free: same trades)")
    show = ["first", "instalments", "edge_vs_open_bp", "close_vs_open_%", "edge_vs_next_open_bp",
            "close_vs_next_open_%"]
    print(sip.loc[[s for s in SERIES if s != EW], show].round(3).to_string())
    st = sip.loc[stocks]
    print(f"\nAcross the {len(stocks)} stocks: close beats that morning's open for "
          f"{(st['close_vs_open_%'] > 0).sum()}, and beats the next morning's open for "
          f"{(st['close_vs_next_open_%'] > 0).sum()}.")
    print(st[["edge_vs_open_bp", "close_vs_open_%", "edge_vs_next_open_bp", "close_vs_next_open_%"]]
          .describe().loc[["min", "25%", "50%", "75%", "max"]].round(3).to_string())
    print("\nedge_*_bp: mean saving per instalment from buying at the close. close_vs_*_%: how much "
          "larger the final SIP value is. An index fund already buys at the close (same-day NAV); an "
          "order placed after the close for the next open does not.")

    print("\nWrote:", ", ".join(str(p.relative_to(config.ROOT)) for p in OUT.values()),
          "and", FIG.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
