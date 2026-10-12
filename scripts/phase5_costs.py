"""Phase 5 checkpoint: does the night/day split survive Indian trading costs?

    python -m scripts.phase5_costs

Reads data/clean/ (Phase 2 output). Writes:
  reports/phase5_costs.csv       round-trip cost by component, per trade type (bp)
  reports/phase5_breakeven.csv   every series x strategy x period: break-evens, net results
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import breakeven, config, costs, decompose, significance
from src.universe import instruments

OUT = {
    "costs": config.REPORTS / "phase5_costs.csv",
    "be": config.REPORTS / "phase5_breakeven.csv",
}
EW = "EW stocks"
HEADLINE = ["^NSEI", "^NSEBANK", "^BSESN", EW, "NIFTYBEES.NS"]
PERIODS = [significance.FULL, *config.SUBPERIODS]
LATEST = list(config.SUBPERIODS)[-1]


def _heading(text: str) -> None:
    print(f"\n{'=' * 8} {text} {'=' * max(0, 70 - len(text))}")


def _hold_cagr(daily: pd.Series) -> tuple[float, float]:
    """(CAGR, years) of compounding a daily close-to-close return series."""
    d = daily.dropna()
    years = (d.index[-1] - d.index[0]).days / 365.25
    return float(np.exp(np.log1p(d).sum() / years) - 1), years


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", None)

    inst = instruments()
    kind_of = {t: k for t, (_, k) in inst.items()}
    stocks = [t for t in inst if kind_of[t] == "stock"]
    clean = {t: decompose.load_clean(t) for t in inst}
    legs = {t: decompose.leg_returns(c) for t, c in clean.items()}
    # Buy-and-hold, close to close, only across consecutive sessions (no gap nights).
    cc = {t: (c["Close"] / c["Close"].shift(1) - 1).where(~c["night_kind"].isin(["first", "gap"]))
          for t, c in clean.items()}
    legs[EW] = significance.equal_weight({t: legs[t] for t in stocks})
    cc[EW] = pd.DataFrame({t: cc[t] for t in stocks}).mean(axis=1)
    kind_of[EW] = "portfolio"

    cost_table = costs.cost_table()
    cost_table.to_csv(OUT["costs"])
    trip_cost = cost_table["total"].to_dict()

    rows = []
    for s in legs:
        kind = kind_of[s]
        nights = clean[s]["night_days"] if kind == "index" else None
        for strategy in breakeven.STRATEGIES:
            r_all = breakeven.strategy_returns(legs[s], kind, strategy, night_days=nights)
            trade_type = breakeven.TRADE_TYPE[kind][strategy]
            for period, mask in significance.period_masks(r_all.index).items():
                r = r_all[mask]
                if len(r) < config.MIN_TEST_SESSIONS:
                    continue
                hold_cagr, years = _hold_cagr(cc[s].loc[r.index[0]:r.index[-1]])
                be = breakeven.breakeven(r, years, hold_cagr if strategy == "long night" else None,
                                         trip_cost[trade_type])
                rows.append({"series": s, "kind": kind, "strategy": strategy, "trade_type": trade_type,
                             "period": period, "first": r.index[0].date(), "last": r.index[-1].date(), **be})
    table = pd.DataFrame(rows)
    table.to_csv(OUT["be"], index=False)

    # Checkpoint print ---------------------------------------------------
    _heading(f"Round-trip cost by component, bp (rates checked {config.COST_RATES_CHECKED})")
    print(cost_table.round(2).to_string())
    print(f"Order values: {config.ORDER_VALUE_INR}. Slippage per side: {config.SLIPPAGE_BP_PER_SIDE} bp.")
    print(f"Index futures also give up ~{config.FUTURES_CARRY_ANNUAL:.0%}/yr carry overnight "
          "(already in the gross returns below).")

    cols = ["round_trips", "gross_mean_bp", "gross_cagr_%", "be_cagr_bp", "be_hold_bp", "cost_bp",
            "cost_drag_%_per_year", "net_cagr_%", "hold_cagr_%", "room_bp"]
    for period in (significance.FULL, LATEST):
        _heading(f"Break-even per round trip, bp ({period})")
        t = table[(table["period"] == period) & table["series"].isin(HEADLINE)]
        t = t.set_index(["series", "strategy"]).reindex(
            [(s, st) for s in HEADLINE for st in breakeven.STRATEGIES]).dropna(how="all")
        print(t[["trade_type", *cols]].round(1).to_string())
    print("\nroom_bp = break-even (net CAGR 0) minus the estimated cost: negative means the trade "
          "loses money after costs. be_hold_bp is NaN where it never beats buy-and-hold.")
    print("NIFTYBEES: its opening price is distorted (Phase 3), so its 'edge' is not one you can count on.")

    _heading("Stocks one by one: how many survive costs")
    st = table[table["kind"] == "stock"]
    summary = st.groupby(["period", "strategy"]).agg(
        stocks=("series", "size"),
        net_cagr_positive=("net_cagr_%", lambda x: int((x > 0).sum())),
        beats_hold=("net_cagr_%", lambda x: np.nan),
        median_gross_mean_bp=("gross_mean_bp", "median"),
        median_be_cagr_bp=("be_cagr_bp", "median"),
        median_room_bp=("room_bp", "median"),
    )
    beats = st[st["strategy"] == "long night"].assign(
        b=lambda d: d["net_cagr_%"] > d["hold_cagr_%"]).groupby("period")["b"].sum()
    summary["beats_hold"] = pd.array(
        [int(beats.get(p, 0)) if s == "long night" else pd.NA for p, s in summary.index], dtype="Int64")
    print(summary.reindex(PERIODS, level=0).round(1).to_string())

    _heading(f"Slippage sensitivity ({LATEST}): net CAGR %, by assumed slippage per side")
    grid = [0.0, 1.0, 2.0, 5.0, 10.0]
    sens = []
    for s in HEADLINE[:4]:
        kind = kind_of[s]
        nights = clean[s]["night_days"] if kind == "index" else None
        for strategy in breakeven.STRATEGIES:
            r_all = breakeven.strategy_returns(legs[s], kind, strategy, night_days=nights)
            r = r_all[significance.period_masks(r_all.index)[LATEST]]
            years = (r.index[-1] - r.index[0]).days / 365.25
            tt = breakeven.TRADE_TYPE[kind][strategy]
            row = {"series": s, "strategy": strategy, "trade_type": tt}
            for slip in grid:
                c = costs.round_trip(tt, slippage_bp_per_side=slip)["total"] / 1e4
                row[f"slip {slip:g}"] = 100 * breakeven.net_cagr(r.to_numpy(), c, years)
            sens.append(row)
    print(pd.DataFrame(sens).set_index(["series", "strategy"]).round(1).to_string())

    print("\nNot in these numbers: income tax (intraday is speculative business income, overnight "
          "trades short-term capital gains), interest on idle cash, margin for intraday shorts and futures, "
          "and the risk that the pre-open auction or a VWAP close order moves the price.")
    print("Wrote:", ", ".join(str(p.relative_to(config.ROOT)) for p in OUT.values()))


if __name__ == "__main__":
    main()
