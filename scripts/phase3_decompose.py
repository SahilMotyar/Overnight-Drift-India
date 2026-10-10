"""Phase 3 checkpoint: split every series into nights and days, chain each, report.

    python -m scripts.phase3_decompose

Reads data/clean/ only (Phase 2 output). Writes:
  reports/phase3_summary.csv       one row per series: both legs, both holds
  reports/phase3_night_kinds.csv   weeknight / weekend / holiday nights per series
  reports/phase3_sensitivity.csv   main vs no reversed opens / no ex-dividend nights /
                                   genuine +-20% nights restored
  reports/phase3_vs_index.csv      ETF and equal-weight stocks vs ^NSEI, night and day, by year
  reports/figures/phase3_*.png     the charts
"""

from __future__ import annotations

import sys

import pandas as pd

from src import config, decompose, figures, summary
from src.universe import instruments

FIG = config.FIGURES
OUT = {
    "summary": config.REPORTS / "phase3_summary.csv",
    "kinds": config.REPORTS / "phase3_night_kinds.csv",
    "sens": config.REPORTS / "phase3_sensitivity.csv",
    "vs_index": config.REPORTS / "phase3_vs_index.csv",
}
IDENTITY_TOLERANCE = 1e-10
SUBTITLE = "Night = close to next open. Day = open to close. Log scale. Only returns Phase 2 accepted."


def _heading(text: str) -> None:
    print(f"\n{'=' * 8} {text} {'=' * max(0, 70 - len(text))}")


def _hold_note(c: pd.DataFrame) -> str:
    return (f"Actual buy-and-hold {figures.multiple(c['hold'].iloc[-1])}; the gap to nights × days is "
            "returns that cannot be split (e.g. nights across sessions Yahoo lacks).")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows defaults to cp1252, which has no ₹ or ×
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", None)

    inst = instruments()
    kind_of = {t: k for t, (_, k) in inst.items()}
    stocks = [t for t in inst if kind_of[t] == "stock"]
    clean = {t: decompose.load_clean(t) for t in inst}
    legs = {t: decompose.leg_returns(c) for t, c in clean.items()}
    curves = {t: decompose.curves(clean[t], legs[t]) for t in inst}

    _heading("Identity: (1 + night)(1 + day) = Close / prev Close")
    errors = {t: decompose.identity_error(c) for t, c in clean.items()}
    worst = max(errors, key=lambda t: errors[t])
    print(f"worst over {len(errors)} series: {errors[worst]:.1e} ({worst})")
    if errors[worst] >= IDENTITY_TOLERANCE:
        sys.exit(f"identity broken beyond {IDENTITY_TOLERANCE}: {errors}")

    # Tables -------------------------------------------------------------
    table = pd.DataFrame(
        [{"ticker": t, "kind": kind_of[t], "name": inst[t][0], **summary.summarize(clean[t], legs[t])} for t in inst]
    ).set_index("ticker")
    table.to_csv(OUT["summary"])

    kinds = pd.concat(
        [summary.night_kinds(clean[t], legs[t]).assign(ticker=t) for t in inst], ignore_index=True)
    pooled = summary.night_kinds(
        pd.concat([clean[t] for t in stocks], ignore_index=True),
        pd.concat([legs[t] for t in stocks], ignore_index=True),
    ).assign(ticker="all stocks (pooled)")
    kinds = pd.concat([kinds, pooled], ignore_index=True)
    kinds[["ticker", *kinds.columns.drop("ticker")]].to_csv(OUT["kinds"], index=False)

    sens = pd.concat(
        [summary.sensitivity(t, clean[t], legs[t]).assign(ticker=t, kind=kind_of[t]) for t in inst],
        ignore_index=True)
    sens.to_csv(OUT["sens"], index=False)

    nifty = legs["^NSEI"]
    ew = pd.DataFrame({
        "night": pd.DataFrame({t: legs[t]["night"] for t in stocks}).mean(axis=1),
        "day": pd.DataFrame({t: legs[t]["day"] for t in stocks}).mean(axis=1),
    })
    vs = pd.concat([
        summary.gap_by_year(legs["NIFTYBEES.NS"][leg], nifty[leg]).assign(comparison=f"NIFTYBEES - ^NSEI, {leg}")
        for leg in ("night", "day")
    ] + [
        summary.gap_by_year(ew[leg], nifty[leg]).assign(comparison=f"equal-weight stocks - ^NSEI, {leg}")
        for leg in ("night", "day")
    ]).reset_index()
    vs.to_csv(OUT["vs_index"], index=False)

    # Charts -------------------------------------------------------------
    figures.growth_chart(
        curves["^NSEI"], FIG / "phase3_nifty.png",
        f"Nifty 50, {curves['^NSEI'].index[0].year}–{curves['^NSEI'].index[-1].year}: "
        "growth of ₹1 by nights and by days",
        SUBTITLE, note=_hold_note(curves["^NSEI"]))
    top = [t for t in inst if kind_of[t] != "stock"]
    figures.small_multiples(
        {t: curves[t] for t in top}, FIG / "phase3_indices.png",
        "Indices and the Nifty ETF: nights vs days",
        "Each panel on its own log scale; the gray line is ×1, where both legs start.", ncols=4)
    figures.small_multiples(
        {t.removesuffix(".NS"): curves[t] for t in sorted(stocks)}, FIG / "phase3_stocks.png",
        "Nifty 50 stocks: nights vs days",
        "Current constituents, each from its Phase 2 start date, on its own log scale; "
        "the gray line is ×1.", ncols=5)
    longest = table.loc[stocks].sort_values(["night_n", "name"], ascending=[False, True]).index[0]
    lc = curves[longest]
    figures.growth_chart(
        lc, FIG / "phase3_longest_stock.png",
        f"{inst[longest][0]}, {lc.index[0].year}–{lc.index[-1].year}: "
        "growth of ₹1 by nights and by days",
        SUBTITLE + f" Longest clean history of the {len(stocks)} stocks.", note=_hold_note(lc))

    # Checkpoint print ---------------------------------------------------
    _heading("Summary: growth of ₹1 in each leg, and per-return size")
    cols = ["kind", "first", "years", "night_growth_x", "night_cagr_%", "night_mean_bp", "night_median_bp",
            "night_up_%", "day_growth_x", "day_cagr_%", "day_mean_bp", "day_median_bp", "day_up_%",
            "split_hold_cagr_%", "hold_cagr_%"]
    shown = table[cols].copy()
    for c in ("night_growth_x", "day_growth_x"):
        shown[c] = shown[c].map(figures.multiple)
    print(shown.round(1).to_string())

    s = table.loc[stocks]
    print(f"\nStocks ({len(stocks)}): night CAGR above day CAGR in {(s['night_cagr_%'] > s['day_cagr_%']).sum()}; "
          f"day CAGR negative in {(s['day_cagr_%'] < 0).sum()}; night CAGR negative in {(s['night_cagr_%'] < 0).sum()}.")
    print(f"Across stocks, median of mean night {s['night_mean_bp'].median():+.1f} bp, "
          f"of mean day {s['day_mean_bp'].median():+.1f} bp.")

    _heading("Nights by what they span (bp)")
    show_k = kinds[kinds["ticker"].isin([*top, "all stocks (pooled)"])]
    print(show_k.set_index(["ticker", "night_kind"]).round(2).to_string())

    _heading("Sensitivity: mean bp per night / per day")
    pivot = sens.pivot_table(index=["ticker"], columns="variant", values=["night_mean_bp", "day_mean_bp"])
    print(pivot.loc[top].round(2).to_string())
    med = sens[sens["kind"] == "stock"].groupby("variant")[["night_mean_bp", "day_mean_bp"]].median()
    print("\nmedian across stocks:")
    print(med.round(2).to_string())
    affected = sorted({t for t, _ in config.GENUINE_EXTREME_NIGHTS})
    print("\nstocks with NSE-confirmed +-20% nights (open decision in PLAN.md):")
    print(sens[sens["ticker"].isin(affected) & sens["variant"].isin(["main", "genuine +-20% restored"])]
          [["ticker", "variant", "nights", "night_mean_bp", "night_cagr_%"]].round(2).to_string(index=False))

    _heading("Versus the index (bp per session, by year)")
    for comp in vs["comparison"].unique():
        print(f"[{comp}]")
        print(vs[vs["comparison"] == comp].set_index("year")
              [["n", "a_mean_bp", "b_mean_bp", "gap_mean_bp", "gap_median_bp"]].round(1).to_string())

    print(f"\nLongest clean stock history: {longest} ({table.at[longest, 'night_n']} nights)")
    print("Wrote:", ", ".join(str(p.relative_to(config.ROOT)) for p in OUT.values()),
          "and reports/figures/phase3_{nifty,indices,stocks,longest_stock}.png")


if __name__ == "__main__":
    main()
