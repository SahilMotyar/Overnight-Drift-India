"""Phase 7: write reports/findings.md from the reports, and draw its summary chart.

    python -m scripts.phase7_findings

Reads reports/*.csv (Phases 2-6) and data/clean/ (Phase 2). Writes:
  reports/findings.md                     rendered from reports/findings.template.md
  reports/figures/phase7_edge_vs_cost.png gross edge per round trip vs today's cost
"""

from __future__ import annotations

import sys

import pandas as pd

from src import config, figures, findings

TEMPLATE = config.REPORTS / "findings.template.md"
OUT = config.REPORTS / "findings.md"
FIG = config.FIGURES / "phase7_edge_vs_cost.png"
PERIOD_NAMES = {"pre_auction": "before 2011", "2011_2019": "2011–2019",
                "covid_retail": "2020–Jun 2023", "gift_nifty": "Jul 2023 on"}
PANELS = {
    ("^NSEI", "long night"): ("Nifty, long night (futures)", "Nifty's overnight trade"),
    ("^NSEI", "short day"): ("Nifty, short day (futures)", "Nifty's intraday short"),
    ("EW stocks", "long night"): ("EW stocks, long night (delivery)", "the stocks' overnight trade"),
    ("EW stocks", "short day"): ("EW stocks, short day (intraday)", "the stocks' intraday short"),
}


def edge_rows(breakeven: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (series, strategy), (panel, _) in PANELS.items():
        for period, name in PERIOD_NAMES.items():
            r = findings._one(breakeven, series=series, strategy=strategy, period=period)
            rows.append({"panel": panel, "who": PANELS[(series, strategy)][1], "period": name,
                         "edge_bp": r["gross_mean_bp"], "cost_bp": r["cost_bp"]})
    return pd.DataFrame(rows)


def chart_title(rows: pd.DataFrame) -> str:
    """State what the chart shows, from the data, so a rerun cannot leave it wrong."""
    cleared = rows[rows["edge_bp"] > rows["cost_bp"]]
    if cleared.empty:
        return "No trade's edge has ever cleared today's cost"
    parts = [f"{who} in {' and '.join(g['period'])}" for who, g in cleared.groupby("who", sort=False)]
    return f"Only {'; '.join(parts)} cleared today's cost"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    reports = findings.load_reports()
    rows = edge_rows(reports["be"])
    title = chart_title(rows)
    figures.edge_vs_cost(
        rows, FIG, title[0].upper() + title[1:],
        "Mean gross return per round trip (bp) by period, against today's round-trip cost. "
        "Futures edges are after carry. Every number: reports/phase5_breakeven.csv.")

    values = findings.headline(reports)
    text = findings.render(TEMPLATE.read_text(encoding="utf-8"), values)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(values)} numbers filled in; wrote {OUT.relative_to(config.ROOT)} and {FIG.relative_to(config.ROOT)}")
    print(f"chart title: {title}")


if __name__ == "__main__":
    main()
