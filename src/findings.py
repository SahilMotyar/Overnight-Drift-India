"""Phase 7: the headline numbers, read back from the reports the phases wrote.

Every number in reports/findings.md comes from :func:`headline`, so rerunning
the pipeline updates the write-up instead of leaving typed numbers to go
stale. :func:`render` fills ``{{name}}`` placeholders, and refuses a template
that asks for a number :func:`headline` doesn't have, or that leaves one of
its numbers unused, so the two cannot drift apart silently.
"""

from __future__ import annotations

import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config, significance, spotcheck
from .data import cache_path
from .figures import multiple
from .universe import instruments

PLACEHOLDER = re.compile(r"\{\{\s*([a-z0-9_]+)\s*\}\}")
REPORT_FILES = {
    "series": "phase2_series.csv",
    "drops": "phase2_drop_table.csv",
    "spot": "phase2_spotcheck.csv",
    "unusual": "phase2_unusual_vs_nse.csv",
    "summary": "phase3_summary.csv",
    "kinds": "phase3_night_kinds.csv",
    "vs_index": "phase3_vs_index.csv",
    "tests": "phase4_tests.csv",
    "xs": "phase4_cross_section.csv",
    "costs": "phase5_costs.csv",
    "be": "phase5_breakeven.csv",
    "strategies": "phase6_strategies.csv",
    "sip": "phase6_sip.csv",
}
PERIODS = [significance.FULL, *config.SUBPERIODS]
EW = "EW stocks"


def load_reports(reports_dir: Path = config.REPORTS) -> dict[str, pd.DataFrame]:
    missing = [f for f in REPORT_FILES.values() if not (reports_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"run the earlier phases first; missing {missing}")
    return {k: pd.read_csv(reports_dir / f) for k, f in REPORT_FILES.items()}


def render(template: str, values: dict[str, str]) -> str:
    """Fill ``{{name}}`` placeholders; fail on a missing or an unused number."""
    used = set(PLACEHOLDER.findall(template))
    missing = sorted(used - values.keys())
    unused = sorted(values.keys() - used)
    if missing:
        raise KeyError(f"template asks for numbers headline() does not provide: {missing}")
    if unused:
        raise KeyError(f"headline() provides numbers the template never uses: {unused}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


# --------------------------------------------------------------------------
# Formatting: one place, so the write-up is consistent
# --------------------------------------------------------------------------

def bp(x: float) -> str:
    return f"{x:+.1f}".replace("-", "−")


def pct(x: float) -> str:
    return f"{x:+.1f}%".replace("-", "−")


def num(x: float, places: int = 1) -> str:
    return f"{x:.{places}f}".replace("-", "−")


def count(x: float) -> str:
    return f"{int(x):,}"


def _one(df: pd.DataFrame, **match) -> pd.Series:
    """The single row matching every ``column=value``; fail loudly otherwise."""
    m = np.ones(len(df), dtype=bool)
    for col, v in match.items():
        m &= (df[col] == v).to_numpy()
    if m.sum() != 1:
        raise LookupError(f"expected one row for {match}, found {m.sum()}")
    return df[m].iloc[0]


def hidden_sessions(clean_dir: Path = config.DATA_CLEAN, checks: Path = config.NSE_SESSION_CHECKS) -> int:
    """NSE sessions (by its own archive) that no cleaned series has a row for."""
    verdicts = pd.read_csv(checks, parse_dates=["date"])
    sessions = pd.DatetimeIndex(verdicts.loc[verdicts["nse_session"].astype(bool), "date"])
    have = pd.DatetimeIndex([])
    for t in instruments():
        path = cache_path(t, clean_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path}: run scripts.phase2_validate first")
        have = have.union(pd.read_csv(path, usecols=["Date"], parse_dates=["Date"])["Date"])
    return int((~sessions.isin(have)).sum())


# --------------------------------------------------------------------------
# The numbers
# --------------------------------------------------------------------------

def headline(r: dict[str, pd.DataFrame], clean_dir: Path = config.DATA_CLEAN) -> dict[str, str]:
    """Every number reports/findings.md quotes, formatted, keyed by placeholder name."""
    v: dict[str, str] = {}
    kind_of = {t: k for t, (_, k) in instruments().items()}
    stocks = [t for t in kind_of if kind_of[t] == "stock"]

    # Method constants the prose quotes (from config, not typed) ------------
    v["spot_tolerance"] = f"{100 * config.SPOTCHECK_TOLERANCE:.1f}%"
    v["futures_carry"] = f"{100 * config.FUTURES_CARRY_ANNUAL:.0f}%"
    slip = sorted(set(config.SLIPPAGE_BP_PER_SIDE.values()))
    v["slippage_range"] = f"{slip[0]:g}–{slip[-1]:g}" if len(slip) > 1 else f"{slip[0]:g}"
    v["reversed_jump"] = f"{100 * config.REVERSED_OPEN_MIN_JUMP:.0f}%"

    # Data -----------------------------------------------------------------
    series = r["series"]
    v["n_series"] = count(len(series))
    v["n_stocks"] = count(len(stocks))
    v["data_last"] = str(series["last"].max())
    v["stock_first"] = str(series.loc[series["kind"] == "stock", "start"].min())
    v["usable_nights"] = count(series["nights_ok"].sum())
    drops = r["drops"]
    v["placeholder_rows"] = count(drops.loc[drops["reason"].str.startswith("placeholder"), "count"].sum())
    v["bad_dividend_series"] = count((series["start_by"] == "implausible dividend").sum())
    v["hidden_sessions"] = count(hidden_sessions(clean_dir))
    v["nse_verdicts"] = count(len(pd.read_csv(config.NSE_SESSION_CHECKS)))
    spot = r["spot"]
    found = spot[spot["found"].astype(bool)]
    nights = found[found["night_ok"].astype(bool)]
    match = spotcheck._log_gap(nights["night_yahoo"], nights["night_nse"]) < config.SPOTCHECK_TOLERANCE
    v["spot_nights"] = count(len(nights))
    v["spot_match_pct"] = f"{100 * match.mean():.0f}%"
    unusual = r["unusual"]
    rev = unusual[unusual["check"] == "flagged: reversed open"]
    same = int((rev["verdict"] == spotcheck.SAME).sum())
    checked = int(rev["verdict"].isin([spotcheck.SAME, spotcheck.DIFFERENT]).sum())
    v["reversed_confirmed"] = f"{same} of {checked}"

    # Phase 3: the split ---------------------------------------------------
    s = r["summary"].set_index("ticker")

    def leg(prefix: str, ticker: str) -> None:
        row = s.loc[ticker]
        v[f"{prefix}_first_year"] = str(row["first"])[:4]
        v[f"{prefix}_night_x"] = multiple(row["night_growth_x"])
        v[f"{prefix}_day_x"] = multiple(row["day_growth_x"])
        v[f"{prefix}_night_bp"] = bp(row["night_mean_bp"])
        v[f"{prefix}_day_bp"] = bp(row["day_mean_bp"])

    leg("nifty", "^NSEI")
    leg("sensex", "^BSESN")
    st = s.loc[stocks]
    longest = st.sort_values(["night_n", "name"], ascending=[False, True]).index[0]
    leg("longest", longest)
    v["longest_stock_name"] = str(s.at[longest, "name"])
    v["stocks_night_beats_day"] = count((st["night_cagr_%"] > st["day_cagr_%"]).sum())
    v["stocks_day_negative"] = count((st["day_cagr_%"] < 0).sum())
    v["stocks_day_nonnegative"] = ", ".join(t.removesuffix(".NS") for t in st.index[st["day_cagr_%"] >= 0]) or "none"
    v["stock_median_night_bp"] = bp(st["night_mean_bp"].median())
    v["stock_median_day_bp"] = bp(st["day_mean_bp"].median())
    k = r["kinds"]
    for who, ticker in (("nifty", "^NSEI"), ("stocks", "all stocks (pooled)")):
        for kind in ("weeknight", "weekend"):
            v[f"{who}_{kind}_bp"] = bp(_one(k, ticker=ticker, night_kind=kind)["mean_bp"])
    vi = r["vs_index"]
    v["etf_gap_2021_bp"] = bp(_one(vi, comparison="NIFTYBEES - ^NSEI, night", year=2021)["gap_median_bp"])

    # Phase 4: is it real --------------------------------------------------
    t = r["tests"]
    for who, name in (("nifty", "^NSEI"), ("ew", EW)):
        for p in PERIODS:
            row = _one(t, series=name, period=p)
            v[f"{who}_gap_{p}"] = bp(row["gap_mean_bp"])
            v[f"{who}_gap_t_{p}"] = num(row["gap_t"])
    full = _one(t, series="^NSEI", period=significance.FULL)
    v["nifty_gap_ci"] = f"{num(full['gap_ci_lo_bp'])} to {num(full['gap_ci_hi_bp'])}"
    v["nifty_night_t"] = num(full["night_t"])
    v["nifty_day_t"] = num(full["day_t"])
    latest = list(config.SUBPERIODS)[-1]
    bank = _one(t, series="^NSEBANK", period=latest)
    v["bank_gap_latest"] = bp(bank["gap_mean_bp"])
    v["bank_gap_t_latest"] = num(bank["gap_t"])
    for who, name in (("sensex", "^BSESN"), ("nifty", "^NSEI")):
        v[f"{who}_night_2011_2019"] = bp(_one(t, series=name, period="2011_2019")["night_mean_bp"])
    stock_tests = t[(t["kind"] == "stock") & t["gap_t"].notna()]
    n_tests = len(stocks) * len(PERIODS)
    bonf = NormalDist().inv_cdf(1 - 0.05 / (2 * n_tests))
    full_stocks = stock_tests[stock_tests["period"] == significance.FULL]
    v["stock_tests"] = count(n_tests)
    v["bonferroni_t"] = num(bonf, 2)
    v["stocks_gap_t2"] = count((full_stocks["gap_t"] > 2).sum())
    v["stocks_gap_bonf"] = count((full_stocks["gap_t"] > bonf).sum())
    v["stocks_gap_t2_latest"] = count((stock_tests.loc[stock_tests["period"] == latest, "gap_t"] > 2).sum())
    xs = r["xs"]
    eligible = xs[xs["years"] >= config.CROSS_SECTION_MIN_YEARS]
    rho = significance.cross_section(eligible, "traded_value_cr", "gap_mean_bp", config.rng("crosssection"))
    v["liquidity_rho"] = num(rho["rho"], 2)
    v["liquidity_p"] = num(rho["p"], 2)

    # Phase 5: costs -------------------------------------------------------
    c = r["costs"].set_index("trade_type")
    v["rates_checked"] = config.COST_RATES_CHECKED
    for name, tt in (("stock_delivery", "stock_delivery"), ("stock_intraday", "stock_intraday"),
                     ("futures", "index_futures")):
        v[f"cost_{name}"] = num(c.at[tt, "total"])
        v[f"stt_{name}"] = num(c.at[tt, "stt"])
    be = r["be"]
    for p in ("2011_2019", "covid_retail", latest):
        row = _one(be, series="^NSEI", strategy="long night", period=p)
        v[f"nifty_s1_net_{p}"] = pct(row["net_cagr_%"])
        v[f"nifty_hold_{p}"] = pct(row["hold_cagr_%"])
    v[f"nifty_gross_{latest}"] = num(_one(be, series="^NSEI", strategy="long night", period=latest)["gross_mean_bp"])
    for p in (significance.FULL, latest):
        v[f"ew_s1_net_{p}"] = pct(_one(be, series=EW, strategy="long night", period=p)["net_cagr_%"])
        stock_be = be[(be["kind"] == "stock") & (be["strategy"] == "long night") & (be["period"] == p)]
        v[f"stocks_s1_positive_{p}"] = count((stock_be["net_cagr_%"] > 0).sum())
    v["ew_s2_net_full"] = pct(_one(be, series=EW, strategy="short day", period=significance.FULL)["net_cagr_%"])
    sx = _one(be, series="^BSESN", strategy="long night", period="2011_2019")
    v["sensex_s1_net_2011_2019"] = pct(sx["net_cagr_%"])
    v["sensex_hold_2011_2019"] = pct(sx["hold_cagr_%"])
    v["etf_s1_net_full"] = pct(_one(be, series="NIFTYBEES.NS", strategy="long night",
                                    period=significance.FULL)["net_cagr_%"])

    # Phase 6: backtests and the SIP ---------------------------------------
    sg = r["strategies"]

    def strat(series_: str, strategy: str, basis: str, col: str) -> float:
        return float(_one(sg, series=series_, period=significance.FULL, basis=basis, strategy=strategy)[col])

    v["nifty_s0_full"] = pct(strat("^NSEI", "S0 hold", "net", "cagr_%"))
    v["nifty_s1_gross_full"] = pct(strat("^NSEI", "S1 long night", "gross", "cagr_%"))
    v["nifty_s1_net_full"] = pct(strat("^NSEI", "S1 long night", "net", "cagr_%"))
    v["nifty_s0_retvol"] = num(strat("^NSEI", "S0 hold", "net", "ret_per_vol"), 2)
    v["nifty_s1_retvol_gross"] = num(strat("^NSEI", "S1 long night", "gross", "ret_per_vol"), 2)
    v["round_trips_per_year"] = count(round(strat("^NSEI", "S1 long night", "net", "round_trips_per_year")))
    sip = r["sip"].set_index("series")
    nifty = sip.loc["^NSEI"]
    sp = sip.loc[stocks]
    for which, col_bp, col_pct in (("next", "edge_vs_next_open_bp", "close_vs_next_open_%"),
                                   ("open", "edge_vs_open_bp", "close_vs_open_%")):
        v[f"sip_nifty_{which}_bp"] = bp(nifty[col_bp])
        v[f"sip_nifty_{which}_pct"] = f"{nifty[col_pct]:+.2f}%".replace("-", "−")
        v[f"sip_stocks_{which}_bp"] = bp(sp[col_bp].median())
        v[f"sip_stocks_{which}_pct"] = f"{sp[col_pct].median():+.2f}%".replace("-", "−")
        v[f"sip_stocks_{which}_wins"] = count((sp[col_pct] > 0).sum())
    return v
