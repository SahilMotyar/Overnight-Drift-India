"""Phase 2 checkpoint: validate every series, write data/clean/, report what was removed.

    python -m scripts.phase2_validate

Reads the Phase 1 cache only; never calls Yahoo. Asks NSE about disputed
trading days and fetches the spot-check files; both are cached, so a re-run
is offline.

Writes:
  data/clean/<ticker>.csv                validated, dividend-adjusted, with return verdicts
  data/reference/nse_session_checks.csv  NSE's verdict on every disputed date (committed)
  reports/phase2_drop_table.csv          ticker x stage x reason -> count
  reports/phase2_series.csv              start date and why, rows, usable nights / days
  reports/phase2_extreme_nights.csv      every rejected +-20% night, for review
  reports/phase2_spotcheck.csv           cleaned Yahoo returns vs NSE on sampled sessions
  reports/phase2_unusual_vs_nse.csv      each rejected +-20% night and flagged reversed open, vs NSE
"""

from __future__ import annotations

import logging
from functools import reduce

import pandas as pd

from src import config, nse, sessions, spotcheck
from src.clean import DropLog, finish, row_stage
from src.data import cache_path, load
from src.universe import instruments, nse_ids

OUT = {
    "drop": config.REPORTS / "phase2_drop_table.csv",
    "series": config.REPORTS / "phase2_series.csv",
    "extreme": config.REPORTS / "phase2_extreme_nights.csv",
    "spot": config.REPORTS / "phase2_spotcheck.csv",
    "judged": config.REPORTS / "phase2_unusual_vs_nse.csv",
}


def _heading(text: str) -> None:
    print(f"\n{'=' * 8} {text} {'=' * max(0, 70 - len(text))}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 60)

    inst = instruments()
    kind_of = {t: k for t, (_, k) in inst.items()}
    raw = {t: load(t, adjusted=False) for t in inst}

    # Pass 1: row checks that need no calendar.
    log = DropLog()
    rows = {t: row_stage(raw[t], t, kind_of[t] == "index", log) for t in inst}

    # Calendar: stock consensus, disputes settled by NSE.
    stocks = {t: rows[t] for t in inst if kind_of[t] == "stock"}
    share = sessions.consensus_share(stocks)
    index_dates = reduce(pd.DatetimeIndex.union, (rows[t].index for t in config.CALENDAR_TICKERS))
    disputes = sessions.disputed_dates(share, index_dates)
    verified = nse.verify_sessions(disputes)
    calendar = sessions.build_calendar(share, verified)
    # Sessions no Yahoo series has are invisible to consensus: put every closed day to NSE.
    closed = sessions.closed_days_to_check(calendar)
    hidden = nse.verify_sessions(closed)
    calendar = sessions.build_calendar(share, pd.concat([verified, hidden]))
    kinds = sessions.session_kinds(calendar)

    # Pass 2: calendar-dependent checks and return verdicts. Coverage (for start
    # dates) is measured only against sessions at least one Yahoo series has.
    observed = calendar[calendar.isin(reduce(pd.DatetimeIndex.union, (r.index for r in rows.values())))]
    clean, summaries = {}, []
    for t in inst:
        clean[t], summary = finish(rows[t], t, kind_of[t] == "index", calendar, kinds, log,
                                   coverage_calendar=observed)
        clean[t].to_csv(cache_path(t, config.DATA_CLEAN))
        summaries.append({"kind": kind_of[t], **summary})

    drop = log.table()
    drop.to_csv(OUT["drop"], index=False)
    series = pd.DataFrame(summaries).set_index("ticker")
    series.to_csv(OUT["series"])

    _heading("Calendar")
    print(f"{len(calendar)} sessions, {calendar[0].date()} .. {calendar[-1].date()}; "
          f"special: {kinds[kinds != 'regular'].value_counts().to_dict()}")
    print(f"{len(disputes)} disputed dates settled by NSE: {int(verified.sum())} sessions, "
          f"{int((~verified).sum())} not. Not sessions: {[str(d.date()) for d in verified.index[~verified]]}")
    print(f"{len(closed)} closed days put to NSE; it traded on {int(hidden.sum())}: "
          f"{[f'{d.date()} {d.day_name()[:3]}' for d in hidden.index[hidden]]}")
    missing_everywhere = [str(d.date()) for d in calendar if not any(d in rows[t].index for t in inst)]
    print(f"Sessions no Yahoo series has (nights across them are 'gap'): {missing_everywhere or 'none'}")

    _heading("Drop table: rows removed (all series)")
    by_reason = drop.groupby(["stage", "reason"], sort=False).agg(
        count=("count", "sum"), series=("ticker", "nunique"))
    print(by_reason.loc["row"].sort_values("count", ascending=False).to_string())
    _heading("Drop table: returns rejected (all series)")
    for stage in ("night", "day"):
        if stage in by_reason.index.get_level_values(0):
            print(f"[{stage}]")
            print(by_reason.loc[stage].sort_values("count", ascending=False).to_string())
    print(f"\nPer-ticker detail: {OUT['drop'].relative_to(config.ROOT)}")

    _heading("Series: start date and usable returns")
    total_raw = {t: len(raw[t]) for t in inst}
    print(series.assign(raw_rows=pd.Series(total_raw))[
        ["kind", "start", "start_by", "raw_rows", "rows", "nights_ok", "days_ok", "reversed_opens"]].to_string())

    _heading("Rejected nights beyond +-20% (review: corporate action or bad data?)")
    extreme_rows = []
    for t, cf in clean.items():
        hit = cf["night_note"] == "night beyond +-20%"
        night = (cf["Open"] / cf["Close"].shift(1) - 1)[hit]
        extreme_rows += [{"ticker": t, "date": d.date(), "night": round(v, 4)} for d, v in night.items()]
    extreme = pd.DataFrame(extreme_rows, columns=["ticker", "date", "night"])
    extreme.to_csv(OUT["extreme"], index=False)
    print(extreme.to_string(index=False) if len(extreme) else "none")

    _heading("Short sessions by volume (cross-check of MUHURAT_SESSIONS)")
    ratio = sessions.volume_ratio(stocks, calendar)
    low = ratio[ratio < 0.3]
    print(pd.DataFrame({
        "weekday": low.index.day_name().str[:3],
        "volume_ratio": low.round(2),
        "session_kind": kinds.reindex(low.index),
    }).to_string())

    _heading("Spot-check against NSE official prices")
    days = spotcheck.sample_sessions(calendar, kinds, config.SPOTCHECK_SESSIONS,
                                     config.SPOTCHECK_FROM, config.rng("spotcheck"))
    ids = nse_ids()
    recs = [r for d in days for r in spotcheck.compare_session(d, raw, clean, ids)]
    spot = pd.DataFrame(recs)
    spot.to_csv(OUT["spot"], index=False)
    print(f"{len(days)} sessions, {days[0].date()} .. {days[-1].date()}")
    print(spotcheck.summarize(spot, kind_of).to_string())

    found = spot[spot["found"].astype(bool)]
    tol = config.SPOTCHECK_TOLERANCE
    bad_n = found[found["night_ok"].astype(bool)
                  & (spotcheck._log_gap(found["night_yahoo"], found["night_nse"]) >= tol)]
    bad_d = found[found["day_ok"].astype(bool)
                  & (spotcheck._log_gap(found["day_yahoo"], found["day_nse"]) >= tol)]
    cols = ["date", "ticker", "night_yahoo", "night_nse", "day_yahoo", "day_nse"]
    print(f"\nMismatched accepted nights: {len(bad_n)}, days: {len(bad_d)}")
    if len(bad_n) or len(bad_d):
        print(pd.concat([bad_n, bad_d])[cols].drop_duplicates().round(4).to_string(index=False))
    print("\nNot found at NSE (symbol/ISIN changed):",
          sorted(set(spot.loc[~spot["found"].astype(bool), "ticker"])) or "none")

    _heading("Unusual nights checked against NSE")
    judged = pd.concat([
        spotcheck.check_against_nse(raw, clean, ids, lambda f: f["night_note"] == "night beyond +-20%",
                                    "rejected: night beyond +-20%"),
        spotcheck.check_against_nse(raw, clean, ids, lambda f: f["reversed_open"].astype(bool),
                                    "flagged: reversed open"),
    ], ignore_index=True)
    judged.to_csv(OUT["judged"], index=False)
    print(judged.groupby(["check", "verdict"]).size().unstack(fill_value=0).to_string())
    extreme_judged = judged[judged["check"].str.startswith("rejected")]
    cols = ["ticker", "date", "verdict", "night_yahoo", "night_nse"]
    print("\nEach rejected +-20% night:")
    print(extreme_judged.reindex(columns=cols).round(4).to_string(index=False))

    print("\nWrote:", ", ".join(str(p.relative_to(config.ROOT)) for p in OUT.values()),
          "and data/clean/*.csv")


if __name__ == "__main__":
    main()
