"""Phase 1 checkpoint: freeze the universe, download every series, print coverage.

    python -m scripts.phase1_acquire                     # reuse today's cache
    python -m scripts.phase1_acquire --refresh           # re-download prices
    python -m scripts.phase1_acquire --refresh-universe  # re-fetch NSE's list

Writes reports/phase1_coverage.csv. Exits non-zero if any ticker failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from src import config
from src.coverage import coverage_table, reference_calendar
from src.data import cache_path, download, is_fresh
from src.universe import freeze_universe, stock_tickers

OUT = config.REPORTS / "phase1_coverage.csv"
KINDS = ["index", "etf", "stock"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    parser.add_argument("--refresh-universe", action="store_true", help="re-fetch NSE's list")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    universe = freeze_universe(refresh=args.refresh_universe)
    instruments: dict[str, tuple[str, str]] = {
        **{t: (name, "index") for t, name in config.INDICES.items()},
        **{t: (name, "etf") for t, name in config.ETFS.items()},
        **{t: (name, "stock") for t, name in stock_tickers(universe).items()},
    }

    frames: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    for ticker in instruments:
        hit_network = args.refresh or not is_fresh(cache_path(ticker))
        try:
            frames[ticker] = download(ticker, refresh=args.refresh)
        except Exception as e:  # one bad ticker must not sink the run; reported below
            failed[ticker] = f"{type(e).__name__}: {e}"
        if hit_network:
            time.sleep(config.YF_REQUEST_DELAY_SECONDS)

    ref = reference_calendar(frames)
    table = coverage_table(frames, ref)
    table.insert(0, "kind", pd.Categorical([instruments[t][1] for t in table.index], KINDS))
    table.insert(1, "name", [instruments[t][0] for t in table.index])
    table = table.sort_values(["kind", "first"])
    table.to_csv(OUT)

    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 32):
        print(f"\nReference calendar: {'+'.join(config.CALENDAR_TICKERS)}, "
              f"{ref[0].date()} .. {ref[-1].date()}, {len(ref)} trading days\n")
        print(table.to_string())

    print(f"\n{len(frames)}/{len(instruments)} series downloaded; table written to "
          f"{OUT.relative_to(config.ROOT)}")
    if failed:
        print("FAILED:")
        for t, err in failed.items():
            print(f"  {t}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
