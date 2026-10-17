# Overnight drift in Indian stocks

A short video claimed that a stock can make all of its gains overnight (close to the next open) and lose ground during trading hours. This repo tests that on Indian data: the Nifty 50, Sensex, Nifty Bank, the NIFTYBEES ETF and the 50 current Nifty 50 stocks, from 1996 to 2026. It asks whether the split is real, whether it is more than noise, and whether anyone could trade it after Indian costs.

**Short version:** the split is strong and statistically solid in Indian data too. But at today's Indian trading costs, trading it loses money, and what's left is a small, free timing rule for long-term buyers.

**The results, with every number:** [`reports/findings.md`](reports/findings.md).

![Nifty 50: growth of ₹1 by nights and by days](reports/figures/phase3_nifty.png)

## Run it

Python 3.12. Dependencies are pinned with hashes in `requirements.txt` (5 direct packages, 37 in all; see `requirements.in`).

```
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip        # Windows; .venv/bin/python elsewhere
.venv\Scripts\python -m pip install --require-hashes --only-binary :all: -r requirements.txt

python -m scripts.phase1_acquire     # download prices (Yahoo) and the Nifty 50 list (NSE)
python -m scripts.reproduce          # Phases 2-7 from that cache; rewrites reports/
python -m pytest                     # the test suite
```

Phase 1 needs the network, and later runs pull in new trading days, so your numbers will move a little from the committed ones. `reports/findings.md` is regenerated from the reports, so it always matches them. A test fails if it ever goes stale.

Changing a dependency? Follow the supply-chain rules in [`PLAN.md`](PLAN.md#packages-approved-2026-10-06): lock with a 7-day upload cutoff, then run `scripts/audit_deps.py` before installing.

## How it works

Each phase has a script in `scripts/` that ends at a checkpoint: printed diagnostics, plus CSVs and charts in `reports/`.

| Phase | Script | What it does |
|---|---|---|
| 1 | `phase1_acquire` | Freezes NSE's Nifty 50 list; downloads daily prices; date-coverage table |
| 2 | `phase2_validate` | Cleans the data against NSE's own archive: trading calendar, placeholder rows, glitches, corporate actions; spot-checks against NSE's official prices |
| 3 | `phase3_decompose` | Splits every series into nights and days and chains each |
| 4 | `phase4_significance` | Newey-West t-stats and bootstrap CIs by subperiod; a cross-section across stocks |
| 5 | `phase5_costs` | Indian round-trip costs (STT, stamp duty, exchange, GST, DP, slippage) and break-even edges |
| 6 | `phase6_backtest` | Equity curves before and after costs; the SIP timing comparison |
| 7 | `phase7_findings` | Renders `reports/findings.md` from all of the above |

[`PLAN.md`](PLAN.md) has the full method, every decision as it was made, and why.

```
src/            the library: data, validation, calendar, decomposition, statistics, costs, backtests, charts
scripts/        one script per phase, plus reproduce.py and audit_deps.py
tests/          unit tests, plus checks on the real data when it is present
reports/        checkpoint CSVs, charts, findings.md (generated) and findings.template.md
data/universe/  the frozen Nifty 50 list (committed)
data/reference/ NSE's verdict on every disputed or closed trading day (committed)
data/raw/, data/clean/   downloads and cleaned series (not committed; made by Phases 1 and 2)
```

## Data

- **Prices:** daily OHLC from Yahoo Finance, via `yfinance`.
- **Checks:** NSE's public archives. The bhavcopies and index-close files settle which days were trading days and spot-check Yahoo's prices; the Nifty 50 constituent list defines the stocks.
- **Committed:** only small reference files are. Prices are downloaded locally, and you are responsible for each source's terms of use.

## Not advice

This is research on historical data. Nothing here is investment advice, and the costs, taxes and market rules it models will change.
