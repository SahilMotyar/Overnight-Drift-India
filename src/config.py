"""Global configuration: seeds, paths, instruments, and fixed study choices.

Every tunable number in the project lives here. Every stochastic call takes an
explicit generator from :func:`rng`; there are no bare ``np.random.*`` calls
anywhere (``tests/test_no_bare_random.py`` enforces this).

Cost parameters are added in Phase 5, once each rate has been checked against
current NSE / SEBI / broker schedules. They are deliberately absent until then.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

SEED: int = 20261006
"""Master seed.  Never consumed directly; always spawned from."""

_ROOT_SEQ = np.random.SeedSequence(SEED)

# Named streams.  Adding a name here is cheap; reordering this list is NOT,
# because it changes every downstream draw.  Append only.
_STREAMS: tuple[str, ...] = (
    "bootstrap",
    "test",
    "spotcheck",
    "crosssection",
)

_STREAM_SEQS = dict(zip(_STREAMS, _ROOT_SEQ.spawn(len(_STREAMS))))


def rng(stream: str) -> np.random.Generator:
    """Return a fresh, independent, reproducible generator for ``stream``.

    Two calls with the same ``stream`` name return generators with identical
    state, so a stage can be re-run in isolation and reproduce itself.
    Different names are statistically independent by construction.
    """
    if stream not in _STREAM_SEQS:
        raise KeyError(
            f"unknown rng stream {stream!r}; add it to _STREAMS in src/config.py "
            f"(append only). Known: {sorted(_STREAM_SEQS)}"
        )
    seq = _STREAM_SEQS[stream]
    return np.random.default_rng(np.random.SeedSequence(
        entropy=seq.entropy, spawn_key=seq.spawn_key
    ))


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
DATA_UNIVERSE = ROOT / "data" / "universe"  # committed, unlike raw/ and clean/
DATA_REFERENCE = ROOT / "data" / "reference"  # committed: facts fetched once from NSE
DATA_NSE = DATA_RAW / "nse"  # NSE archive files, git-ignored like the rest of raw/
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_CLEAN, DATA_UNIVERSE, DATA_REFERENCE, DATA_NSE, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

INDICES: dict[str, str] = {
    "^NSEI": "Nifty 50",
    "^BSESN": "Sensex",
    "^NSEBANK": "Nifty Bank",
}

# An index cannot be bought; the ETF's own open and close are the prices a
# trader would actually get.
ETFS: dict[str, str] = {
    "NIFTYBEES.NS": "Nippon India ETF Nifty BeES",
}

# Stocks: the Nifty 50 constituents as NSE publishes them, frozen into a
# committed snapshot the first time Phase 1 runs. Frozen so that a semi-annual
# index rebalance mid-study cannot silently change the sample; refresh only on
# purpose (scripts/phase1_acquire.py --refresh-universe).
NIFTY50_SNAPSHOT = DATA_UNIVERSE / "nifty50.csv"

# Official sources, tried in order. Both serve the same file.
NIFTY50_LIST_URLS: tuple[str, ...] = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

# Download each series' full history; Phase 2 decides its first clean day.
YF_PERIOD = "max"

# Pause between Yahoo downloads (cache hits don't wait).
YF_REQUEST_DELAY_SECONDS = 1.0

# Phase 1 coverage compares every series against the union of these indices'
# trading days, a stand-in for the NSE holiday calendar (NSE and BSE almost
# always close on the same days). Sensex extends it back before Nifty's Yahoo
# history starts in 2007. Checking against NSE's own holiday lists is Phase 2.
CALENDAR_TICKERS: tuple[str, ...] = ("^NSEI", "^BSESN")

# Prices are downloaded with auto_adjust=False and cached as Yahoo serves them:
# OHLC already split-adjusted (so splits and bonus issues do not show up as
# fake overnight crashes), plus "Adj Close", "Dividends" and "Stock Splits".
# src.data.adjust_for_dividends then rescales OHLC by Adj Close / Close, which
# is what auto_adjust=True does internally. Doing it ourselves keeps the raw
# Close (Phase 2 cross-checks it against NSE) and the ex-dates (Phase 3
# reports the night with and without them, since dividends land in the night).

# --------------------------------------------------------------------------
# Validation (Phase 2)
# --------------------------------------------------------------------------

# NSE's capital market segment opened on this date. Yahoo has earlier ".NS"
# rows (TMPV back to 1991), which cannot be NSE prices.
NSE_EQUITY_START = "1994-11-03"

# A close-to-close jump beyond this, reversed within GLITCH_MAX_ROWS rows back
# to the old level, is a scale glitch (typically one row Yahoo forgot to
# split-adjust, e.g. TATASTEEL +950% then -90% the next day).
GLITCH_MIN_JUMP = 0.25
GLITCH_MAX_ROWS = 5

# Overnight moves beyond +-20% in these large caps are corporate actions Yahoo
# did not adjust (demergers: LT 2004, BAJAJ 2008, TMPV 2025) or bad data. The
# night is rejected; the rows are kept. Each one is listed for review.
EXTREME_NIGHT = 0.20

# A night and the following day that both move more than this, in opposite
# directions, and nearly cancel: a freak opening print. NSE confirms almost
# all of them (64 of 66 checked), so they are flagged, not rejected.
REVERSED_OPEN_MIN_JUMP = 0.15

# A dividend above this share of the previous close is one Yahoo never scaled
# for later splits. Yahoo's adjustment factor (1 - D / prev Close) then wrecks
# every adjusted price before it: ADANIENT 2002-09-04 (121% -> negative
# prices), 2003-09-04 (88%), 2004-06-28 (52%), EICHERMOT 1997-07-21 (41%).
# The largest real one in the data is Coal India's 9.6% (2014-01-17). Found at
# the Phase 3 checkpoint; a series starts on its last such ex-date.
MAX_DIVIDEND_YIELD = 0.20

# A series starts after its last COVERAGE_WINDOW-session stretch with rows on
# fewer than MIN_COVERAGE of sessions. Catches NESTLEIND's 2003-09 placeholder
# years (-> 2010-01-08) and NIFTYBEES' 2012-13 no-trade spell.
COVERAGE_WINDOW = 250
MIN_COVERAGE = 0.9

# Where Yahoo's history is complete but belongs to another security, coverage
# cannot see it. Each override is a "not before" date, backed by a jump in the
# data on that day; the later of rule and override wins.
START_OVERRIDES: dict[str, tuple[str, str]] = {
    "TCS.NS": ("2004-08-25", "listed on NSE 2004-08-25 (IPO); earlier rows are not TCS"),
    "ULTRACEMCO.NS": ("2004-08-24", "listed 2004-08-24 after the L&T cement demerger; -71% night that day"),
    "BAJAJ-AUTO.NS": ("2008-05-26", "new Bajaj Auto listed 2008-05-26; earlier rows are the pre-demerger company"),
    "BAJAJFINSV.NS": ("2008-05-26", "listed 2008-05-26 in the Bajaj demerger; -91% night that day"),
}

# --------------------------------------------------------------------------
# Trading calendar (Phase 2)
# --------------------------------------------------------------------------

# A date is a candidate session if at least this share of the stocks listed at
# the time have a real row on it, and at least CONSENSUS_MIN_SERIES are listed.
CONSENSUS_SHARE = 0.5
CONSENSUS_MIN_SERIES = 5

# Dates where the stocks are split between these shares, or where consensus
# and the index calendar disagree, are settled by whether NSE published a
# bhavcopy that day. Results are cached in NSE_SESSION_CHECKS (committed).
DISPUTED_SHARE: tuple[float, float] = (0.1, 0.9)
NSE_SESSION_CHECKS = DATA_REFERENCE / "nse_session_checks.csv"
NSE_REQUEST_DELAY_SECONDS = 0.5

# Muhurat trading: the short evening session on Diwali. Its "night" and "day"
# mean something else, so these sessions and the nights around them are kept
# out of the main series. One per year, 1996-2025, each confirmed one of two
# ways: stock volume on Yahoo collapses to 9-40% of its trailing 20-session
# median, or (16 dates Yahoo has no row for at all) NSE published a bhavcopy
# on what is otherwise a holiday (data/reference/nse_session_checks.csv).
MUHURAT_SESSIONS: tuple[str, ...] = (
    "1996-11-10", "1997-10-30", "1998-10-19", "1999-11-07", "2000-10-26",
    "2001-11-14", "2002-11-04", "2003-10-25", "2004-11-12", "2005-11-01",
    "2006-10-21", "2007-11-09", "2008-10-28", "2009-10-17", "2010-11-05",
    "2011-10-26", "2012-11-13", "2013-11-03", "2014-10-23", "2015-11-11",
    "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27", "2020-11-14",
    "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-21",
)

# Spot-check of cleaned Yahoo returns against NSE's official prices, on
# randomly drawn regular sessions. Index files exist from 2012, so about half
# the draws also check ^NSEI and ^NSEBANK. A return matches if its log
# differs from NSE's by less than the tolerance (~2 ticks of 0.05 at Rs 50).
SPOTCHECK_SESSIONS = 40
SPOTCHECK_FROM = "1996-01-01"
SPOTCHECK_TOLERANCE = 0.002

# --------------------------------------------------------------------------
# Decomposition (Phase 3)
# --------------------------------------------------------------------------

# Phase 2 rejects every night beyond +-20%, because a corporate action cannot
# be told from a real move without a corporate-action database. NSE's own
# bhavcopy shows these 7 as genuine price moves, not corporate actions
# (reports/phase2_unusual_vs_nse.csv). They stay rejected in the main results;
# Phase 3 reports what putting them back changes (open decision in PLAN.md).
GENUINE_EXTREME_NIGHTS: tuple[tuple[str, str], ...] = (
    ("EICHERMOT.NS", "1998-10-07"), ("EICHERMOT.NS", "1999-03-19"),
    ("EICHERMOT.NS", "1999-04-01"), ("EICHERMOT.NS", "2001-07-13"),
    ("EICHERMOT.NS", "2001-07-16"), ("ONGC.NS", "2002-05-06"),
    ("NESTLEIND.NS", "2010-04-23"),
)

# A return smaller than this in absolute value counts as flat. Dividend
# adjustment leaves ~1e-7 noise on what is exactly zero in raw prices (an
# official Open equal to the previous close: ~4% of stock nights).
FLAT_TOLERANCE = 1e-6

# --------------------------------------------------------------------------
# Inference (Phase 4)
# --------------------------------------------------------------------------

BOOTSTRAP_REPS = 2000
CI_LEVEL = 0.95

# A period is tested only with at least this many accepted returns (~1 year).
MIN_TEST_SESSIONS = 250

# Cross-section: stocks are compared on a common window after NSE's pre-open
# call auction (late 2010), when every stock's open is set the same way, and
# need at least this many years in it.
CROSS_SECTION_FROM = "2011-01-01"
CROSS_SECTION_MIN_YEARS = 3.0
PERMUTATIONS = 10_000

# --------------------------------------------------------------------------
# Trading costs (Phase 5)
# --------------------------------------------------------------------------

# Statutory and exchange rates as fractions of traded value, checked on
# 2026-10-12 against Zerodha's charges page (live) and NSE's STT circular for
# FY2026-27. They change (futures STT went 0.0125% -> 0.02% -> 0.05% between
# Oct 2024 and Apr 2026), so re-check before relying on a result.
COST_SOURCES: tuple[str, ...] = (
    "https://zerodha.com/charges",
    "https://nsearchives.nseindia.com/content/circulars/FATAX73524.pdf",
)
COST_RATES_CHECKED = "2026-10-12"
GST = 0.18             # on brokerage + exchange charges + SEBI fee
SEBI_FEE = 10 / 1e7    # Rs 10 per crore, both sides
# NSE's IPFT levy (Rs 0.01 per crore) is ~1e-9 of value and left out.

# Per trade type. "delivery" = held overnight (a buy-today-sell-tomorrow
# trade still pays delivery STT on both legs); "intraday" = opened and closed
# the same session. Brokerage is a discount broker's: min(pct x value, cap) per
# order. DP = the depository charge per stock per day it is sold out of demat
# (Rs 15.34 incl. GST); it may not apply to every BTST sale, so this is the
# conservative case.
TRADE_TYPES: dict[str, dict[str, float]] = {
    "stock_delivery": {"stt_buy": 0.001, "stt_sell": 0.001, "stamp_buy": 0.00015,
                       "exchange": 0.0000307, "brokerage_pct": 0.0, "brokerage_cap": 0.0, "dp_per_sell": 15.34},
    "stock_intraday": {"stt_buy": 0.0, "stt_sell": 0.00025, "stamp_buy": 0.00003,
                       "exchange": 0.0000307, "brokerage_pct": 0.0003, "brokerage_cap": 20.0, "dp_per_sell": 0.0},
    "etf_delivery": {"stt_buy": 0.0, "stt_sell": 0.00001, "stamp_buy": 0.00015,
                     "exchange": 0.0000307, "brokerage_pct": 0.0, "brokerage_cap": 0.0, "dp_per_sell": 15.34},
    "etf_intraday": {"stt_buy": 0.0, "stt_sell": 0.00025, "stamp_buy": 0.00003,
                     "exchange": 0.0000307, "brokerage_pct": 0.0003, "brokerage_cap": 20.0, "dp_per_sell": 0.0},
    "index_futures": {"stt_buy": 0.0, "stt_sell": 0.0005, "stamp_buy": 0.00002,
                      "exchange": 0.0000183, "brokerage_pct": 0.0003, "brokerage_cap": 20.0, "dp_per_sell": 0.0},
}

# Rupee value of one order, which sets how much the fixed fees (Rs 20
# brokerage cap, Rs 15.34 DP) weigh. A Nifty futures lot is ~Rs 15 lakh.
ORDER_VALUE_INR: dict[str, float] = {
    "stock_delivery": 100_000, "stock_intraday": 100_000,
    "etf_delivery": 100_000, "etf_intraday": 100_000,
    "index_futures": 1_500_000,
}

# Slippage per side, bp: an assumption, not a quote. The open is the pre-open
# auction price (achievable); the official close is a 30-minute VWAP, which a
# VWAP order only approximates. Phase 5 reports break-even costs, so any other
# slippage can be checked against them.
SLIPPAGE_BP_PER_SIDE: dict[str, float] = {
    "stock_delivery": 2.0, "stock_intraday": 2.0,
    "etf_delivery": 2.0, "etf_intraday": 2.0,
    "index_futures": 1.0,
}

# A futures price decays towards spot at the cost of carry (interest minus
# dividend yield, ~6.3% - 1.3%). A long held overnight gives that up for the
# hours the night lasts; a short held through the day earns it for 6.25 h.
FUTURES_CARRY_ANNUAL = 0.05
SESSION_HOURS = 6.25  # 09:15-15:30

# --------------------------------------------------------------------------
# Subperiods (Phase 4)
# --------------------------------------------------------------------------

# Fixed in PLAN.md before any result was seen. Inclusive ISO dates; None means
# "from the start of the series" / "to the end of the series". Do not move
# these after looking at Phase 3 output.
SUBPERIODS: dict[str, tuple[str | None, str | None]] = {
    "pre_auction": (None, "2010-12-31"),        # before NSE's pre-open call auction
    "2011_2019": ("2011-01-01", "2019-12-31"),
    "covid_retail": ("2020-01-01", "2023-06-30"),  # COVID and the retail surge
    "gift_nifty": ("2023-07-01", None),         # SGX Nifty moved to GIFT City
}
