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
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_CLEAN, DATA_UNIVERSE, REPORTS, FIGURES):
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

# Prices are downloaded with auto_adjust=False and cached as Yahoo serves them:
# OHLC already split-adjusted (so splits and bonus issues do not show up as
# fake overnight crashes), plus "Adj Close", "Dividends" and "Stock Splits".
# src.data.adjust_for_dividends then rescales OHLC by Adj Close / Close, which
# is what auto_adjust=True does internally. Doing it ourselves keeps the raw
# Close (Phase 2 cross-checks it against NSE) and the ex-dates (Phase 3
# reports the night with and without them, since dividends land in the night).

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
