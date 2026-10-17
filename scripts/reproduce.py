"""Rerun Phases 2-7 in order from the Phase 1 cache, stopping at the first failure.

    python -m scripts.reproduce

Phase 1 (the Yahoo download) is left out on purpose: it hits the network and
pulls in new days, which changes every number. Run `python -m
scripts.phase1_acquire` first on a fresh clone. Phase 2 also fetches NSE's
archive files the first time (its verdicts on trading days are committed in
data/reference/; the spot-check files are cached under data/raw/nse/).
"""

from __future__ import annotations

import subprocess
import sys
import time

from src import config

PHASES: tuple[str, ...] = (
    "scripts.phase2_validate",
    "scripts.phase3_decompose",
    "scripts.phase4_significance",
    "scripts.phase5_costs",
    "scripts.phase6_backtest",
    "scripts.phase7_findings",
)


def main() -> None:
    for module in PHASES:
        print(f"==> {module}", flush=True)
        t0 = time.time()
        done = subprocess.run([sys.executable, "-m", module], cwd=config.ROOT,
                              stdout=subprocess.DEVNULL)
        if done.returncode != 0:
            sys.exit(f"{module} failed (exit {done.returncode}); rerun it alone to see why")
        print(f"    ok, {time.time() - t0:.0f}s", flush=True)
    print("All phases rerun; reports/findings.md is up to date.")


if __name__ == "__main__":
    main()
