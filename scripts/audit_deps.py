"""Audit the dependency lock before (or after) installing it.

Two checks, standard library only, so the audit itself adds no packages:

1. **Upload age.** Every file pinned by hash in ``requirements.txt`` must have
   been uploaded to PyPI before the cutoff (default: 7 days ago). A version's
   wheels can be uploaded on different days, so this checks the exact files
   the lock allows, not just the version's first release date.
2. **Known vulnerabilities.** Every pinned ``name==version`` is looked up in
   the OSV database (https://osv.dev), the same source pip-audit uses.

Exits non-zero if any file is too new, any locked hash is not on PyPI, or any
package has a known vulnerability, so it can gate an install.

    python scripts/audit_deps.py                     # cutoff = now - 7 days
    python scripts/audit_deps.py --cutoff 2026-09-29
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.txt"
MIN_AGE = timedelta(days=7)

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)")
HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def parse_lock(path: Path) -> dict[tuple[str, str], set[str]]:
    """Return ``{(name, version): {sha256, ...}}`` from a hashed lock file."""
    pins: dict[tuple[str, str], set[str]] = {}
    current: tuple[str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PIN_RE.match(line)
        if m:
            current = (m.group(1).lower(), m.group(2))
            pins[current] = set()
        if current is not None:
            pins[current].update(HASH_RE.findall(line))
    unhashed = [f"{n}=={v}" for (n, v), h in pins.items() if not h]
    if unhashed:
        sys.exit(f"lock has pins without hashes: {unhashed}")
    return pins


def _get_json(url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def check_upload_age(pins: dict[tuple[str, str], set[str]], cutoff: datetime) -> list[str]:
    problems: list[str] = []
    print(f"{'package':22s} {'version':14s} newest locked file (UTC)")
    for (name, version), hashes in sorted(pins.items()):
        files = _get_json(f"https://pypi.org/pypi/{name}/{version}/json")["urls"]
        by_hash = {f["digests"]["sha256"]: f for f in files}

        missing = hashes - by_hash.keys()
        if missing:
            problems.append(f"{name}=={version}: {len(missing)} locked hash(es) not on PyPI")

        uploads = [
            datetime.fromisoformat(by_hash[h]["upload_time_iso_8601"].replace("Z", "+00:00"))
            for h in hashes & by_hash.keys()
        ]
        newest = max(uploads) if uploads else None
        flag = ""
        if newest is not None and newest >= cutoff:
            flag = "  <-- TOO NEW"
            problems.append(f"{name}=={version}: file uploaded {newest:%Y-%m-%d}, after cutoff")
        shown = f"{newest:%Y-%m-%d %H:%M}" if newest else "?"
        print(f"{name:22s} {version:14s} {shown}{flag}")
    return problems


def check_osv(pins: dict[tuple[str, str], set[str]]) -> list[str]:
    keys = sorted(pins)
    queries = [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in keys]
    results = _get_json("https://api.osv.dev/v1/querybatch", {"queries": queries})["results"]
    problems = []
    for (name, version), res in zip(keys, results):
        ids = [v["id"] for v in res.get("vulns", [])]
        if ids:
            problems.append(f"{name}=={version}: known vulnerabilities {ids}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument(
        "--cutoff",
        help="ISO date/datetime (UTC); files must be uploaded before it. Default: now - 7 days.",
    )
    args = parser.parse_args()

    if args.cutoff:
        cutoff = datetime.fromisoformat(args.cutoff)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
    else:
        cutoff = datetime.now(timezone.utc) - MIN_AGE

    pins = parse_lock(args.lock)
    print(f"Auditing {len(pins)} pinned packages in {args.lock.name}; cutoff {cutoff:%Y-%m-%d %H:%M} UTC\n")

    problems = check_upload_age(pins, cutoff)
    problems += check_osv(pins)

    print()
    if problems:
        print("FAIL")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK: all {len(pins)} packages predate the cutoff, every locked hash is on PyPI, "
          "no known vulnerabilities in OSV.")


if __name__ == "__main__":
    main()
