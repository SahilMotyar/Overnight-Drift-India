"""The reproduce script names every phase script, in order, and nothing else."""

from __future__ import annotations

import re

from scripts import reproduce
from src.config import ROOT


def test_every_phase_script_after_phase_1_is_rerun_in_order() -> None:
    on_disk = sorted(p.stem for p in (ROOT / "scripts").glob("phase*_*.py") if not p.stem.startswith("phase1_"))
    named = [m.removeprefix("scripts.") for m in reproduce.PHASES]
    assert named == on_disk
    numbers = [int(re.match(r"phase(\d+)_", m).group(1)) for m in named]
    assert numbers == sorted(numbers) == list(range(2, 8))
