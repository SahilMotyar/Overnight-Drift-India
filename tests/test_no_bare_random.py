"""Enforce the seeding discipline.

Every stochastic call takes an explicit generator from ``src.config.rng``.
Bare ``np.random.*`` draws from a hidden global state and silently destroys
reproducibility, so we fail the build on them rather than trusting review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.config import ROOT, rng

# Constructing a generator is fine; *drawing* from the global state is not.
ALLOWED_NP_RANDOM = {"default_rng", "SeedSequence", "Generator", "PCG64", "Philox"}


def _python_files() -> list[Path]:
    return [p for d in ("src", "scripts", "tests") for p in sorted((ROOT / d).rglob("*.py"))]


def _attr_chain(node: ast.AST) -> list[str] | None:
    """Flatten ``np.random.normal`` into ``['np', 'random', 'normal']``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_bare_numpy_random(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offences: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _attr_chain(node)
            if (
                chain
                and len(chain) >= 3
                and chain[0] in {"np", "numpy"}
                and chain[1] == "random"
                and chain[2] not in ALLOWED_NP_RANDOM
            ):
                offences.append(f"line {node.lineno}: {'.'.join(chain)}")

        # `from numpy.random import normal` smuggles the same global state in.
        if isinstance(node, ast.ImportFrom) and node.module == "numpy.random":
            bad = [a.name for a in node.names if a.name not in ALLOWED_NP_RANDOM]
            if bad:
                offences.append(f"line {node.lineno}: from numpy.random import {bad}")

    assert not offences, (
        f"{path.relative_to(ROOT)} draws from numpy's global RNG state:\n  "
        + "\n  ".join(offences)
        + "\nUse src.config.rng('<stream>') instead."
    )


def test_rng_is_reproducible() -> None:
    """Same stream name, same draws: a stage can be re-run in isolation."""
    a = rng("test").normal(size=64)
    b = rng("test").normal(size=64)
    assert (a == b).all()


def test_rng_streams_are_independent() -> None:
    """Different stream names must not collide."""
    a = rng("bootstrap").normal(size=256)
    b = rng("test").normal(size=256)
    assert not (a == b).any()


def test_unknown_stream_fails_loudly() -> None:
    with pytest.raises(KeyError, match="unknown rng stream"):
        rng("stream_that_does_not_exist")
