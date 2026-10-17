"""The findings numbers: strict rendering, and values that match the reports they come from."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, findings
from src.data import cache_path


def test_render_fills_placeholders() -> None:
    assert findings.render("Nifty {{ a }} and {{b}}.", {"a": "x55.5", "b": "+8.9"}) == "Nifty x55.5 and +8.9."


def test_render_refuses_a_missing_number() -> None:
    with pytest.raises(KeyError, match="does not provide.*'c'"):
        findings.render("{{a}} {{c}}", {"a": "1"})


def test_render_refuses_an_unused_number() -> None:
    with pytest.raises(KeyError, match="never uses.*'b'"):
        findings.render("{{a}}", {"a": "1", "b": "2"})


def test_formats_use_a_real_minus_sign() -> None:
    assert findings.bp(-4.9) == "−4.9" and findings.bp(8.92) == "+8.9"
    assert findings.pct(-2.08) == "−2.1%" and findings.count(300282) == "300,282"


def test_one_row_lookup_fails_loudly() -> None:
    df = pd.DataFrame({"a": [1, 1, 2]})
    assert findings._one(df, a=2)["a"] == 2
    with pytest.raises(LookupError, match="found 2"):
        findings._one(df, a=1)


@pytest.fixture(scope="module")
def numbers() -> dict[str, str]:
    if not all((config.REPORTS / f).exists() for f in findings.REPORT_FILES.values()):
        pytest.skip("reports/ incomplete; run the phase scripts")
    if not cache_path("^NSEI", config.DATA_CLEAN).exists():
        pytest.skip("no data/clean/ yet; run scripts.phase2_validate")
    return findings.headline(findings.load_reports())


def test_headline_numbers_match_their_reports(numbers) -> None:
    s = pd.read_csv(config.REPORTS / "phase3_summary.csv", index_col=0)
    assert numbers["nifty_night_x"] == findings.multiple(s.at["^NSEI", "night_growth_x"])
    t = pd.read_csv(config.REPORTS / "phase4_tests.csv")
    row = t[(t["series"] == "^NSEI") & (t["period"] == "full")].iloc[0]
    assert numbers["nifty_gap_full"] == findings.bp(row["gap_mean_bp"])
    assert numbers["n_stocks"] == "50" and numbers["stock_tests"] == "250"


def test_every_number_is_a_non_empty_string(numbers) -> None:
    assert numbers and all(isinstance(x, str) and x for x in numbers.values())


TEMPLATE = config.REPORTS / "findings.template.md"


def test_the_template_uses_exactly_the_numbers_headline_provides(numbers) -> None:
    findings.render(TEMPLATE.read_text(encoding="utf-8"), numbers)  # raises on any mismatch


def test_committed_findings_are_up_to_date(numbers) -> None:
    """Rerunning a phase without rerunning Phase 7 leaves findings.md stale: catch that."""
    out = config.REPORTS / "findings.md"
    if not out.exists():
        pytest.skip("findings.md not generated yet")
    expected = findings.render(TEMPLATE.read_text(encoding="utf-8"), numbers)
    assert out.read_text(encoding="utf-8") == expected, "run `python -m scripts.phase7_findings`"
