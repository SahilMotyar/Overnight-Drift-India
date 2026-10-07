"""The universe loader must freeze a valid list and never re-fetch silently."""

from __future__ import annotations

import pytest

from src import universe

HEADER = "Company Name,Industry,Symbol,Series,ISIN Code"


def _csv(n: int = 50, extra_rows: list[str] | None = None) -> str:
    rows = [f"Company {i} Ltd.,Industry,SYM{i},EQ,INE{i:07d}01" for i in range(n)]
    return "\n".join([HEADER, *rows, *(extra_rows or [])]) + "\n"


def _parse(text: str):
    return universe.parse_constituents(text, source="fixture", fetched="2026-10-07T00:00:00Z")


def test_valid_list_maps_symbols_to_yahoo_tickers() -> None:
    text = _csv(48, [
        "Mahindra & Mahindra Ltd.,Automobile,M&M,EQ,INE101A01026",
        "Tata Consultancy Services Ltd.,IT, TCS ,EQ,INE467B01029",
    ])
    df = _parse(text)
    tickers = dict(zip(df["Symbol"], df["Ticker"]))
    assert tickers["M&M"] == "M&M.NS"
    assert tickers["TCS"] == "TCS.NS"  # whitespace stripped
    assert (df["Source"] == "fixture").all()


def test_wrong_row_count_fails() -> None:
    with pytest.raises(ValueError, match="expected 50"):
        _parse(_csv(49))


def test_wrong_columns_fail() -> None:
    with pytest.raises(ValueError, match="unexpected columns"):
        _parse(_csv().replace("ISIN Code", "ISIN"))


def test_duplicate_symbols_fail() -> None:
    with pytest.raises(ValueError, match="duplicate symbols"):
        _parse(_csv(49, ["Copy Ltd.,Industry,SYM0,EQ,INE999999901"]))


def test_existing_snapshot_is_never_refetched(tmp_path, monkeypatch) -> None:
    path = tmp_path / "nifty50.csv"
    _parse(_csv()).to_csv(path, index=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("fetched despite a frozen snapshot")

    monkeypatch.setattr(universe, "fetch_nifty50", boom)
    df = universe.freeze_universe(path=path)
    assert len(df) == 50


def test_missing_snapshot_fails_loudly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="freeze the universe"):
        universe.load_universe(tmp_path / "absent.csv")
