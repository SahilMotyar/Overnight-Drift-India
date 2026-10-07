"""Guards on the fixed study choices in ``src.config``."""

from __future__ import annotations

from datetime import date, timedelta

from src import config


def test_subperiods_tile_the_timeline() -> None:
    """Open-ended at both ends, contiguous, no gaps or overlaps between buckets."""
    spans = list(config.SUBPERIODS.values())
    assert spans[0][0] is None and spans[-1][1] is None

    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert date.fromisoformat(next_start) - date.fromisoformat(prev_end) == timedelta(days=1)

    for start, end in spans:
        if start and end:
            assert date.fromisoformat(start) < date.fromisoformat(end)


def test_data_paths_stay_inside_the_project() -> None:
    for p in (config.DATA_RAW, config.DATA_CLEAN, config.DATA_UNIVERSE, config.REPORTS, config.FIGURES):
        assert p.is_relative_to(config.ROOT)
        assert p.is_dir()


def test_no_ticker_listed_twice() -> None:
    tickers = list(config.INDICES) + list(config.ETFS)
    assert len(tickers) == len(set(tickers))
