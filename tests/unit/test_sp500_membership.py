"""Tests for S&P 500 membership parser using a fixture HTML file.

These tests verify the parsing logic without requiring network access.
The fixture file mirrors the structure of the live Wikipedia page.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from packages.ingestion.universe.sp500_history import (
    _parse_changes_table,
    _parse_current_constituents,
    _reconstruct_timeline,
)

FIXTURE = Path(__file__).parent / "fixtures_sp500.html"


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text()


def test_parse_current_constituents(html: str) -> None:
    df = _parse_current_constituents(html)
    assert "symbol" in df.columns
    assert "company_name" in df.columns
    assert len(df) == 6
    assert "AAPL" in df["symbol"].values
    assert "BRK.B" in df["symbol"].values
    aapl = df[df["symbol"] == "AAPL"].iloc[0]
    assert aapl["date_added"] == date(1982, 11, 30)


def test_parse_changes_table(html: str) -> None:
    df = _parse_changes_table(html)
    assert {"change_date", "added_ticker", "removed_ticker"}.issubset(df.columns)
    assert len(df) == 5
    tsla_row = df[df["added_ticker"] == "TSLA"].iloc[0]
    assert tsla_row["change_date"] == date(2020, 12, 21)
    assert tsla_row["removed_ticker"] == "AIV"


def test_reconstruct_timeline(html: str) -> None:
    current = _parse_current_constituents(html)
    changes = _parse_changes_table(html)
    timeline = _reconstruct_timeline(current, changes, today=date(2025, 1, 1))

    # Shape checks
    assert {"universe", "symbol", "start_date", "end_date"}.issubset(timeline.columns)
    assert (timeline["universe"] == "SP500").all()

    # Currently-active rows must equal the count of current constituents (6).
    # Historical-only tickers (FB, KDP, JCP, AIV, AAL, PLTR) must all have a
    # non-null end_date.
    active = timeline[timeline["end_date"].isna()]
    assert len(active) == 6
    assert set(active["symbol"]) == {"AAPL", "MSFT", "NVDA", "META", "TSLA", "BRK.B"}

    # TSLA: added 2020-12-21, currently active
    tsla = timeline[timeline["symbol"] == "TSLA"].iloc[0]
    assert tsla["start_date"] == date(2020, 12, 21)
    assert tsla["end_date"] is None

    # AIV: removed 2020-12-21, never re-added
    aiv = timeline[timeline["symbol"] == "AIV"]
    assert len(aiv) == 1
    assert aiv.iloc[0]["end_date"] == date(2020, 12, 21)
    assert aiv.iloc[0]["start_date"] < date(2020, 12, 21)

    # JCP: removed 2013-12-23
    jcp = timeline[timeline["symbol"] == "JCP"]
    assert len(jcp) == 1
    assert jcp.iloc[0]["end_date"] == date(2013, 12, 23)

    # PLTR: added but no matching remove and not in current. We close at today
    # as a safe fallback (with a warning logged). Verify end_date is set.
    pltr = timeline[timeline["symbol"] == "PLTR"]
    assert len(pltr) == 1
    assert pltr.iloc[0]["start_date"] == date(2024, 9, 23)
    assert pltr.iloc[0]["end_date"] is not None
