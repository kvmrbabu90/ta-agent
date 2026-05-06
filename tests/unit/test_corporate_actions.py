"""Unit tests for the corporate-actions cross-checker."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from packages.common.config import settings
from packages.ingestion.corporate_actions import audit_universe, compare_sources
from packages.ingestion.storage import get_conn, upsert_membership, upsert_ohlcv


def _bar(symbol: str, d: date, close: float, source: str, *, close_unadj: float | None = None) -> dict:
    return {
        "symbol": symbol,
        "exchange": "NASDAQ",
        "bar_date": d,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1_000_000,
        "close_unadj": close_unadj,
        "source": source,
        "ingested_at": datetime(2024, 1, 5, 0, 0),
    }


def _seed(path: str, rows: list[dict]) -> None:
    with get_conn(path) as conn:
        upsert_ohlcv(pd.DataFrame(rows), conn=conn)


def test_compare_sources_flags_50pct_drop_as_split(tmp_path: Path) -> None:
    db = str(tmp_path / "test.duckdb")
    _seed(
        db,
        [
            _bar("AAPL", date(2024, 1, 2), 200.0, "ib", close_unadj=200.0),
            _bar("AAPL", date(2024, 1, 2), 100.0, "yfinance", close_unadj=100.0),
            _bar("AAPL", date(2024, 1, 3), 201.0, "ib", close_unadj=201.0),
            _bar("AAPL", date(2024, 1, 3), 200.5, "yfinance", close_unadj=200.5),
        ],
    )
    df = compare_sources(
        "AAPL", date(2024, 1, 1), date(2024, 1, 5), tolerance_pct=1.0, duckdb_path=db
    )
    assert len(df) == 1  # Only the 50% drop is over tolerance
    row = df.iloc[0]
    assert pd.Timestamp(row["bar_date"]).date() == date(2024, 1, 2)
    assert row["pct_diff"] > 30.0
    assert row["suspected_cause"] == "split"


def test_compare_sources_within_tolerance_returns_empty(tmp_path: Path) -> None:
    db = str(tmp_path / "test.duckdb")
    _seed(
        db,
        [
            _bar("AAPL", date(2024, 1, 2), 100.0, "ib"),
            _bar("AAPL", date(2024, 1, 2), 100.5, "yfinance"),  # 0.5% diff < 1.0
        ],
    )
    df = compare_sources(
        "AAPL", date(2024, 1, 1), date(2024, 1, 5), tolerance_pct=1.0, duckdb_path=db
    )
    assert df.empty


def test_compare_sources_handles_single_source(tmp_path: Path) -> None:
    db = str(tmp_path / "test.duckdb")
    _seed(
        db,
        [
            _bar("AAPL", date(2024, 1, 2), 100.0, "ib"),
            _bar("AAPL", date(2024, 1, 3), 101.0, "ib"),
        ],
    )
    df = compare_sources(
        "AAPL", date(2024, 1, 1), date(2024, 1, 5), tolerance_pct=1.0, duckdb_path=db
    )
    assert df.empty


def test_compare_sources_dividend_heuristic(tmp_path: Path) -> None:
    """A small disagreement plus differing close_unadj/close ratios → 'dividend'."""
    db = str(tmp_path / "test.duckdb")
    _seed(
        db,
        [
            # ib treats prices as div-adjusted: close=100, unadj=102 (ratio 1.02)
            _bar("AAPL", date(2024, 1, 2), 100.0, "ib", close_unadj=102.0),
            # yfinance leaves close == unadj (ratio 1.0); the 2% disagreement is dividend
            _bar("AAPL", date(2024, 1, 2), 102.0, "yfinance", close_unadj=102.0),
        ],
    )
    df = compare_sources(
        "AAPL", date(2024, 1, 1), date(2024, 1, 5), tolerance_pct=1.0, duckdb_path=db
    )
    assert len(df) == 1
    assert df.iloc[0]["suspected_cause"] == "dividend"


def test_audit_universe_summary_sort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)

    today = date.today()
    members = pd.DataFrame(
        [
            {"universe": "SP500", "symbol": "AAPL", "exchange": "NASDAQ",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Apple"},
            {"universe": "SP500", "symbol": "QUIET", "exchange": "NASDAQ",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Quiet Co"},
        ]
    )
    with get_conn(db) as conn:
        upsert_membership(members, conn=conn)

    _seed(
        db,
        [
            _bar("AAPL", today, 100.0, "ib"),
            _bar("AAPL", today, 200.0, "yfinance"),  # huge disagreement
            _bar("QUIET", today, 50.0, "ib"),
            _bar("QUIET", today, 50.1, "yfinance"),  # within tolerance
        ],
    )
    df = audit_universe("SP500", lookback_days=30, tolerance_pct=1.0, duckdb_path=db)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "AAPL"
