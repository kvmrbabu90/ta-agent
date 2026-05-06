"""Tests for the storage layer."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from packages.ingestion.storage import (
    get_conn,
    get_ohlcv,
    query_membership_at,
    upsert_membership,
    upsert_ohlcv,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test.duckdb")


def test_upsert_ohlcv_idempotent(tmp_db: str) -> None:
    df = pd.DataFrame([
        {
            "symbol": "AAPL", "exchange": "NASDAQ",
            "bar_date": date(2024, 1, 2),
            "open": 187.0, "high": 188.5, "low": 186.5, "close": 187.4,
            "volume": 50_000_000, "close_unadj": 187.4,
            "source": "yfinance", "ingested_at": datetime(2024, 1, 3, 0, 0),
        },
    ])

    with get_conn(tmp_db) as conn:
        n1 = upsert_ohlcv(df, conn=conn)
        n2 = upsert_ohlcv(df, conn=conn)
        assert n1 == 1 and n2 == 1
        out = get_ohlcv("AAPL", conn=conn)
        assert len(out) == 1
        assert out["close"].iloc[0] == 187.4


def test_upsert_ohlcv_updates_existing(tmp_db: str) -> None:
    base = {
        "symbol": "AAPL", "exchange": "NASDAQ",
        "bar_date": date(2024, 1, 2),
        "open": 187.0, "high": 188.5, "low": 186.5, "close": 187.4,
        "volume": 50_000_000, "close_unadj": 187.4,
        "source": "yfinance", "ingested_at": datetime(2024, 1, 3, 0, 0),
    }
    df1 = pd.DataFrame([base])
    df2 = pd.DataFrame([{**base, "close": 999.0,
                         "ingested_at": datetime(2024, 1, 4, 0, 0)}])

    with get_conn(tmp_db) as conn:
        upsert_ohlcv(df1, conn=conn)
        upsert_ohlcv(df2, conn=conn)
        out = get_ohlcv("AAPL", conn=conn)
        assert len(out) == 1
        assert out["close"].iloc[0] == 999.0


def test_membership_point_in_time(tmp_db: str) -> None:
    df = pd.DataFrame([
        {"universe": "SP500", "symbol": "OLDCO", "exchange": "NYSE",
         "start_date": date(2010, 1, 1), "end_date": date(2018, 6, 1),
         "company_name": "Old Co"},
        {"universe": "SP500", "symbol": "NEWCO", "exchange": "NYSE",
         "start_date": date(2018, 6, 1), "end_date": None,
         "company_name": "New Co"},
    ])

    with get_conn(tmp_db) as conn:
        upsert_membership(df, conn=conn)

        # Before 2018-06-01: only OLDCO
        members_2015 = query_membership_at("SP500", "2015-01-01", conn=conn)
        assert set(members_2015["symbol"]) == {"OLDCO"}

        # On 2018-06-01: both (overlap day — both pass the filter)
        members_2018 = query_membership_at("SP500", "2018-06-01", conn=conn)
        assert set(members_2018["symbol"]) == {"OLDCO", "NEWCO"}

        # After 2018-06-01: only NEWCO
        members_2020 = query_membership_at("SP500", "2020-01-01", conn=conn)
        assert set(members_2020["symbol"]) == {"NEWCO"}
