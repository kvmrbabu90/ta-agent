"""Canonical schemas for market data passing between modules.

Every adapter must produce data conforming to OHLCVBar. Every consumer reads
this schema. This is the single source of truth for the pipeline.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Exchange(str, Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    NSE = "NSE"
    BSE = "BSE"


class Universe(str, Enum):
    SP500 = "SP500"
    NIFTY100 = "NIFTY100"


class OHLCVBar(BaseModel):
    """A single daily OHLCV bar, fully adjusted for splits & dividends."""

    symbol: str
    exchange: Exchange
    bar_date: date  # Trading date (calendar date in exchange's local time)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    # Unadjusted close kept for sanity-checking and display purposes.
    close_unadj: float | None = None
    source: str  # "ib" | "kite" | "yfinance"
    ingested_at: datetime


class IndexMembership(BaseModel):
    """A row in the point-in-time index membership table."""

    universe: Universe
    symbol: str
    exchange: Exchange
    start_date: date  # Inclusive
    end_date: date | None = None  # None means currently a member
    company_name: str | None = None


# Canonical column order for OHLCV parquet/duckdb tables.
OHLCV_COLUMNS = [
    "symbol",
    "exchange",
    "bar_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_unadj",
    "source",
    "ingested_at",
]
