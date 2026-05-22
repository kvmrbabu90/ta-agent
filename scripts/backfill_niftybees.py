"""Backfill NIFTYBEES ETF history 2014-01-01 → 2020-12-31 via yfinance.

NIFTYBEES is our NIFTY100 strategy's benchmark in the strict walk-forward.
Our DB has it from 2021-01-01 onward (likely picked up by an incidental
backfill), but the WF runs from 2016, so the 2016-2020 benchmark column
shows blank in the Live WF UI. This script fills that gap.

Run while the SP500 / NIFTY100 strict WFs are in flight — uses the same
retry-on-lock helper as the NSE bhavcopy adapter to coexist with their
DuckDB writes.
"""

from __future__ import annotations

import sys
import time
from datetime import date

import pandas as pd
import yfinance as yf

from packages.common.logging import log
from packages.ingestion.adapters.nse_bhavcopy import _upsert_with_retry


def main() -> int:
    start = date(2014, 1, 1)
    end = date(2021, 1, 1)
    log.info(f"backfill NIFTYBEES.NS {start} -> {end} via yfinance")

    raw = yf.download(
        "NIFTYBEES.NS",
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        log.error("yfinance returned no data")
        return 1

    # yfinance multi-index columns when downloading a single ticker:
    # flatten so we have plain OHLCV column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    # yfinance "Close" is the raw close; "Adj Close" is split+dividend-adjusted.
    # Our schema: close=adjusted, close_unadj=raw — matches the convention used
    # by the yfinance equities adapter.
    df = pd.DataFrame({
        "symbol": "NIFTYBEES",
        "exchange": "NSE",
        "bar_date": raw.index.date,
        "open": raw["Open"].astype(float).values,
        "high": raw["High"].astype(float).values,
        "low": raw["Low"].astype(float).values,
        "close": raw["Adj Close"].astype(float).values,
        "close_unadj": raw["Close"].astype(float).values,
        "volume": raw["Volume"].fillna(0).astype("int64").values,
        "source": "yfinance",
        "ingested_at": pd.Timestamp.utcnow(),
    })
    # Drop any all-NaN OHLC rows (yfinance occasionally emits these on holidays).
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    log.info(f"writing {len(df):,} NIFTYBEES rows to DuckDB (retry on lock)")
    _upsert_with_retry(df)
    log.info("NIFTYBEES backfill complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
