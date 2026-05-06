"""Macro-level series ingestion: VIX, USD/INR, and friends.

Pulled from yfinance, stored in a separate ``macro_daily`` table. The macro
feature group reads from this table and broadcasts values across the (symbol,
bar_date) panel.

Why a separate table from ohlcv_daily:
    Macro series are not symbols in the usual sense. ``^VIX`` is an index,
    not a tradable. Mixing them into ohlcv_daily would pollute symbol-level
    queries and confuse the membership join.

Series we currently ingest:
    'vix'      from ^VIX  — CBOE Volatility Index
    'usd_inr'  from INR=X — USD/INR FX rate (for the NIFTY universe)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta

import duckdb
import pandas as pd
import yfinance as yf

from packages.common.logging import log
from packages.ingestion.storage import _connect

# --- DDL ---------------------------------------------------------------------

_MACRO_DDL = """
CREATE TABLE IF NOT EXISTS macro_daily (
    macro_date    DATE       NOT NULL,
    series        VARCHAR    NOT NULL,
    value         DOUBLE     NOT NULL,
    source        VARCHAR    NOT NULL,
    ingested_at   TIMESTAMP  NOT NULL,
    PRIMARY KEY (macro_date, series)
);
CREATE INDEX IF NOT EXISTS macro_series_date ON macro_daily (series, macro_date);
"""

_SOURCE = "yfinance"

# Series name -> yfinance ticker
SERIES_TICKERS: dict[str, str] = {
    "vix": "^VIX",
    "usd_inr": "INR=X",
}


def _ensure_ddl(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(_MACRO_DDL)


@contextmanager
def _macro_conn(path: str | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = _connect(path)
    try:
        _ensure_ddl(conn)
        yield conn
    finally:
        conn.close()


# --- Fetch + store -----------------------------------------------------------


def _download(ticker: str, start: date, end: date) -> pd.DataFrame:
    """yfinance download with our standard kwargs. End is exclusive in yf."""
    return yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
        actions=False,
    )


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_macro_daily(start: date, end: date) -> pd.DataFrame:
    """Fetch all macro series over [start, end] and return a long-format DataFrame.

    Columns: macro_date, series, value, source, ingested_at.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    rows: list[pd.DataFrame] = []
    now_utc = datetime.now(UTC)
    for series, ticker in SERIES_TICKERS.items():
        try:
            raw = _download(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 — log and continue with other series
            log.warning(f"macro download failed for {series} ({ticker}): {exc!r}")
            continue
        if raw is None or raw.empty:
            log.warning(f"macro: empty result for {series} ({ticker})")
            continue
        flat = _flatten(raw)
        if "Close" not in flat.columns:
            log.warning(f"macro: no Close column in {series} response: {flat.columns}")
            continue
        # yfinance index is DatetimeIndex; coerce to date.
        rows.append(
            pd.DataFrame(
                {
                    "macro_date": [pd.Timestamp(d).date() for d in flat.index],
                    "series": series,
                    "value": flat["Close"].astype(float).values,
                    "source": _SOURCE,
                    "ingested_at": now_utc,
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["macro_date", "series", "value", "source", "ingested_at"]
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def upsert_macro(
    df: pd.DataFrame, conn: duckdb.DuckDBPyConnection | None = None
) -> int:
    """Idempotent upsert into macro_daily."""
    if df.empty:
        return 0
    own = conn is None
    if own:
        conn = _connect()
    try:
        _ensure_ddl(conn)
        conn.register("incoming_macro", df)
        conn.execute(
            """
            INSERT INTO macro_daily (macro_date, series, value, source, ingested_at)
            SELECT macro_date, series, value, source, ingested_at FROM incoming_macro
            ON CONFLICT (macro_date, series) DO UPDATE SET
                value = EXCLUDED.value,
                source = EXCLUDED.source,
                ingested_at = EXCLUDED.ingested_at
            """
        )
        conn.unregister("incoming_macro")
        return len(df)
    finally:
        if own:
            conn.close()


def backfill_macro(
    start: date, end: date, *, duckdb_path: str | None = None
) -> dict:
    df = fetch_macro_daily(start, end)
    with _macro_conn(duckdb_path) as conn:
        n = upsert_macro(df, conn=conn)
    log.info(
        f"macro backfill: rows_written={n} series={list(df['series'].unique()) if not df.empty else []}"
    )
    return {
        "requested": len(SERIES_TICKERS),
        "rows_written": n,
        "series": sorted(df["series"].unique().tolist()) if not df.empty else [],
    }


def load_macro_series(
    series: str, start: date, end: date, *, duckdb_path: str | None = None
) -> pd.Series:
    """Read a single macro series from DuckDB as a Series indexed by date."""
    with _macro_conn(duckdb_path) as conn:
        df = conn.execute(
            "SELECT macro_date, value FROM macro_daily WHERE series = ? AND macro_date BETWEEN ? AND ? ORDER BY macro_date",
            [series, start, end],
        ).df()
    if df.empty:
        return pd.Series(dtype=float, name=series)
    return pd.Series(
        data=df["value"].astype(float).values,
        index=pd.to_datetime(df["macro_date"]),
        name=series,
    )


def has_macro_data(*, duckdb_path: str | None = None) -> bool:
    """Cheap check used by the pipeline to decide whether to register the
    macro feature group."""
    with _macro_conn(duckdb_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM macro_daily").fetchone()
    return bool(row and row[0] > 0)
