"""DuckDB-backed storage for OHLCV bars and index membership.

Design choices:
- One DuckDB file (data/processed/market.duckdb) holds all tables.
- OHLCV is keyed by (symbol, exchange, bar_date, source). Idempotent upserts.
- Index membership is keyed by (universe, symbol, start_date).
- We use DuckDB's COPY ... FROM / INSERT ... ON CONFLICT for performance.

Why DuckDB rather than Postgres for this stage:
- Embedded, zero-ops, single-file backups
- Reads parquet natively, columnar so analytics are fast
- Handles 10s of millions of rows on a laptop without breaking a sweat
- We can always re-ingest into Postgres later for serving if needed
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from packages.common.config import settings
from packages.common.logging import log

# DDL is idempotent — safe to run on every connection.
_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    symbol         VARCHAR  NOT NULL,
    exchange       VARCHAR  NOT NULL,
    bar_date       DATE     NOT NULL,
    open           DOUBLE   NOT NULL,
    high           DOUBLE   NOT NULL,
    low            DOUBLE   NOT NULL,
    close          DOUBLE   NOT NULL,
    volume         BIGINT   NOT NULL,
    close_unadj    DOUBLE,
    source         VARCHAR  NOT NULL,
    ingested_at    TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, exchange, bar_date, source)
);

CREATE INDEX IF NOT EXISTS ohlcv_symbol_date
    ON ohlcv_daily (symbol, bar_date);

CREATE TABLE IF NOT EXISTS index_membership (
    universe       VARCHAR  NOT NULL,
    symbol         VARCHAR  NOT NULL,
    exchange       VARCHAR  NOT NULL,
    start_date     DATE     NOT NULL,
    end_date       DATE,
    company_name   VARCHAR,
    PRIMARY KEY (universe, symbol, start_date)
);

CREATE INDEX IF NOT EXISTS membership_universe_dates
    ON index_membership (universe, start_date, end_date);
"""


def _connect(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    db_path = str(path) if path else settings.duckdb_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    conn.execute(_DDL)
    return conn


@contextmanager
def get_conn(path: str | Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager for a DuckDB connection. Always use this in app code."""
    conn = _connect(path)
    try:
        yield conn
    finally:
        conn.close()


def upsert_ohlcv(df: pd.DataFrame, conn: duckdb.DuckDBPyConnection | None = None) -> int:
    """Idempotent upsert of OHLCV bars.

    df must have columns matching OHLCV_COLUMNS in packages.common.schemas.
    Returns number of rows written.
    """
    if df.empty:
        return 0

    required = {
        "symbol", "exchange", "bar_date", "open", "high", "low",
        "close", "volume", "source", "ingested_at",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV dataframe missing columns: {missing}")

    own_conn = conn is None
    if own_conn:
        conn = _connect()

    try:
        # Register the dataframe as a temporary view, then INSERT ... ON CONFLICT.
        conn.register("incoming_ohlcv", df)
        result = conn.execute(
            """
            INSERT INTO ohlcv_daily
                (symbol, exchange, bar_date, open, high, low, close, volume,
                 close_unadj, source, ingested_at)
            SELECT
                symbol, exchange, bar_date, open, high, low, close, volume,
                close_unadj, source, ingested_at
            FROM incoming_ohlcv
            ON CONFLICT (symbol, exchange, bar_date, source) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                close_unadj = EXCLUDED.close_unadj,
                ingested_at = EXCLUDED.ingested_at
            """
        )
        rows = len(df)
        log.debug(f"upsert_ohlcv wrote {rows} rows")
        return rows
    finally:
        conn.unregister("incoming_ohlcv")
        if own_conn:
            conn.close()


def upsert_membership(df: pd.DataFrame, conn: duckdb.DuckDBPyConnection | None = None) -> int:
    """Idempotent upsert of index membership rows."""
    if df.empty:
        return 0

    own_conn = conn is None
    if own_conn:
        conn = _connect()

    try:
        conn.register("incoming_membership", df)
        conn.execute(
            """
            INSERT INTO index_membership
                (universe, symbol, exchange, start_date, end_date, company_name)
            SELECT
                universe, symbol, exchange, start_date, end_date, company_name
            FROM incoming_membership
            ON CONFLICT (universe, symbol, start_date) DO UPDATE SET
                end_date = EXCLUDED.end_date,
                company_name = EXCLUDED.company_name,
                exchange = EXCLUDED.exchange
            """
        )
        rows = len(df)
        log.debug(f"upsert_membership wrote {rows} rows")
        return rows
    finally:
        conn.unregister("incoming_membership")
        if own_conn:
            conn.close()


def query_membership_at(
    universe: str,
    as_of: str | pd.Timestamp,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Return the index constituents for `universe` on date `as_of`.

    This is the core point-in-time query. NEVER bypass this when training.
    """
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        sql = """
            SELECT universe, symbol, exchange, start_date, end_date, company_name
            FROM index_membership
            WHERE universe = ?
              AND start_date <= ?
              AND (end_date IS NULL OR end_date >= ?)
            ORDER BY symbol
        """
        return conn.execute(sql, [universe, as_of, as_of]).df()
    finally:
        if own_conn:
            conn.close()


def get_ohlcv(
    symbol: str | list[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    source: str | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars. If multiple sources exist for the same bar, the most
    recently ingested wins (handled at the query level)."""
    symbols = [symbol] if isinstance(symbol, str) else list(symbol)

    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        params: list = [symbols]
        sql = """
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                         PARTITION BY symbol, exchange, bar_date
                         ORDER BY ingested_at DESC
                       ) AS rn
                FROM ohlcv_daily
                WHERE symbol = ANY(?)
        """
        if start is not None:
            sql += " AND bar_date >= ?"
            params.append(start)
        if end is not None:
            sql += " AND bar_date <= ?"
            params.append(end)
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        sql += """
            )
            SELECT symbol, exchange, bar_date, open, high, low, close,
                   volume, close_unadj, source, ingested_at
            FROM ranked WHERE rn = 1
            ORDER BY symbol, bar_date
        """
        return conn.execute(sql, params).df()
    finally:
        if own_conn:
            conn.close()
