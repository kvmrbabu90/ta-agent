"""FastAPI dependency providers.

Per-request connections to keep DuckDB and SQLite away from FastAPI's
async event loop. Both libraries are sync — FastAPI runs sync route
handlers in a threadpool automatically, so we declare routes as ``def``
(not ``async def``) and let the framework handle off-loading.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import duckdb

from packages.inference.db import get_sqlite_conn as _sqlite_ctx
from packages.ingestion.storage import _connect as _duck_connect


def get_duckdb_conn() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh DuckDB connection scoped to one request."""
    conn = _duck_connect()
    try:
        yield conn
    finally:
        conn.close()


def get_sqlite_conn() -> Iterator[sqlite3.Connection]:
    """Yield a fresh SQLite connection scoped to one request."""
    with _sqlite_ctx() as conn:
        yield conn
