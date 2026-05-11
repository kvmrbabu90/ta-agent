"""FastAPI dependency providers.

Per-request connections to keep DuckDB and SQLite away from FastAPI's
async event loop. Both libraries are sync — FastAPI runs sync route
handlers in a threadpool automatically, so we declare routes as ``def``
(not ``async def``) and let the framework handle off-loading.

DuckDB connections in the API are opened READ-ONLY so the API process
can coexist with the writer (the daily-pipeline scheduled task that
fires at 8 AM and 5 PM CT). DuckDB allows one writer + many readers,
but if the API holds an exclusive lock the writer is blocked and
vice versa.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import duckdb

from packages.common.config import settings
from packages.common.logging import log
from packages.inference.db import get_sqlite_conn as _sqlite_ctx


def get_duckdb_conn() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh READ-ONLY DuckDB connection scoped to one request.

    Opening read-only allows the API to keep serving while the daily
    ingest/predict pipeline is running with the database open for writes.

    If the database file doesn't exist yet (fresh install) we fall back
    to the standard read-write connect path which will create the file.
    """
    db_path = settings.duckdb_path
    if not Path(db_path).exists():
        # Fresh install: let _connect create the file with the DDL.
        from packages.ingestion.storage import _connect
        conn = _connect()
    else:
        try:
            conn = duckdb.connect(db_path, read_only=True)
        except duckdb.IOException as exc:
            log.warning(f"DuckDB read-only open failed: {exc!r}; retrying read-write")
            from packages.ingestion.storage import _connect
            conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def get_sqlite_conn() -> Iterator[sqlite3.Connection]:
    """Yield a fresh SQLite connection scoped to one request."""
    with _sqlite_ctx() as conn:
        yield conn
