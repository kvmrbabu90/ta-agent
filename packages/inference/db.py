"""SQLite-backed predictions log.

Why SQLite (and not DuckDB) for this:
    Predictions are transactional, not analytical. We do many small upserts
    + selective lookups, plus the API will read this. SQLite handles that
    pattern with zero ops; DuckDB is overkill for it.

The DB lives at ``settings.predictions_sqlite_path``. The schema is
created idempotently on every connection.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from packages.common.config import settings

_PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions_log (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    universe                      TEXT NOT NULL,
    symbol                        TEXT NOT NULL,
    as_of                         DATE NOT NULL,
    horizon_days                  INTEGER NOT NULL,
    predicted_return              REAL NOT NULL,
    predicted_quintile            INTEGER,
    top_quintile_proba            REAL,
    bottom_quintile_proba         REAL,
    model_version_regression      TEXT,
    model_version_classification  TEXT,
    realized_return               REAL,
    realized_quintile             INTEGER,
    settled_at                    TIMESTAMP,
    created_at                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (universe, symbol, as_of, horizon_days)
);
CREATE INDEX IF NOT EXISTS pred_universe_date ON predictions_log (universe, as_of);
CREATE INDEX IF NOT EXISTS pred_settled ON predictions_log (settled_at);
"""


def init_predictions_db(path: str | Path | None = None) -> None:
    """Create the predictions DB + schema if absent. Safe to call repeatedly."""
    db_path = str(path) if path else settings.predictions_sqlite_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_PREDICTIONS_DDL)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_sqlite_conn(
    path: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Connection context manager. Schema is ensured on every open.

    ``check_same_thread=False`` is required because FastAPI runs sync
    dependencies and sync handlers in its threadpool — and the worker
    thread that yields the connection is NOT guaranteed to be the same
    one that executes the handler. Each request gets its OWN connection
    (no sharing across requests), so the cross-thread access is safe;
    it's purely sequential within one request's lifecycle.
    """
    db_path = str(path) if path else settings.predictions_sqlite_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    try:
        conn.executescript(_PREDICTIONS_DDL)
        conn.commit()
        yield conn
    finally:
        conn.close()
