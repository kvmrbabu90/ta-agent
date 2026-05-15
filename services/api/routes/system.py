"""System status: surface "freshness" indicators to the UI.

The pipeline (Windows Scheduled Tasks at 08:00 and 17:00 CT) runs five
steps and finishes with `paper_backtest`. Whenever that step completes,
``paper_runs.started_at`` for ``run_id='default'`` is overwritten with
``utcnow().isoformat()``. That's the most reliable "last pipeline ran
to completion" timestamp we have — surface it as `last_refresh_utc`.

We also expose `latest_bar_date` (max bar_date in ohlcv_daily) so the
UI can show *data through* date — important because the 8 AM CT run
fires before equity markets open (8:30 AM CT), so on pre-market days
no new bars arrive even though the refresh "happened".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
from fastapi import APIRouter

from packages.common.config import settings
from packages.common.logging import log
from packages.paper_trading import init_paper_db
from services.api.schemas import SystemStatusResponse

router = APIRouter(prefix="/system", tags=["meta"])

_PAPER_DB = str(Path(settings.predictions_sqlite_path).parent / "paper.sqlite")


@router.get("/status", response_model=SystemStatusResponse)
def status() -> SystemStatusResponse:
    last_refresh = _last_pipeline_run()
    latest_bar = _latest_bar_date()
    return SystemStatusResponse(
        last_refresh_utc=last_refresh,
        latest_bar_date=latest_bar,
    )


def _last_pipeline_run() -> str | None:
    """Most recent paper_runs.started_at across all run_ids.

    The default backtest run is overwritten at the end of every pipeline
    fire, so its started_at == "when the last pipeline run finished".
    """
    try:
        init_paper_db(_PAPER_DB)
        # check_same_thread=False: see services/api/routes/paper.py — FastAPI
        # threadpool may dispatch one request across multiple worker threads.
        conn = sqlite3.connect(_PAPER_DB, check_same_thread=False)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"paper.sqlite open failed: {exc!r}")
        return None
    try:
        row = conn.execute(
            "SELECT started_at FROM paper_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _latest_bar_date() -> str | None:
    """Latest bar_date in ohlcv_daily — i.e. how recent the price data is."""
    db_path = settings.duckdb_path
    if not Path(db_path).exists():
        return None
    try:
        conn = duckdb.connect(db_path, read_only=True)
    except duckdb.IOException as exc:
        log.warning(f"duckdb open failed for /system/status: {exc!r}")
        return None
    try:
        row = conn.execute("SELECT MAX(bar_date) FROM ohlcv_daily").fetchone()
        if row and row[0] is not None:
            return str(row[0])
        return None
    finally:
        conn.close()
