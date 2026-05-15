"""News verdicts API.

Returns the LLM-generated PANIC/RESET/UNCLEAR verdicts for the model's
top long picks. Audit-only: this is read by the dashboard to show a chip
next to each pick but does not drive trading decisions.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Query

from packages.common.config import settings
from packages.news.storage import init_news_db
from services.api.schemas import NewsVerdict, NewsVerdictsResponse

router = APIRouter(prefix="/news", tags=["news"])

_NEWS_DB = str(Path(settings.predictions_sqlite_path).parent / "news.sqlite")


def _conn() -> sqlite3.Connection:
    init_news_db(_NEWS_DB)
    # check_same_thread=False — same FastAPI threadpool reasoning as
    # services/api/routes/paper.py.
    return sqlite3.connect(_NEWS_DB, check_same_thread=False)


def _latest_verdict_date(conn: sqlite3.Connection, universe: str) -> date | None:
    row = conn.execute(
        "SELECT MAX(as_of) FROM predictions_news_verdict WHERE universe = ?",
        (universe,),
    ).fetchone()
    if row and row[0]:
        return date.fromisoformat(row[0])
    return None


@router.get("/verdicts", response_model=NewsVerdictsResponse)
def verdicts(
    universe: str = Query("SP500"),
    as_of: date | None = Query(
        default=None,
        description="Defaults to MAX(as_of) in predictions_news_verdict.",
    ),
) -> NewsVerdictsResponse:
    conn = _conn()
    try:
        if as_of is None:
            as_of = _latest_verdict_date(conn, universe)
        if as_of is None:
            # No verdicts yet — empty list is a valid response.
            return NewsVerdictsResponse(universe=universe, as_of=date.today(), verdicts=[])
        rows = conn.execute(
            """
            SELECT symbol, verdict, confidence, key_factors_json,
                   evidence_sources_json, n_sources, model_name,
                   trail_5d, trail_20d, predicted_return
            FROM predictions_news_verdict
            WHERE universe = ? AND as_of = ?
            ORDER BY symbol
            """,
            (universe, as_of.isoformat()),
        ).fetchall()
        verdicts_out = [
            NewsVerdict(
                symbol=r[0],
                verdict=r[1],
                confidence=r[2],
                key_factors=_safe_json_list(r[3]),
                evidence_sources=_safe_json_list(r[4]),
                n_sources=r[5],
                model_name=r[6],
                trail_5d=r[7],
                trail_20d=r[8],
                predicted_return=r[9],
            )
            for r in rows
        ]
        return NewsVerdictsResponse(universe=universe, as_of=as_of, verdicts=verdicts_out)
    finally:
        conn.close()


def _safe_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return []
    except (json.JSONDecodeError, TypeError):
        return []
