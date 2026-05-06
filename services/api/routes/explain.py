"""SHAP attribution route."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from services.api.deps import get_sqlite_conn
from services.api.schemas import ExplainResponse
from services.api.services.predictions_service import explain_for_symbol

router = APIRouter(prefix="/explain", tags=["explain"])


@router.get("/{universe}/{symbol}", response_model=ExplainResponse)
def explain(
    universe: str,
    symbol: str,
    as_of: date | None = Query(default=None),
    top_k: int = Query(5, ge=1, le=30),
    sqlite=Depends(get_sqlite_conn),
) -> ExplainResponse:
    out = explain_for_symbol(sqlite, universe, symbol, as_of, top_k=top_k)
    if out is None:
        raise HTTPException(
            status_code=404,
            detail=f"no explanation available for {universe}/{symbol}@{as_of}",
        )
    return out
