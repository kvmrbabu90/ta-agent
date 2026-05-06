"""Predictions routes: top picks + per-stock history."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from services.api.deps import get_duckdb_conn, get_sqlite_conn
from services.api.schemas import StockHistoryResponse, TopPicksResponse
from services.api.services.predictions_service import (
    get_stock_prediction_history,
    get_top_picks,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/top", response_model=TopPicksResponse)
def top_picks(
    universe: str = Query(..., description="Universe name (e.g. SP500, NIFTY100)"),
    direction: str = Query("long", pattern="^(long|short)$"),
    limit: int = Query(20, ge=1, le=500),
    as_of: date | None = Query(default=None, description="Defaults to MAX(as_of) in DB"),
    sqlite=Depends(get_sqlite_conn),
    duck=Depends(get_duckdb_conn),
) -> TopPicksResponse:
    try:
        return get_top_picks(sqlite, duck, universe, direction, limit, as_of)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{universe}/{symbol}", response_model=StockHistoryResponse)
def stock_prediction_history(
    universe: str,
    symbol: str,
    lookback_days: int = Query(180, ge=1, le=2000),
    sqlite=Depends(get_sqlite_conn),
) -> StockHistoryResponse:
    out = get_stock_prediction_history(sqlite, universe, symbol, lookback_days)
    if out is None:
        raise HTTPException(
            status_code=404,
            detail=f"no predictions found for {universe}/{symbol} in the last {lookback_days} days",
        )
    return out
