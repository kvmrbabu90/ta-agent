"""Per-stock OHLCV route."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query

from services.api.deps import get_duckdb_conn
from services.api.schemas import OHLCVResponse
from services.api.services.predictions_service import get_ohlcv_for_charting

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/{symbol}/ohlcv", response_model=OHLCVResponse)
def ohlcv(
    symbol: str,
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    duck=Depends(get_duckdb_conn),
) -> OHLCVResponse:
    end = end or date.today()
    start = start or (end - timedelta(days=365))
    return get_ohlcv_for_charting(duck, symbol, start, end)
