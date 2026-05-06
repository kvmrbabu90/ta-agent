"""Performance summary route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from services.api.deps import get_sqlite_conn
from services.api.schemas import PerformanceResponse
from services.api.services.predictions_service import get_performance

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("/{universe}", response_model=PerformanceResponse)
def performance(
    universe: str,
    lookback_days: int = Query(90, ge=7, le=2000),
    sqlite=Depends(get_sqlite_conn),
) -> PerformanceResponse:
    return get_performance(sqlite, universe, lookback_days)
