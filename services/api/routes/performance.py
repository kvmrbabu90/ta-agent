"""Performance summary routes.

  GET /performance/model/{universe}             — current production model snapshot
  GET /performance/walkforward/{universe}       — tax-adjusted WF equity vs benchmark
  GET /performance/{universe}                   — settled-prediction quality + equity curve

Order matters: the static-prefix routes (model, walkforward) must be
registered BEFORE the catch-all /{universe} or FastAPI will route
'/performance/model' as universe='model'.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from services.api.deps import get_duckdb_conn, get_sqlite_conn
from services.api.schemas import (
    ModelInfoResponse,
    PerformanceResponse,
    StrictWfMonthDetail,
    StrictWfResponse,
    WalkforwardResponse,
)
from services.api.services.predictions_service import (
    get_model_info,
    get_performance,
    get_strict_wf_month_detail,
    get_strict_wf_status,
    get_walkforward_taxadjusted,
)

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("/model/{universe}", response_model=ModelInfoResponse)
def model_info(
    universe: str,
    duck=Depends(get_duckdb_conn),
) -> ModelInfoResponse:
    """Current production model metadata: training window, CV results, hyperparams."""
    return get_model_info(duck, universe)


@router.get("/strict-wf/{universe}", response_model=StrictWfResponse)
def strict_wf(
    universe: str,
    duck=Depends(get_duckdb_conn),
) -> StrictWfResponse:
    """Live status of the strict (per-retrain Optuna, no look-ahead) WF.
    Polls predictions.sqlite mtime + count; returns per-year metrics +
    progress + ETA. Cached in-memory by file mtime."""
    from fastapi import HTTPException
    try:
        return get_strict_wf_status(duck, universe)
    except ValueError as exc:
        # Unknown universe → 404 rather than a 500 stack trace. Happens
        # if a stale client (or someone curl-poking) requests a universe
        # we no longer support.
        raise HTTPException(status_code=404, detail=str(exc))


@router.get(
    "/strict-wf/{universe}/month/{year}/{month}",
    response_model=StrictWfMonthDetail,
)
def strict_wf_month_detail(
    universe: str,
    year: int,
    month: int,
    duck=Depends(get_duckdb_conn),
) -> StrictWfMonthDetail:
    """Drill-down for a single heatmap cell. Daily strategy/SPY returns,
    best/worst days by excess, top holdings during the month, risk stats.
    Used by the UI's click-to-expand cell card."""
    from fastapi import HTTPException
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail=f"month must be 1..12, got {month}")
    try:
        return get_strict_wf_month_detail(duck, universe, year, month)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/walkforward/{universe}", response_model=WalkforwardResponse)
def walkforward(
    universe: str,
    duck=Depends(get_duckdb_conn),
) -> WalkforwardResponse:
    """Tax-adjusted per-year equity curve from the walk-forward backtest,
    compared to a buy-and-hold benchmark (SPY for US, NIFTYBEES for India).

    Tax model:
      - Strategy: STCG (25% US blended, 20% India) applied to net P&L each year.
        5-day holds => all gains short-term.
      - Benchmark: LTCG (15% US, 12.5% India) applied ONLY at terminal sale;
        capital gains compound tax-deferred.
    """
    return get_walkforward_taxadjusted(duck, universe)


@router.get("/{universe}", response_model=PerformanceResponse)
def performance(
    universe: str,
    lookback_days: int = Query(90, ge=7, le=2000),
    sqlite=Depends(get_sqlite_conn),
) -> PerformanceResponse:
    """Recent prediction-quality metrics (Sharpe, IC, decile spread, calibration)."""
    return get_performance(sqlite, universe, lookback_days)
