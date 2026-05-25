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
    StrictWfAnalysisResponse,
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
    "/strict-wf/{universe}/analysis",
    response_model=StrictWfAnalysisResponse,
)
def strict_wf_analysis(universe: str) -> StrictWfAnalysisResponse:
    """Return the latest published WF analysis markdown for ``universe``.

    Written by the /wf-analysis slash command, stored in
    data/processed/walkforward_10yr_strict/latest_analysis.md. Reads with
    a tiny YAML-frontmatter parse so the front-matter metadata (covers
    through, retrain count, written-at timestamp) is available to the
    UI separately from the body markdown.
    """
    import os
    from fastapi import HTTPException

    # Mapping universe → analysis path. Hardcoded for now since SP500 is
    # the only live strict-WF.
    paths = {
        "SP500": "data/processed/walkforward_10yr_strict/latest_analysis.md",
    }
    p = paths.get(universe)
    if p is None:
        raise HTTPException(status_code=404, detail=f"no analysis path for {universe!r}")
    if not os.path.exists(p):
        # File hasn't been written yet — return empty payload so the UI
        # can render a placeholder.
        return StrictWfAnalysisResponse()
    try:
        with open(p, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return StrictWfAnalysisResponse()

    # Tiny inline YAML-frontmatter parse. Expected shape:
    #   ---
    #   universe: SP500
    #   covers_through: "YYYY-MM-DD"
    #   retrain_count: N
    #   written_at: "ISO-8601"
    #   ---
    #   ...body...
    covers_through: str | None = None
    retrain_count: int | None = None
    written_at: str | None = None
    body = content
    if content.startswith("---"):
        try:
            end = content.index("\n---", 3)
            front = content[3:end].strip()
            body = content[end + 4 :].lstrip("\n")
            for line in front.splitlines():
                line = line.strip()
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key == "covers_through":
                    covers_through = val
                elif key == "retrain_count":
                    try:
                        retrain_count = int(val)
                    except ValueError:
                        pass
                elif key == "written_at":
                    written_at = val
        except ValueError:
            # No closing --- found; treat the whole thing as body.
            pass

    return StrictWfAnalysisResponse(
        markdown=body,
        covers_through=covers_through,
        retrain_count=retrain_count,
        written_at=written_at,
    )


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
