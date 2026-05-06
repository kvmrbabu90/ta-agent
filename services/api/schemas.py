"""Pydantic v2 response schemas for the FastAPI backend.

Routes never return raw rows or dicts — every response goes through one of
these models so OpenAPI stays accurate.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class _BaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# /universes
# ---------------------------------------------------------------------------


class UniverseInfo(_BaseResponse):
    name: str
    n_members: int


class MemberInfo(_BaseResponse):
    symbol: str
    company_name: str | None = None
    exchange: str | None = None


# ---------------------------------------------------------------------------
# /predictions/top
# ---------------------------------------------------------------------------


class TopPick(_BaseResponse):
    rank: int
    symbol: str
    company_name: str | None = None
    predicted_return_5d: float
    predicted_quintile: int | None = None
    top_quintile_proba: float | None = None
    bottom_quintile_proba: float | None = None
    model_version_regression: str | None = None
    model_version_classification: str | None = None


class TopPicksResponse(_BaseResponse):
    as_of: date
    universe: str
    direction: str
    picks: list[TopPick] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /predictions/{universe}/{symbol}
# ---------------------------------------------------------------------------


class HistoryPoint(_BaseResponse):
    as_of: date
    predicted_return_5d: float
    realized_return_5d: float | None = None
    predicted_quintile: int | None = None
    realized_quintile: int | None = None


class StockHistoryResponse(_BaseResponse):
    universe: str
    symbol: str
    history: list[HistoryPoint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /stocks/{symbol}/ohlcv
# ---------------------------------------------------------------------------


class OHLCVPoint(_BaseResponse):
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class OHLCVResponse(_BaseResponse):
    symbol: str
    bars: list[OHLCVPoint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /performance/{universe}
# ---------------------------------------------------------------------------


class CalibrationBucket(_BaseResponse):
    proba_bucket: str
    predicted_count: int
    actual_top_quintile_rate: float | None = None
    mean_proba: float | None = None


class ICPoint(_BaseResponse):
    date: date
    daily_ic: float
    n_stocks: int


class PerformanceResponse(_BaseResponse):
    universe: str
    lookback_days: int
    n_predictions: int
    n_settled: int
    mean_daily_ic: float | None = None
    std_daily_ic: float | None = None
    ic_t_stat: float | None = None
    mean_daily_rank_ic: float | None = None
    hit_rate: float | None = None
    decile_spread_5d: float | None = None
    calibration: list[CalibrationBucket] = Field(default_factory=list)
    ic_timeseries: list[ICPoint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /explain/{universe}/{symbol}
# ---------------------------------------------------------------------------


class FeatureContribution(_BaseResponse):
    rank: int
    feature_name: str
    feature_value: float | None = None
    shap_value: float
    contribution_direction: str


class ExplainResponse(_BaseResponse):
    universe: str
    symbol: str
    as_of: date
    predicted_return_5d: float | None = None
    top_features: list[FeatureContribution] = Field(default_factory=list)
