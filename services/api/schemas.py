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


class StrategyEquityPoint(_BaseResponse):
    bar_date: date
    strategy_return: float
    spy_return: float | None = None
    cum_strategy_return: float
    cum_spy_return: float | None = None


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
    # New (frontend-driven): direction-only accuracy excluding tiny preds.
    directional_accuracy: float | None = None
    n_directional_observations: int | None = None
    # Risk-adjusted metrics on the long-short decile spread P&L (vs SPY):
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    spy_sharpe_ratio: float | None = None
    spy_sortino_ratio: float | None = None
    # Cumulative return curves for the strategy vs SPY (chart fuel):
    equity_curve: list[StrategyEquityPoint] = Field(default_factory=list)
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


# ---------------------------------------------------------------------------
# /paper/* — paper-trading API (a backtest-driven equity curve + positions)
# ---------------------------------------------------------------------------


class PaperRunSummary(_BaseResponse):
    run_id: str
    universe: str
    starting_cash: float
    position_size: float
    n_long: int
    n_short: int
    short_threshold: float
    started_at: str
    first_trade_date: date | None = None
    last_trade_date: date | None = None
    final_equity: float | None = None
    final_realized_pnl: float | None = None
    notes: str | None = None
    # v2 strategy fields — overlapping-portfolios + stop-loss + IBKR Lite
    holding_days: int | None = None
    commission_model: str | None = None
    stop_loss_enabled: bool | None = None
    support_lookback_days: int | None = None
    stop_buffer_pct: float | None = None


class PaperEquityPoint(_BaseResponse):
    trade_date: date
    snapshot_kind: str  # 'open_8am_ct' | 'close_5pm_ct'
    equity: float
    cash: float
    long_mv: float
    short_mv: float
    realized_pnl: float
    unrealized_pnl: float


class PaperPosition(_BaseResponse):
    symbol: str
    side: str  # 'long' | 'short'
    qty: float
    entry_price: float
    entry_date: date
    last_price: float | None = None
    unrealized_pnl: float = 0.0
    # Forced-close date from the holding-period rule (entry_date + N trading
    # days, where N = holding_days). The paper engine guarantees this exit
    # unless the stop-loss fires first.
    planned_exit_date: date | None = None
    # Stop-loss level (price the position closes at if hit). Aggregated
    # across lots: the HIGHEST stop level across all of this symbol's
    # active lots (the tightest stop — the most conservative for a long).
    stop_level: float | None = None
    # Number of distinct open lots (entry orders) aggregated into this row.
    # With overlapping portfolios a symbol is often rebought across several
    # trading days; this is how many of those legs are still held.
    lot_count: int = 1


class PaperTrade(_BaseResponse):
    trade_date: date
    symbol: str
    # 'long_open' | 'long_close' | 'stop_close' (v2); 'short_*' kept for legacy rows
    side: str
    qty: float
    fill_price: float
    cash_delta: float
    realized_pnl: float | None = None
    # For close trades, the opening lot's entry date and fill price (joined
    # on lot_id). Null for open trades and for legacy rows with no matching
    # open leg.
    entry_date: date | None = None
    entry_price: float | None = None


class PaperBenchmarkPoint(_BaseResponse):
    """SPY B&H equity rebased to the paper starting capital. One row per
    close-of-day snapshot in the paper equity_curve."""
    trade_date: date
    equity: float


class PaperPostTaxPoint(_BaseResponse):
    """Strategy equity AFTER 30% STCG (reduced-base compounding).
    Same dates as paper equity_curve close snapshots.

    Method: for each completed calendar year y, the year's gain on the
    PRE-TAX equity is taxed at 30%, and that tax is deducted from
    capital at the start of year y+1. Within-year intra-year scaling
    follows the pre-tax curve proportionally. Matches the convention
    used on the Live WF page.

    IBKR Lite commissions are ALREADY in the pre-tax curve (paper engine
    uses commission_model='ibkr_lite' so SEC fee on sells is subtracted
    from cash). So "tax-adjusted" here means *additional* 30% tax on
    top of the existing fee deductions.
    """
    trade_date: date
    equity: float


class PaperSnapshotResponse(_BaseResponse):
    run: PaperRunSummary
    equity_curve: list[PaperEquityPoint] = Field(default_factory=list)
    positions: list[PaperPosition] = Field(default_factory=list)
    # SPY benchmark — same date span as equity_curve, rebased to the
    # paper run's starting capital.
    benchmark_curve: list[PaperBenchmarkPoint] = Field(default_factory=list)
    benchmark_symbol: str | None = None
    # Strategy AFTER 30% STCG (IBKR Lite fees already in equity_curve).
    post_tax_curve: list[PaperPostTaxPoint] = Field(default_factory=list)
    strategy_tax_rate: float = 0.30


class PaperTradesResponse(_BaseResponse):
    run_id: str
    trades: list[PaperTrade] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /system/status — UI freshness indicator
# ---------------------------------------------------------------------------


class NewsVerdict(_BaseResponse):
    symbol: str
    verdict: str  # 'PANIC' | 'RESET' | 'UNCLEAR'
    confidence: float
    key_factors: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    n_sources: int | None = None
    model_name: str | None = None
    trail_5d: float | None = None
    trail_20d: float | None = None
    predicted_return: float | None = None


class NewsVerdictsResponse(_BaseResponse):
    universe: str
    as_of: date
    verdicts: list[NewsVerdict] = Field(default_factory=list)


class SystemStatusResponse(_BaseResponse):
    # ISO-8601 UTC timestamp of the last pipeline run that ran to completion
    # (paper_runs.started_at, updated at the end of the daily pipeline).
    last_refresh_utc: str | None = None
    # Latest bar_date present in ohlcv_daily (YYYY-MM-DD).
    latest_bar_date: str | None = None


# ---------------------------------------------------------------------------
# /performance/model/{universe} — current production model snapshot
# ---------------------------------------------------------------------------


class ModelTargetInfo(_BaseResponse):
    """Per-target (regression/classification) model metadata."""
    target: str  # 'regression' | 'classification'
    model_id: str  # directory name e.g. 'SP500_regression_20260517_064753'
    train_start: date
    train_end: date
    n_features: int
    horizon_days: int
    # Tuned hyperparameters (subset; full config available in the model dir)
    learning_rate: float | None = None
    num_leaves: int | None = None
    min_data_in_leaf: int | None = None
    # CV mean metrics. For regression: mean_daily_ic, mean_daily_rank_ic,
    # mean_decile_spread, hit_rate. For classification: val_accuracy.
    cv_mean_metrics: dict = Field(default_factory=dict)
    cv_std_metrics: dict = Field(default_factory=dict)
    cv_fold_count: int = 0


class ModelInfoResponse(_BaseResponse):
    universe: str
    n_members: int  # current active members in this universe
    training_rows: int | None = None  # row count of the latest training parquet
    training_symbols: int | None = None  # distinct symbols in training data
    training_date_range: list[date] | None = None  # [min_bar_date, max_bar_date]
    targets: list[ModelTargetInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /performance/walkforward/{universe} — tax-adjusted equity curves
# ---------------------------------------------------------------------------


class WalkforwardEquityPoint(_BaseResponse):
    """One per-year row of compounded equity, both pre- and post-tax."""
    year: int
    strategy_return_pct: float       # pre-tax annual return %
    strategy_aftertax_pct: float     # post-tax annual return % (STCG applied)
    strategy_equity: float           # starting=1000, end-of-year post-tax equity
    benchmark_return_pct: float
    benchmark_equity_pretax: float   # mark-to-market pre-tax equity (compounds tax-deferred)
    benchmark_equity_aftertax: float # liquidation value if you sold this year (LTCG applied)


class WalkforwardSummary(_BaseResponse):
    starting_capital: float
    strategy_final_pretax: float
    strategy_final_aftertax: float
    benchmark_final_pretax: float
    benchmark_final_aftertax: float
    outperformance_multiple: float   # strategy / benchmark, both after-tax
    strategy_stcg_rate: float        # tax rate used for strategy
    benchmark_ltcg_rate: float       # tax rate used for benchmark


class WalkforwardResponse(_BaseResponse):
    universe: str
    benchmark_symbol: str            # 'SPY' for US, 'NIFTYBEES' for India
    benchmark_label: str             # human-readable e.g. 'SPY B&H', 'NIFTY 50 B&H'
    currency: str                    # 'USD' or 'INR'
    years: list[WalkforwardEquityPoint] = Field(default_factory=list)
    summary: WalkforwardSummary


# ---------------------------------------------------------------------------
# /performance/strict-wf/{universe} — LIVE strict-walk-forward progress
# ---------------------------------------------------------------------------


class StrictWfYearPoint(_BaseResponse):
    year: int
    strategy_return_pct: float
    # Pre-tax → after-tax haircut, populated ONLY when the calendar year
    # has fully elapsed in the WF (so partial-year picks aren't unfairly
    # taxed before they've had a chance to settle). For losing years the
    # value equals strategy_return_pct (losses pass through; capital-loss
    # carryforward is not modeled at this level). Rate: 30% short-term
    # US capital gains.
    strategy_return_after_tax_pct: float | None = None
    benchmark_return_pct: float | None = None
    # Benchmark (SPY) max intra-year drawdown for the same window the
    # strategy traded. Year-internal peak-to-trough %, signed positive
    # (e.g. 19.0 means SPY drew down 19%). Companion to the strategy
    # max_dd_pct column — exposes "how stressful was this year for the
    # benchmark" so the reader can intuit when the strategy's defensive
    # alpha had room to operate.
    benchmark_max_dd_pct: float | None = None
    # Peak VIX (intraday high) during the strategy's actual trading
    # window for the year. Sourced from a standalone parquet
    # (data/raw/vix_daily.parquet); never touches market.duckdb. None if
    # the VIX series is missing or the year window has no overlapping
    # VIX bars.
    vix_peak: float | None = None
    excess_pct: float | None = None
    sharpe: float | None = None
    max_dd_pct: float | None = None
    n_days: int = 0


class StrictWfSummary(_BaseResponse):
    starting_capital: float = 1000.0
    strategy_cum_return_pct: float = 0.0
    # After-tax cumulative — same equity simulation but with capital
    # reduced by the prior year's tax bill on Jan 1 of each subsequent
    # year. Populated whenever at least one calendar year has fully
    # elapsed in the WF data; equals strategy_cum_return_pct as long as
    # no year is complete yet. Display-only; pre-tax pipeline unchanged.
    strategy_cum_return_after_tax_pct: float | None = None
    strategy_annualized_after_tax_pct: float | None = None
    strategy_multiple_after_tax: float | None = None
    benchmark_cum_return_pct: float = 0.0
    # Benchmark cum after LTCG — assumes a single liquidation at the
    # final date (B&H investor sells once). 15% US federal LTCG,
    # mid-bracket; Texas resident → no state income tax add-on.
    # Populated only when bench_cum_return_pct > 0 (no tax on losses).
    benchmark_cum_return_after_tax_pct: float | None = None
    strategy_annualized_pct: float = 0.0
    benchmark_annualized_pct: float = 0.0
    n_years: float = 0.0
    strategy_multiple: float = 1.0  # final equity / starting capital


class StrictWfProgress(_BaseResponse):
    retrains_complete: int = 0
    retrains_total: int = 0
    last_retrain_date: str | None = None      # 'YYYY-MM-DD' of last completed train_end
    last_retrain_at_utc: str | None = None    # ISO timestamp
    avg_retrain_minutes: float | None = None  # rolling-average over last 5 retrains
    eta_completion_utc: str | None = None
    is_running: bool = False                   # heuristic: progress in last 2h


class StrictWfDailyPoint(_BaseResponse):
    """Single trading day inside a strict-WF month detail payload."""
    date: str
    strategy_pct: float | None = None
    benchmark_pct: float | None = None
    excess_pct: float | None = None


class StrictWfHolding(_BaseResponse):
    """One symbol held during a strict-WF month, with avg exposure stats."""
    symbol: str
    days_held: int
    avg_weight_pct: float


class StrictWfMonthDetail(_BaseResponse):
    """Drill-down payload for a single (year, month) heatmap cell.

    Returned by /performance/strict-wf/{universe}/month/{year}/{month}.
    Used by the dashboard's click-to-expand cell card. Pure quantitative
    derivation from paper_equity + paper_positions + ohlcv_daily — no
    commentary.
    """
    universe: str
    year: int
    month: int
    n_days: int = 0

    # Headline pcts (same convention as the heatmap cell — first→last close).
    strategy_pct: float | None = None
    benchmark_pct: float | None = None
    excess_pct: float | None = None

    # Risk stats over the month.
    sharpe: float | None = None
    max_dd_pct: float | None = None
    vol_pct: float | None = None  # annualized stdev of daily returns

    # Daily series for the chart (sorted ascending).
    daily: list[StrictWfDailyPoint] = Field(default_factory=list)

    # Highlight days within the month (best/worst by EXCESS pct).
    best_days: list[StrictWfDailyPoint] = Field(default_factory=list)
    worst_days: list[StrictWfDailyPoint] = Field(default_factory=list)

    # Top holdings during the month (top 10 by avg weight).
    top_holdings: list[StrictWfHolding] = Field(default_factory=list)


class StrictWfAnalysisResponse(_BaseResponse):
    """Latest published WF analysis (written by the /wf-analysis slash
    command). Markdown body + YAML frontmatter metadata."""
    markdown: str | None = None
    covers_through: str | None = None  # "YYYY-MM-DD" of last completed retrain
    retrain_count: int | None = None
    written_at: str | None = None  # ISO timestamp


class StrictWfMonthlyExcessCell(_BaseResponse):
    """One (year, month) cell of the monthly excess heatmap.

    All three pct fields use the convention:
        ret = (last_close_of_month / first_close_of_month) - 1
    so they're directly comparable. ``excess_pct`` is the strategy
    minus the benchmark for that calendar month. Months where either
    leg has no data are omitted entirely from the list.
    """

    year: int
    month: int  # 1..12
    strategy_pct: float | None = None
    benchmark_pct: float | None = None
    excess_pct: float | None = None


class StrictWfEquityCurve(_BaseResponse):
    """Columnar equity time series for the Live WF chart.

    Three parallel arrays of the same length:
        dates              — 'YYYY-MM-DD' strings, sorted ascending
        equity_pre_tax     — paper-engine equity at each date (pre-tax)
        equity_post_tax    — pre-tax minus cumulative tax paid up to date
        benchmark_equity   — buy-and-hold benchmark equity, indexed to
                             the same starting capital (omitted if no
                             benchmark history covers the WF window)

    Plus a single scalar:
        benchmark_post_ltcg_endpoint — what the B&H investor walks away
            with on the LAST chart date after paying LTCG on the
            unrealized gain. Equals benchmark_equity[-1] when the
            benchmark hasn't gained (no tax on losses). Rendered as a
            single dot on the chart.

    Empty when there are no equity points yet.
    """

    dates: list[str] = Field(default_factory=list)
    equity_pre_tax: list[float] = Field(default_factory=list)
    equity_post_tax: list[float] = Field(default_factory=list)
    benchmark_equity: list[float] = Field(default_factory=list)
    benchmark_post_ltcg_endpoint: float | None = None


class StrictWfResponse(_BaseResponse):
    universe: str
    benchmark_symbol: str
    benchmark_label: str
    currency: str
    progress: StrictWfProgress
    years: list[StrictWfYearPoint] = Field(default_factory=list)
    summary: StrictWfSummary
    equity_curve: StrictWfEquityCurve = Field(default_factory=StrictWfEquityCurve)
    monthly_excess: list[StrictWfMonthlyExcessCell] = Field(default_factory=list)
