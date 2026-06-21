"""Service layer: SQL + business logic.

Routes are thin wrappers around these functions. Connection objects are
passed in (FastAPI Depends), so unit tests can supply an in-memory DB.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from packages.common.logging import log
from packages.modeling.evaluate import (
    decile_spread,
    information_coefficient,
)
from services.api.schemas import (
    CalibrationBucket,
    ExplainResponse,
    FeatureContribution,
    HistoryPoint,
    ICPoint,
    MemberInfo,
    ModelInfoResponse,
    ModelTargetInfo,
    OHLCVPoint,
    OHLCVResponse,
    PerformanceResponse,
    StockHistoryResponse,
    StrategyEquityPoint,
    StrictWfDailyPoint,
    StrictWfHolding,
    StrictWfMonthDetail,
    StrictWfProgress,
    StrictWfResponse,
    StrictWfSummary,
    StrictWfYearPoint,
    TopPick,
    TopPicksResponse,
    UniverseInfo,
    WalkforwardEquityPoint,
    WalkforwardResponse,
    WalkforwardSummary,
)

_CALIBRATION_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_TOP_QUINTILE = 4  # n_classes - 1, default n_classes=5


# ---------------------------------------------------------------------------
# /universes
# ---------------------------------------------------------------------------


def list_universes(duck: duckdb.DuckDBPyConnection) -> list[UniverseInfo]:
    rows = duck.execute(
        """
        SELECT universe, COUNT(DISTINCT symbol) AS n_members
        FROM index_membership
        WHERE end_date IS NULL
        GROUP BY universe
        ORDER BY universe
        """
    ).fetchall()
    return [UniverseInfo(name=u, n_members=int(n)) for u, n in rows]


def list_members(
    duck: duckdb.DuckDBPyConnection, universe: str, as_of: date
) -> list[MemberInfo]:
    rows = duck.execute(
        """
        SELECT symbol, company_name, exchange
        FROM index_membership
        WHERE universe = ?
          AND start_date <= ?
          AND (end_date IS NULL OR end_date >= ?)
        ORDER BY symbol
        """,
        [universe, as_of, as_of],
    ).fetchall()
    return [
        MemberInfo(symbol=s, company_name=c, exchange=e) for s, c, e in rows
    ]


# ---------------------------------------------------------------------------
# /predictions/top
# ---------------------------------------------------------------------------


def _company_names_map(
    duck: duckdb.DuckDBPyConnection, universe: str, symbols: list[str]
) -> dict[str, str | None]:
    if not symbols:
        return {}
    placeholders = ",".join("?" * len(symbols))
    rows = duck.execute(
        f"""
        SELECT symbol, MAX(company_name)
        FROM index_membership
        WHERE universe = ? AND symbol IN ({placeholders})
        GROUP BY symbol
        """,
        [universe, *symbols],
    ).fetchall()
    return {sym: name for sym, name in rows}


def get_top_picks(
    sqlite_conn: sqlite3.Connection,
    duck: duckdb.DuckDBPyConnection,
    universe: str,
    direction: str,
    limit: int,
    as_of: date | None = None,
) -> TopPicksResponse:
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    if limit < 1 or limit > 500:
        raise ValueError(f"limit must be in [1, 500], got {limit}")

    if as_of is None:
        row = sqlite_conn.execute(
            "SELECT MAX(as_of) FROM predictions_log WHERE universe = ?",
            [universe],
        ).fetchone()
        if row is None or row[0] is None:
            return TopPicksResponse(
                as_of=date.today(), universe=universe, direction=direction, picks=[]
            )
        as_of = (
            date.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
        )

    # Conviction-weighted ranking — matches Kubera V1 strategy spec.
    # combined_score = predicted_return × (1 + direction_agreement)
    # where direction_agreement = top_q_proba − bot_q_proba.
    #
    # This is what the simulator (packages/paper_trading/engine.py),
    # the live Alpaca engine (services/alpaca/engine.py), and the
    # /paper/next-day-picks endpoint all use. Ranking the Dashboard's
    # Top picks by raw predicted_return surfaces names where the
    # classification head disagrees with the regression head (e.g.
    # NTAP with top_q=0.31 / bot_q=0.39 — classifier puts it in the
    # bottom quintile despite a positive regression pred). The conviction
    # weighting correctly demotes those.
    order = "DESC" if direction == "long" else "ASC"
    rows = sqlite_conn.execute(
        f"""
        SELECT symbol, predicted_return, predicted_quintile,
               top_quintile_proba, bottom_quintile_proba,
               model_version_regression, model_version_classification
        FROM predictions_log
        WHERE universe = ? AND as_of = ?
        ORDER BY predicted_return * (1.0 + (COALESCE(top_quintile_proba, 0.0)
                                            - COALESCE(bottom_quintile_proba, 0.0))) {order},
                 symbol ASC
        LIMIT ?
        """,
        [universe, as_of.isoformat() if isinstance(as_of, date) else as_of, limit],
    ).fetchall()

    symbols = [r[0] for r in rows]
    company_names = _company_names_map(duck, universe, symbols)

    picks = [
        TopPick(
            rank=i + 1,
            symbol=r[0],
            company_name=company_names.get(r[0]),
            predicted_return_5d=float(r[1]),
            predicted_quintile=int(r[2]) if r[2] is not None else None,
            top_quintile_proba=float(r[3]) if r[3] is not None else None,
            bottom_quintile_proba=float(r[4]) if r[4] is not None else None,
            model_version_regression=r[5],
            model_version_classification=r[6],
        )
        for i, r in enumerate(rows)
    ]
    return TopPicksResponse(
        as_of=as_of, universe=universe, direction=direction, picks=picks
    )


# ---------------------------------------------------------------------------
# /predictions/{universe}/{symbol}
# ---------------------------------------------------------------------------


def get_stock_prediction_history(
    sqlite_conn: sqlite3.Connection,
    universe: str,
    symbol: str,
    lookback_days: int,
) -> StockHistoryResponse | None:
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    rows = sqlite_conn.execute(
        """
        SELECT as_of, predicted_return, realized_return,
               predicted_quintile, realized_quintile
        FROM predictions_log
        WHERE universe = ? AND symbol = ? AND as_of >= ?
        ORDER BY as_of
        """,
        [universe, symbol, cutoff],
    ).fetchall()
    if not rows:
        return None

    history = [
        HistoryPoint(
            as_of=date.fromisoformat(r[0]) if isinstance(r[0], str) else r[0],
            predicted_return_5d=float(r[1]),
            realized_return_5d=float(r[2]) if r[2] is not None else None,
            predicted_quintile=int(r[3]) if r[3] is not None else None,
            realized_quintile=int(r[4]) if r[4] is not None else None,
        )
        for r in rows
    ]
    return StockHistoryResponse(universe=universe, symbol=symbol, history=history)


# ---------------------------------------------------------------------------
# /stocks/{symbol}/ohlcv
# ---------------------------------------------------------------------------


def get_ohlcv_for_charting(
    duck: duckdb.DuckDBPyConnection,
    symbol: str,
    start: date,
    end: date,
) -> OHLCVResponse:
    df = duck.execute(
        """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY symbol, exchange, bar_date
                ORDER BY ingested_at DESC
            ) AS rn
            FROM ohlcv_daily
            WHERE symbol = ? AND bar_date BETWEEN ? AND ?
        )
        SELECT bar_date, open, high, low, close, volume
        FROM ranked WHERE rn = 1
        ORDER BY bar_date
        """,
        [symbol, start, end],
    ).df()
    bars = [
        OHLCVPoint(
            bar_date=pd.Timestamp(r.bar_date).date()
            if not isinstance(r.bar_date, date)
            else r.bar_date,
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=int(r.volume),
        )
        for r in df.itertuples()
    ]
    return OHLCVResponse(symbol=symbol, bars=bars)


# ---------------------------------------------------------------------------
# /performance/{universe}
# ---------------------------------------------------------------------------


def _calibration_table(df: pd.DataFrame) -> list[CalibrationBucket]:
    if (
        "top_quintile_proba" not in df.columns
        or "realized_quintile" not in df.columns
        or df["top_quintile_proba"].isna().all()
        or df["realized_quintile"].isna().all()
    ):
        return []
    sub = df.dropna(subset=["top_quintile_proba", "realized_quintile"]).copy()
    if sub.empty:
        return []
    sub["bucket"] = pd.cut(
        sub["top_quintile_proba"],
        bins=_CALIBRATION_BINS,
        include_lowest=True,
    )
    grouped = sub.groupby("bucket", observed=True).agg(
        predicted_count=("realized_quintile", "size"),
        mean_proba=("top_quintile_proba", "mean"),
        actual_top_quintile_rate=(
            "realized_quintile",
            lambda x: float((x == _TOP_QUINTILE).mean()),
        ),
    )
    out: list[CalibrationBucket] = []
    for bucket_label, row in grouped.iterrows():
        out.append(
            CalibrationBucket(
                proba_bucket=str(bucket_label),
                predicted_count=int(row["predicted_count"]),
                mean_proba=float(row["mean_proba"]),
                actual_top_quintile_rate=float(row["actual_top_quintile_rate"]),
            )
        )
    return out


def _ic_timeseries(df: pd.DataFrame) -> list[ICPoint]:
    if df.empty:
        return []
    out: list[ICPoint] = []
    for d, group in df.groupby("as_of"):
        if len(group) < 5:
            continue
        ic = information_coefficient(
            group["realized_return"].to_numpy(),
            group["predicted_return"].to_numpy(),
        )
        if np.isnan(ic):
            continue
        d_norm = d if isinstance(d, date) else date.fromisoformat(str(d))
        out.append(ICPoint(date=d_norm, daily_ic=float(ic), n_stocks=len(group)))
    return out


def _annualized_sharpe(daily_returns: np.ndarray) -> float | None:
    """Annualized Sharpe assuming risk-free = 0. Need ≥5 obs."""
    valid = daily_returns[np.isfinite(daily_returns)]
    if len(valid) < 5:
        return None
    mu = float(np.mean(valid))
    sigma = float(np.std(valid, ddof=1))
    if sigma == 0:
        return None
    return float(mu / sigma * np.sqrt(252))


def _annualized_sortino(daily_returns: np.ndarray) -> float | None:
    valid = daily_returns[np.isfinite(daily_returns)]
    if len(valid) < 5:
        return None
    mu = float(np.mean(valid))
    downside = valid[valid < 0.0]
    if len(downside) == 0:
        return None
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    if downside_dev == 0:
        return None
    return float(mu / downside_dev * np.sqrt(252))


def _strategy_daily_returns(settled: pd.DataFrame) -> pd.Series:
    """Build a per-date long-short decile-spread return series.

    For each date with ≥10 settled predictions:
      - top decile = top 10% by predicted_return
      - bottom decile = bottom 10%
      - daily return = mean(top.realized) - mean(bottom.realized)

    Returns a pandas Series indexed by as_of (date), values = daily returns.
    """
    if settled.empty:
        return pd.Series(dtype=float)
    out: dict = {}
    for d, group in settled.groupby("as_of"):
        if len(group) < 10:
            continue
        sorted_group = group.sort_values("predicted_return")
        n = max(1, int(round(len(group) * 0.1)))
        bot = sorted_group.head(n)["realized_return"].to_numpy()
        top = sorted_group.tail(n)["realized_return"].to_numpy()
        if len(top) == 0 or len(bot) == 0:
            continue
        out[d] = float(top.mean() - bot.mean())
    if not out:
        return pd.Series(dtype=float)
    return pd.Series(out).sort_index()


def _spy_daily_returns(start: date, end: date) -> pd.Series:
    """Pull SPY 5-day forward returns from market.duckdb.

    We use 5-day returns to match the strategy horizon (model predicts 5d
    forward return; strategy holds for similar horizon implicitly).
    Returns a Series indexed by as_of date.
    """
    import duckdb

    from packages.common.config import settings as cfg

    duck = duckdb.connect(cfg.duckdb_path, read_only=True)
    try:
        df = duck.execute(
            """
            WITH ranked AS (
                SELECT bar_date, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol, bar_date ORDER BY ingested_at DESC) AS rn
                FROM ohlcv_daily
                WHERE symbol = 'SPY' AND bar_date BETWEEN ? AND ?
            )
            SELECT bar_date, close FROM ranked WHERE rn = 1 ORDER BY bar_date
            """,
            [start, end + timedelta(days=10)],
        ).df()
    finally:
        duck.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
    df = df.sort_values("bar_date").reset_index(drop=True)
    df["spy_5d_return"] = df["close"].pct_change(periods=5).shift(-5)
    return df.dropna(subset=["spy_5d_return"]).set_index("bar_date")["spy_5d_return"]


def _build_equity_curve(
    strategy_ret: pd.Series, spy_ret: pd.Series
) -> list[StrategyEquityPoint]:
    """Combine strategy + SPY daily returns into a cumulative-return curve."""
    if strategy_ret.empty:
        return []
    df = pd.DataFrame({"strategy_return": strategy_ret})
    if not spy_ret.empty:
        df = df.join(spy_ret.rename("spy_return"), how="left")
    else:
        df["spy_return"] = float("nan")
    df = df.sort_index()
    df["cum_strategy_return"] = (1.0 + df["strategy_return"].fillna(0.0)).cumprod() - 1.0
    df["cum_spy_return"] = (1.0 + df["spy_return"].fillna(0.0)).cumprod() - 1.0
    out = []
    for d, row in df.iterrows():
        out.append(StrategyEquityPoint(
            bar_date=d,
            strategy_return=float(row["strategy_return"]),
            spy_return=(None if pd.isna(row["spy_return"]) else float(row["spy_return"])),
            cum_strategy_return=float(row["cum_strategy_return"]),
            cum_spy_return=(None if pd.isna(row["cum_spy_return"]) else float(row["cum_spy_return"])),
        ))
    return out


def get_performance(
    sqlite_conn: sqlite3.Connection,
    universe: str,
    lookback_days: int,
) -> PerformanceResponse:
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    df = pd.read_sql_query(
        """
        SELECT *
        FROM predictions_log
        WHERE universe = ? AND as_of >= ?
        """,
        sqlite_conn,
        params=[universe, cutoff],
    )

    if df.empty:
        return PerformanceResponse(
            universe=universe, lookback_days=lookback_days,
            n_predictions=0, n_settled=0, calibration=[], ic_timeseries=[],
        )

    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    settled = df[df["realized_return"].notna()].copy()

    daily_ics: list[float] = []
    for _, group in settled.groupby("as_of"):
        if len(group) < 5:
            continue
        ic = information_coefficient(
            group["realized_return"].to_numpy(),
            group["predicted_return"].to_numpy(),
        )
        if not np.isnan(ic):
            daily_ics.append(ic)

    def _safe_t_stat(values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        std = float(np.std(values, ddof=1))
        if std == 0:
            return None
        return float(np.mean(values) / (std / np.sqrt(len(values))))

    mean_ic = float(np.mean(daily_ics)) if daily_ics else None
    std_ic = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else None
    t_stat = _safe_t_stat(daily_ics)

    if not settled.empty:
        signs = (
            np.sign(settled["realized_return"].to_numpy())
            == np.sign(settled["predicted_return"].to_numpy())
        )
        hit_rate: float | None = float(signs.mean()) if len(signs) else None
        spread: float | None = decile_spread(
            settled["realized_return"].to_numpy(),
            settled["predicted_return"].to_numpy(),
        )
        if spread is not None and np.isnan(spread):
            spread = None
    else:
        hit_rate = None
        spread = None

    # Directional accuracy excluding "neutral" predictions (|pred| < 0.1%).
    neutral_threshold = 0.001
    directional_acc: float | None = None
    n_directional = 0
    if not settled.empty:
        non_neutral = settled[settled["predicted_return"].abs() > neutral_threshold]
        if not non_neutral.empty:
            non_neutral_signs = (
                np.sign(non_neutral["realized_return"].to_numpy())
                == np.sign(non_neutral["predicted_return"].to_numpy())
            )
            directional_acc = float(non_neutral_signs.mean())
            n_directional = int(len(non_neutral))

    # Strategy daily returns + SPY benchmark + Sharpe/Sortino + equity curve.
    strategy_ret = _strategy_daily_returns(settled)
    if not strategy_ret.empty:
        spy_ret = _spy_daily_returns(strategy_ret.index.min(), strategy_ret.index.max())
    else:
        spy_ret = pd.Series(dtype=float)

    sharpe = _annualized_sharpe(strategy_ret.to_numpy()) if not strategy_ret.empty else None
    sortino = _annualized_sortino(strategy_ret.to_numpy()) if not strategy_ret.empty else None
    spy_sharpe = _annualized_sharpe(spy_ret.to_numpy()) if not spy_ret.empty else None
    spy_sortino = _annualized_sortino(spy_ret.to_numpy()) if not spy_ret.empty else None
    equity_curve = _build_equity_curve(strategy_ret, spy_ret)

    return PerformanceResponse(
        universe=universe,
        lookback_days=lookback_days,
        n_predictions=int(len(df)),
        n_settled=int(len(settled)),
        mean_daily_ic=mean_ic,
        std_daily_ic=std_ic,
        ic_t_stat=t_stat,
        hit_rate=hit_rate,
        decile_spread_5d=spread,
        directional_accuracy=directional_acc,
        n_directional_observations=n_directional,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        spy_sharpe_ratio=spy_sharpe,
        spy_sortino_ratio=spy_sortino,
        equity_curve=equity_curve,
        calibration=_calibration_table(settled),
        ic_timeseries=_ic_timeseries(settled),
    )


# ---------------------------------------------------------------------------
# /explain/{universe}/{symbol}
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _cached_models(universe: str) -> Any:
    """Cache per-universe model loads. Cleared on app restart."""
    from packages.inference.predict import load_models_for_universe
    return load_models_for_universe(universe)


# Per-(universe, as_of) feature cache. Bounded by hand to avoid memory bloat.
_FEATURE_CACHE: dict[tuple[str, date], pd.DataFrame] = {}
_FEATURE_CACHE_MAX = 8
_EXPLAINER_CACHE: dict[str, Any] = {}


def _get_features_for_explain(universe: str, as_of: date, feature_cols: list[str]) -> pd.DataFrame:
    from packages.inference.predict import build_inference_features

    key = (universe, as_of)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    if len(_FEATURE_CACHE) >= _FEATURE_CACHE_MAX:
        # Evict the oldest entry; cheap LRU.
        _FEATURE_CACHE.pop(next(iter(_FEATURE_CACHE)))
    feats = build_inference_features(universe, as_of, feature_cols)
    _FEATURE_CACHE[key] = feats
    return feats


def explain_for_symbol(
    sqlite_conn: sqlite3.Connection,
    universe: str,
    symbol: str,
    as_of: date | None,
    *,
    top_k: int = 5,
) -> ExplainResponse | None:
    """Return SHAP top-K for a single (universe, symbol, as_of) prediction.

    If ``as_of`` is None, uses the most recent logged prediction date.
    Returns None if no matching prediction or feature row is available.
    """
    if as_of is None:
        row = sqlite_conn.execute(
            "SELECT MAX(as_of) FROM predictions_log WHERE universe = ? AND symbol = ?",
            [universe, symbol],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        as_of = date.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]

    pred_row = sqlite_conn.execute(
        """
        SELECT predicted_return
        FROM predictions_log
        WHERE universe = ? AND symbol = ? AND as_of = ?
        """,
        [universe, symbol, as_of.isoformat()],
    ).fetchone()
    predicted_return = float(pred_row[0]) if pred_row and pred_row[0] is not None else None

    try:
        models = _cached_models(universe)
    except FileNotFoundError as exc:
        log.warning(f"explain_for_symbol: no models for {universe}: {exc!r}")
        return None

    feature_cols = list(models.reg_meta["feature_cols"])
    feats = _get_features_for_explain(universe, as_of, feature_cols)
    if feats.empty:
        return None
    sym_row = feats[feats["symbol"] == symbol]
    if sym_row.empty:
        return None

    from packages.inference.explain import explain_predictions

    pseudo_pred = pd.DataFrame({"symbol": [symbol], "as_of": [as_of]})
    matrix = sym_row.head(1)[feature_cols]
    out = explain_predictions(
        pseudo_pred, matrix, models.reg_model, feature_cols, top_k=top_k
    )

    contributions = [
        FeatureContribution(
            rank=int(r.rank),
            feature_name=str(r.feature_name),
            feature_value=(None if pd.isna(r.feature_value) else float(r.feature_value)),
            shap_value=float(r.shap_value),
            contribution_direction=str(r.contribution_direction),
        )
        for r in out.itertuples()
    ]

    return ExplainResponse(
        universe=universe,
        symbol=symbol,
        as_of=as_of,
        predicted_return_5d=predicted_return,
        top_features=contributions,
    )


# ---------------------------------------------------------------------------
# /performance/model/{universe} — current production model snapshot
# ---------------------------------------------------------------------------


def _latest_model_dir(universe: str, target: str) -> str | None:
    """Find the most-recently-saved model directory for (universe, target).

    Directories are named '<UNIVERSE>_<target>_<YYYYMMDD_HHMMSS>'. We pick the
    lexicographically-greatest matching name, which is also the most recent
    because of the timestamp format.
    """
    import os
    from packages.common.config import MODELS_DIR
    prefix = f"{universe}_{target}_"
    candidates = [
        d for d in os.listdir(MODELS_DIR)
        if d.startswith(prefix) and not d.startswith("BROKEN")
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _read_model_metadata(model_dir: str) -> dict | None:
    """Read metadata.json from a model directory. Returns None if missing."""
    import json
    import os
    from packages.common.config import MODELS_DIR
    p = os.path.join(MODELS_DIR, model_dir, "metadata.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _build_target_info(universe: str, target: str) -> ModelTargetInfo | None:
    model_dir = _latest_model_dir(universe, target)
    if model_dir is None:
        return None
    meta = _read_model_metadata(model_dir)
    if meta is None:
        return None
    cfg = meta.get("config", {}) or {}
    cv = meta.get("cv_metrics", {}) or {}
    return ModelTargetInfo(
        target=target,
        model_id=model_dir,
        train_start=meta["train_start"],
        train_end=meta["train_end"],
        n_features=len(meta.get("feature_cols", [])),
        horizon_days=int(meta.get("horizon_days", 5)),
        learning_rate=cfg.get("learning_rate"),
        num_leaves=cfg.get("num_leaves"),
        min_data_in_leaf=cfg.get("min_data_in_leaf"),
        cv_mean_metrics=cv.get("mean", {}) or {},
        cv_std_metrics=cv.get("std", {}) or {},
        cv_fold_count=len(cv.get("fold_metrics", []) or []),
    )


def _training_parquet_stats(universe: str) -> tuple[int | None, int | None, list[date] | None]:
    """Read row count, distinct symbol count, and date range from the
    universe's training parquet on disk. Returns (None, None, None) if absent.
    """
    import os
    from packages.common.config import PROCESSED_DIR
    p = os.path.join(PROCESSED_DIR, f"training_{universe.lower()}.parquet")
    if not os.path.exists(p):
        return None, None, None
    try:
        df = pd.read_parquet(p, columns=["symbol", "bar_date"])
    except Exception:  # noqa: BLE001
        return None, None, None
    return (
        int(len(df)),
        int(df["symbol"].nunique()),
        [pd.to_datetime(df["bar_date"]).min().date(), pd.to_datetime(df["bar_date"]).max().date()],
    )


def get_model_info(duck: duckdb.DuckDBPyConnection, universe: str) -> ModelInfoResponse:
    n_members_row = duck.execute(
        "SELECT COUNT(DISTINCT symbol) FROM index_membership "
        "WHERE universe = ? AND end_date IS NULL",
        [universe],
    ).fetchone()
    n_members = int(n_members_row[0]) if n_members_row else 0

    rows, sym_n, date_range = _training_parquet_stats(universe)

    targets: list[ModelTargetInfo] = []
    for target in ("regression", "classification"):
        info = _build_target_info(universe, target)
        if info is not None:
            targets.append(info)

    return ModelInfoResponse(
        universe=universe,
        n_members=n_members,
        training_rows=rows,
        training_symbols=sym_n,
        training_date_range=date_range,
        targets=targets,
    )


# ---------------------------------------------------------------------------
# /performance/walkforward/{universe} — tax-adjusted equity curves
# ---------------------------------------------------------------------------

# Tax rates per universe. Match the existing tax_adjusted_comparison.py
# defaults (US 25% blended STCG, 15% LTCG).
_TAX_RATES = {
    "SP500": {"strategy_stcg": 0.25, "benchmark_ltcg": 0.15},
}

# Per-universe benchmark mapping.
_BENCHMARK = {
    "SP500": ("SPY", "SPY B&H", "USD"),
}


def _strategy_per_year_returns(universe: str) -> pd.DataFrame:
    """Pull the per-year return of the live paper-backtest 'default'-equivalent
    run from the predictions_log -> paper backtest replay. We re-run the engine
    once on the WF predictions to get a year-by-year equity decomposition.
    """
    import sqlite3
    from packages.paper_trading import StrategyConfig, backtest

    # Cache path: re-run the backtest only if WF preds are newer than the
    # cached paper.sqlite entry. For simplicity we always re-run if a cached
    # equity curve isn't found for the canonical analysis run_id.
    canonical_run_id = f"wf_{universe.lower()}_canon"
    paper_db = "data/processed/walkforward_10yr/analysis.sqlite"
    preds_db = "data/processed/walkforward_10yr/predictions.sqlite"

    import os
    if not os.path.exists(preds_db):
        return pd.DataFrame(columns=["year", "return"])

    need_run = True
    if os.path.exists(paper_db):
        try:
            c = sqlite3.connect(paper_db)
            n = c.execute(
                "SELECT COUNT(*) FROM paper_equity WHERE run_id=? AND snapshot_kind='close_5pm_ct'",
                (canonical_run_id,),
            ).fetchone()[0]
            c.close()
            if n > 0:
                need_run = False
        except Exception:  # noqa: BLE001
            need_run = True

    if need_run:
        cfg = StrategyConfig(
            run_id=canonical_run_id,
            universe=universe,
            predictions_sqlite_path=preds_db,
            paper_db_path=paper_db,
            commission_model="ibkr_lite",
        )
        backtest(cfg)

    c = sqlite3.connect(paper_db)
    df = pd.read_sql_query(
        "SELECT trade_date, equity FROM paper_equity "
        "WHERE run_id = ? AND snapshot_kind = 'close_5pm_ct' "
        "ORDER BY trade_date",
        c, params=[canonical_run_id],
    )
    c.close()
    if df.empty:
        return pd.DataFrame(columns=["year", "return"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    rows = []
    for year, g in df.groupby("year"):
        eqs = g["equity"].to_numpy()
        rows.append({"year": int(year), "return": float(eqs[-1] / eqs[0] - 1)})
    return pd.DataFrame(rows)


def _benchmark_per_year_returns(
    duck: duckdb.DuckDBPyConnection,
    benchmark_symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    rows = duck.execute(
        "SELECT bar_date, close FROM ohlcv_daily WHERE symbol = ? "
        "AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
        [benchmark_symbol, start, end],
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["year", "return"])
    df = pd.DataFrame(rows, columns=["trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    out = []
    for year, g in df.groupby("year"):
        closes = g["close"].to_numpy()
        out.append({"year": int(year), "return": float(closes[-1] / closes[0] - 1)})
    return pd.DataFrame(out)


def get_walkforward_taxadjusted(
    duck: duckdb.DuckDBPyConnection, universe: str
) -> WalkforwardResponse:
    if universe not in _BENCHMARK:
        raise ValueError(f"no benchmark mapping for universe={universe!r}")
    bench_sym, bench_label, currency = _BENCHMARK[universe]
    rates = _TAX_RATES[universe]
    stcg = rates["strategy_stcg"]
    ltcg = rates["benchmark_ltcg"]
    starting = 1000.0

    strat_yr = _strategy_per_year_returns(universe)
    if strat_yr.empty:
        return WalkforwardResponse(
            universe=universe,
            benchmark_symbol=bench_sym,
            benchmark_label=bench_label,
            currency=currency,
            years=[],
            summary=WalkforwardSummary(
                starting_capital=starting,
                strategy_final_pretax=starting,
                strategy_final_aftertax=starting,
                benchmark_final_pretax=starting,
                benchmark_final_aftertax=starting,
                outperformance_multiple=1.0,
                strategy_stcg_rate=stcg,
                benchmark_ltcg_rate=ltcg,
            ),
        )
    start = date(int(strat_yr["year"].min()), 1, 1)
    end = date(int(strat_yr["year"].max()), 12, 31)
    bench_yr = _benchmark_per_year_returns(duck, bench_sym, start, end)

    # Strategy: annual STCG drag.
    strat_eq = starting
    strat_eq_pretax = starting
    strat_rows: dict[int, dict] = {}
    for _, r in strat_yr.iterrows():
        year = int(r["year"])
        ret = float(r["return"])
        aftertax_ret = ret * (1 - stcg)
        strat_eq = strat_eq * (1 + aftertax_ret)
        strat_eq_pretax = strat_eq_pretax * (1 + ret)
        strat_rows[year] = {
            "strategy_return_pct": ret * 100,
            "strategy_aftertax_pct": aftertax_ret * 100,
            "strategy_equity": strat_eq,
        }

    # Benchmark: gains compound tax-deferred; LTCG applied only at terminal.
    bench_eq_pretax = starting
    bench_rows: dict[int, dict] = {}
    for _, r in bench_yr.iterrows():
        year = int(r["year"])
        ret = float(r["return"])
        bench_eq_pretax = bench_eq_pretax * (1 + ret)
        liquidation = starting + (bench_eq_pretax - starting) * (1 - ltcg)
        bench_rows[year] = {
            "benchmark_return_pct": ret * 100,
            "benchmark_equity_pretax": bench_eq_pretax,
            "benchmark_equity_aftertax": liquidation,
        }

    years_out: list[WalkforwardEquityPoint] = []
    for year in sorted(strat_rows.keys()):
        s = strat_rows[year]
        b = bench_rows.get(year)
        years_out.append(WalkforwardEquityPoint(
            year=year,
            strategy_return_pct=s["strategy_return_pct"],
            strategy_aftertax_pct=s["strategy_aftertax_pct"],
            strategy_equity=s["strategy_equity"],
            benchmark_return_pct=(b or {}).get("benchmark_return_pct", 0.0),
            benchmark_equity_pretax=(b or {}).get("benchmark_equity_pretax", starting),
            benchmark_equity_aftertax=(b or {}).get("benchmark_equity_aftertax", starting),
        ))

    final = years_out[-1] if years_out else None
    summary = WalkforwardSummary(
        starting_capital=starting,
        strategy_final_pretax=strat_eq_pretax,
        strategy_final_aftertax=final.strategy_equity if final else starting,
        benchmark_final_pretax=final.benchmark_equity_pretax if final else starting,
        benchmark_final_aftertax=final.benchmark_equity_aftertax if final else starting,
        outperformance_multiple=(
            (final.strategy_equity / final.benchmark_equity_aftertax)
            if final and final.benchmark_equity_aftertax > 0 else 1.0
        ),
        strategy_stcg_rate=stcg,
        benchmark_ltcg_rate=ltcg,
    )

    return WalkforwardResponse(
        universe=universe,
        benchmark_symbol=bench_sym,
        benchmark_label=bench_label,
        currency=currency,
        years=years_out,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# /performance/strict-wf/{universe} — LIVE strict-walk-forward progress
# ---------------------------------------------------------------------------

_STRICT_WF_PATHS = {
    "SP500": {
        "preds": "data/processed/walkforward_10yr_strict/predictions.sqlite",
        "paper": "data/processed/walkforward_10yr_strict/analysis_live.sqlite",
        "commission": "ibkr_lite",
        # Monthly retrains expected between the WF's --start and --end.
        # 148 = Jan 2014 → Apr 2026 inclusive (12 years 4 months). Was
        # 132 (Jan 2014 → Dec 2024) under the original 10-year program;
        # the extension to Apr 2026 added 16 retrains; adding May 2026
        # bumps to 149. **Update this if the WF's --end flag changes** —
        # it drives the progress bar, the X/Y display, the ETA, and the
        # running-state heuristic.
        "expected_retrains": 149,
    },
}


# Selectable WF variants for the Live WF dashboard dropdown. "baseline" is
# the canonical locked V1 run (_STRICT_WF_PATHS); extra variants point at
# experiment output dirs and are only OFFERED when their predictions.sqlite
# exists, so the dropdown never shows an empty option. Reading a variant
# is non-destructive — each has its own predictions + analysis_live sqlite,
# so the baseline (V1 locked) numbers are never touched.
_STRICT_WF_VARIANTS: dict[str, dict[str, dict[str, str]]] = {
    "SP500": {
        # No-gate control run was dropped 2026-06-20 to focus all CPU on the
        # gated run; the locked V1 baseline already represents the
        # always-newest policy, so it serves as the no-gate comparator.
        "gated": {
            "label": "Live-design backtest (gated)",
            "dir": "data/processed/wf_gatetest_gated",
            # WF run log — parsed for live per-month gate decisions while the
            # run is in progress (report.json is only written at completion).
            "log": "logs/wf_gated.log",
        },
    },
}


def _resolve_strict_wf_cfg(universe: str, variant: str) -> dict:
    """Path cfg for (universe, variant). variant='baseline' (or any unknown
    variant) → the canonical _STRICT_WF_PATHS entry. Extra variants derive
    their paths from the experiment dir."""
    base = _STRICT_WF_PATHS.get(universe)
    if base is None:
        raise ValueError(f"no strict-WF config for universe={universe!r}")
    extras = _STRICT_WF_VARIANTS.get(universe, {})
    if variant == "baseline" or variant not in extras:
        return base
    d = extras[variant]["dir"]
    return {
        "preds": f"{d}/predictions.sqlite",
        "paper": f"{d}/analysis_live.sqlite",
        "commission": base["commission"],
        "expected_retrains": base["expected_retrains"],
    }


def _strict_wf_gate_decisions(universe: str, variant: str) -> list[dict]:
    """Per-month promote/retain decisions for a gated variant.

    Sources, in order of preference: the run's report.json (structured,
    written at completion), then the live WF log (parsed for in-progress
    runs). The log overlays the report so an in-progress run still shows
    decisions for retrains past the last report. Baseline (no gate) → [].
    """
    import os, json, re
    meta = _STRICT_WF_VARIANTS.get(universe, {}).get(variant)
    if meta is None:
        return []  # baseline / unknown → no gate
    out: dict[tuple[int, int], dict] = {}
    # 1. report.json (final, structured)
    report_path = os.path.join(meta["dir"], "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, encoding="utf-8") as f:
                rep = json.load(f)
            for r in rep.get("retrains", []):
                rd = r.get("retrain_date")
                if not rd:
                    continue
                out[(int(rd[:4]), int(rd[5:7]))] = {
                    "reg": r.get("gate_reg_decision"),
                    "cls": r.get("gate_cls_decision"),
                }
        except Exception:  # noqa: BLE001
            pass
    # 2. live log (overlays — covers retrains not yet in the report)
    logp = meta.get("log")
    if logp and os.path.exists(logp):
        pat = re.compile(r"(\d{4}-\d{2}-\d{2}): GATE reg=(\w+).*?cls=(\w+)")
        try:
            with open(logp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "GATE reg=" not in line:
                        continue
                    mt = pat.search(line)
                    if mt:
                        rd, reg, cls = mt.group(1), mt.group(2), mt.group(3)
                        out[(int(rd[:4]), int(rd[5:7]))] = {"reg": reg, "cls": cls}
        except Exception:  # noqa: BLE001
            pass
    return [
        {"year": y, "month": m, "reg_decision": v["reg"], "cls_decision": v["cls"]}
        for (y, m), v in sorted(out.items())
    ]


def _available_strict_wf_variants(universe: str) -> list[dict[str, str]]:
    """{key,label} for every variant whose predictions.sqlite exists on disk.
    'baseline' is always offered (it's the locked V1 run)."""
    import os
    out = [{"key": "baseline", "label": "V1 baseline · always-newest"}]
    for key, meta in _STRICT_WF_VARIANTS.get(universe, {}).items():
        if os.path.exists(f"{meta['dir']}/predictions.sqlite"):
            out.append({"key": key, "label": meta["label"]})
    return out


# In-memory cache keyed by (universe, variant, preds_mtime).
_STRICT_WF_CACHE: dict[tuple[str, str, float], StrictWfResponse] = {}

# Process-local lock that serializes paper-engine rebuilds when multiple
# API requests hit the strict-WF endpoint concurrently. Without this,
# os.remove() on analysis_live.sqlite would race with another request
# still reading the same file → PermissionError → 500. The lock is OK
# to be coarse-grained — rebuilds are infrequent (cache hit on most
# polls) and the work itself is the bottleneck, not the lock contention.
import threading as _threading
_REPLAY_LOCK = _threading.Lock()


def _replay_engine_for_strict(
    universe: str, preds_path: str, paper_path: str, commission: str
) -> str | None:
    """Run the paper engine once on the current predictions snapshot.
    Returns the canonical run_id used or None if no predictions yet."""
    import os, sqlite3
    from packages.paper_trading import StrategyConfig, backtest
    if not os.path.exists(preds_path):
        return None
    c = sqlite3.connect(preds_path)
    n = c.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0]
    c.close()
    if n == 0:
        return None
    run_id = f"strict_wf_live_{universe.lower()}"

    # Concurrency-safe rebuild. The previous implementation did
    # ``os.remove(paper_path); backtest(...)`` which raced when two API
    # requests hit the strict-WF endpoint at the same time (Windows file
    # locks the sqlite while it's open elsewhere → PermissionError → 500).
    #
    # New strategy:
    #   1. Hold a process-local lock so concurrent rebuilds serialize
    #      inside this Python process.
    #   2. Build into a temp path, then atomically replace paper_path
    #      with ``os.replace`` (which works even if the target is open
    #      for reading on Windows, unlike os.remove).
    #   3. If the replace itself races against another process holding
    #      the file, retry a few times before giving up.
    with _REPLAY_LOCK:
        tmp_path = f"{paper_path}.rebuild-{os.getpid()}.tmp"
        # If a prior crash left a tmp around, clear it.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        cfg = StrategyConfig(
            run_id=run_id,
            universe=universe,
            predictions_sqlite_path=preds_path,
            paper_db_path=tmp_path,
            commission_model=commission,
        )
        backtest(cfg)
        # Atomic swap. os.replace allows replacing an existing file even
        # if it's open for reading on Windows in most cases. Retry on
        # the rare PermissionError where the OS hasn't released the
        # previous handle yet.
        import time as _time
        last_exc: Exception | None = None
        for _attempt in range(5):
            try:
                os.replace(tmp_path, paper_path)
                last_exc = None
                break
            except PermissionError as exc:
                last_exc = exc
                _time.sleep(0.2)
        if last_exc is not None:
            # Give up cleanly: leave the tmp file, return None.
            log.warning(
                f"strict-wf replay: could not atomically replace {paper_path} "
                f"after 5 attempts ({last_exc!r}); leaving tmp file in place"
            )
            return None
    return run_id


def _strict_wf_per_year(
    paper_path: str, run_id: str
) -> tuple[list[StrictWfYearPoint], dict[int, tuple[object, object]]]:
    """Return per-year strategy stats AND the strategy's actual
    first/last trade dates within each year.

    The second return value (year_window) is used to restrict the
    benchmark comparison to the same date span the strategy traded —
    so a partial in-progress year compares partial-year strategy to
    partial-year SPY (apples-to-apples) instead of full calendar SPY.
    """
    import math, sqlite3
    c = sqlite3.connect(paper_path)
    df = pd.read_sql_query(
        "SELECT trade_date, equity FROM paper_equity "
        "WHERE run_id=? AND snapshot_kind='close_5pm_ct' ORDER BY trade_date",
        c, params=[run_id],
    )
    c.close()
    if df.empty:
        return [], {}
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    out: list[StrictWfYearPoint] = []
    year_window: dict[int, tuple[object, object]] = {}
    for year, g in df.groupby("year"):
        eqs = g["equity"].to_numpy()
        dates_in_year = g["trade_date"].to_list()
        year_window[int(year)] = (dates_in_year[0], dates_in_year[-1])
        rets = [eqs[i] / eqs[i - 1] - 1 for i in range(1, len(eqs)) if eqs[i - 1] > 0]
        m = sum(rets) / max(len(rets), 1)
        s = (sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)) ** 0.5 if len(rets) > 1 else 0
        sharpe = m / s * math.sqrt(252) if s > 0 else None
        peak = eqs[0]
        dd = 0.0
        for e in eqs:
            peak = max(peak, e)
            dd = max(dd, (peak - e) / peak)
        out.append(StrictWfYearPoint(
            year=int(year),
            strategy_return_pct=float(eqs[-1] / eqs[0] - 1) * 100,
            sharpe=sharpe,
            max_dd_pct=dd * 100,
            n_days=int(len(g)),
        ))
    return out, year_window


def _benchmark_year_returns_pct_in_window(
    duck: duckdb.DuckDBPyConnection,
    symbol: str,
    year_window: dict[int, tuple[object, object]],
) -> dict[int, float]:
    """Per-year benchmark return restricted to the strategy's actual
    trading window for each year. For an in-progress year, this gives
    the SPY return from the strategy's first trade in the year through
    its most recent trade — apples-to-apples with the partial-year
    strategy return.
    """
    out: dict[int, float] = {}
    for year, (first_d, last_d) in year_window.items():
        rows = duck.execute(
            "SELECT bar_date, close FROM ohlcv_daily "
            "WHERE symbol = ? AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
            [symbol, first_d, last_d],
        ).fetchall()
        if len(rows) < 2:
            continue
        first_close = float(rows[0][1])
        last_close = float(rows[-1][1])
        if first_close > 0:
            out[int(year)] = (last_close / first_close - 1) * 100
    return out


def _benchmark_year_maxdd_pct_in_window(
    duck: duckdb.DuckDBPyConnection,
    symbol: str,
    year_window: dict[int, tuple[object, object]],
) -> dict[int, float]:
    """Per-year benchmark max drawdown (peak-to-trough %), restricted to
    the strategy's actual trading window for each year. Companion to
    _benchmark_year_returns_pct_in_window — uses the same SPY close
    series so it exposes the same window the user already sees in the
    SPY return column. Returned values are signed positive (e.g. 19.2
    means a 19.2% drawdown).
    """
    out: dict[int, float] = {}
    for year, (first_d, last_d) in year_window.items():
        rows = duck.execute(
            "SELECT bar_date, close FROM ohlcv_daily "
            "WHERE symbol = ? AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
            [symbol, first_d, last_d],
        ).fetchall()
        if len(rows) < 2:
            continue
        peak = 0.0
        dd = 0.0
        for _, c in rows:
            c = float(c)
            if c > peak:
                peak = c
            if peak > 0:
                dd = max(dd, (peak - c) / peak)
        out[int(year)] = dd * 100
    return out


def _vix_peak_by_year(
    year_window: dict[int, tuple[object, object]],
    vix_path: str = "data/raw/vix_daily.parquet",
) -> dict[int, float]:
    """Per-year VIX peak (intraday high) restricted to the same trading
    window as the strategy. Reads from a standalone parquet so this
    function NEVER touches market.duckdb — keeps the live walkforward
    backtest's DuckDB lock uncontended.

    Returns {} silently if the parquet is missing or unreadable; the
    column then renders as em-dash on the dashboard.
    """
    import os
    if not os.path.exists(vix_path):
        return {}
    try:
        df = pd.read_parquet(vix_path)
    except Exception:
        return {}
    if df.empty or "bar_date" not in df.columns or "high" not in df.columns:
        return {}
    # Normalize bar_date to plain date (matches year_window keys).
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
    out: dict[int, float] = {}
    for year, (first_d, last_d) in year_window.items():
        mask = (df["bar_date"] >= first_d) & (df["bar_date"] <= last_d)
        sub = df.loc[mask, "high"]
        if not sub.empty:
            out[int(year)] = float(sub.max())
    return out


def _benchmark_year_returns_pct(
    duck: duckdb.DuckDBPyConnection, symbol: str
) -> dict[int, float]:
    """Legacy: per-year benchmark return using FULL calendar year close
    prices for the symbol. Kept for any caller that still wants
    "what was SPY's full year" rather than "what did SPY do during the
    same window the strategy traded". The Live WF year table now uses
    _benchmark_year_returns_pct_in_window instead.
    """
    rows = duck.execute(
        "SELECT bar_date, close FROM ohlcv_daily WHERE symbol = ? ORDER BY bar_date",
        [symbol],
    ).fetchall()
    df = pd.DataFrame(rows, columns=["bar_date", "close"])
    if df.empty:
        return {}
    df["year"] = pd.to_datetime(df["bar_date"]).dt.year
    out: dict[int, float] = {}
    for year, g in df.groupby("year"):
        closes = g["close"].to_numpy()
        if len(closes) >= 2:
            out[int(year)] = float(closes[-1] / closes[0] - 1) * 100
    return out


def _strict_wf_progress(preds_path: str, expected_total: int) -> StrictWfProgress:
    """Derive retrains-complete + ETA by counting distinct retrain months in
    the predictions table and reading file mtime."""
    import os, sqlite3
    from datetime import datetime, timedelta, timezone
    progress = StrictWfProgress(retrains_total=expected_total)
    if not os.path.exists(preds_path):
        return progress
    mtime = datetime.fromtimestamp(os.path.getmtime(preds_path), tz=timezone.utc)
    progress.last_retrain_at_utc = mtime.isoformat()
    # Heuristic for "running": predictions.sqlite was modified in the last
    # 5 hours. Bumped from 2h after observing contended SP500 retrains
    # hitting ~2.6h between writes — the 2h window was flagging actively-
    # training runs as Idle. 5h is generous enough to cover further
    # contention growth and the occasional bigger-window retrain, but
    # still flips to Idle within ~5h of a true crash so the badge stays
    # diagnostic.
    progress.is_running = (datetime.now(timezone.utc) - mtime).total_seconds() < 18000
    c = sqlite3.connect(preds_path)
    # Each retrain produces predictions for a window with the SAME first as_of.
    # Number of distinct (year, month) of as_of approximates retrains complete.
    rows = c.execute(
        "SELECT MIN(as_of) AS first_d, MAX(as_of) AS last_d, COUNT(*) AS n "
        "FROM predictions_log"
    ).fetchone()
    if rows and rows[2] > 0:
        progress.last_retrain_date = str(rows[1])
        # Count distinct retrain months
        retrains = c.execute(
            "SELECT COUNT(DISTINCT strftime('%Y-%m', as_of)) FROM predictions_log"
        ).fetchone()[0]
        progress.retrains_complete = int(retrains)
    # Rolling-average retrain pace from the actual completion timestamps
    # of the last few retrains. Each retrain writes its window of rows
    # in one batch, so MAX(created_at) per (year, month) of as_of is a
    # good proxy for that retrain's finish time. Diffing consecutive
    # finish times gives per-retrain wall-clock duration.
    if progress.retrains_complete > 0 and progress.retrains_complete < expected_total:
        try:
            ts_rows = c.execute(
                "SELECT strftime('%Y-%m', as_of) AS ym, MAX(created_at) AS finished_at "
                "FROM predictions_log GROUP BY ym ORDER BY ym"
            ).fetchall()
        except sqlite3.Error:
            ts_rows = []
        finishes: list[datetime] = []
        for _ym, ts in ts_rows:
            if not ts:
                continue
            try:
                finishes.append(datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc))
            except ValueError:
                pass
        # Use the last 5 retrains for a rolling average — responsive to
        # recent pace changes (e.g. when contention from a second WF
        # process is removed). Fall back to 75 min if we have <2 samples.
        avg_min = 75.0
        if len(finishes) >= 2:
            sample = finishes[-6:]  # 6 timestamps → 5 deltas
            deltas_min = [
                (sample[i] - sample[i - 1]).total_seconds() / 60.0
                for i in range(1, len(sample))
            ]
            avg_min = float(sum(deltas_min) / len(deltas_min))
        progress.avg_retrain_minutes = avg_min
        remaining = expected_total - progress.retrains_complete
        eta_seconds = remaining * avg_min * 60
        progress.eta_completion_utc = (
            datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
        ).isoformat()
    c.close()
    return progress


# Tax rates used by the Live WF year table (display-only — the pre-tax
# pipeline is unchanged). 30% blanket US short-term capital gains.
_STRICT_WF_TAX_RATES = {"SP500": 0.30}

# LTCG rates applied to the benchmark cumulative return (B&H investor
# who sells once at the end of the window). 15% US federal LTCG,
# mid-bracket; Texas resident → no state income tax add-on.
_STRICT_WF_BENCH_LTCG = {"SP500": 0.15}

# Display starting capital per universe. The paper-engine simulates at
# $1,000 base regardless; we rescale the equity series on the way out
# so the UI shows realistic local-currency amounts.
_STRICT_WF_STARTING_CAPITAL = {"SP500": 1000.0}


def _strict_max_trade_date(paper_path: str, run_id: str):
    """Return the latest trade_date in paper_equity for ``run_id`` as a
    datetime.date, or None if there's no such row (or the file is missing)."""
    import os
    import sqlite3
    from datetime import date as _date
    if not os.path.exists(paper_path):
        return None
    try:
        c = sqlite3.connect(paper_path)
        row = c.execute(
            "SELECT MAX(trade_date) FROM paper_equity WHERE run_id=?", [run_id]
        ).fetchone()
        c.close()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    try:
        return _date.fromisoformat(str(row[0]))
    except ValueError:
        return None


def _strict_apply_tax(
    universe: str, paper_path: str, run_id: str, years: list[StrictWfYearPoint]
) -> None:
    """Set strategy_return_after_tax_pct on every year.

    For completed years this applies the full STCG haircut at end-of-
    year. For the IN-PROGRESS year, the year's YTD return is also
    haircut at the STCG rate — i.e. shown "as if the strategy closes
    every position right now and pays tax on the realized gain." This
    keeps the year-table after-tax + a/t-excess columns sensible for
    the in-progress year (otherwise they'd be em-dash until the year
    completes, which is uninformative).

    Loss years (and loss YTDs) pass through unchanged — no carryforward
    modeled at this level.
    """
    rate = _STRICT_WF_TAX_RATES.get(universe, 0.0)
    if rate <= 0 or not years:
        return
    for y in years:
        if y.strategy_return_pct > 0:
            y.strategy_return_after_tax_pct = y.strategy_return_pct * (1 - rate)
        else:
            y.strategy_return_after_tax_pct = y.strategy_return_pct


def _strict_build_equity_curve(
    universe: str,
    paper_path: str,
    run_id: str,
    duck: duckdb.DuckDBPyConnection,
    bench_sym: str,
) -> "StrictWfEquityCurve":
    """Build the per-day equity-curve payload: pre-tax, post-tax, and
    benchmark (B&H, indexed to the strategy's starting capital).

    Post-tax recipe (reduced-base compounding — matches what a real
    investor experiences when taxes are paid out of the portfolio):

        - Year 1 post-tax = pre-tax (no taxes yet; year not complete).
        - End of completed year Y with positive year_return r_Y:
              post_eq_end_of_Y = post_eq_start_of_Y * (1 + r_Y * (1 − rate))
          where r_Y = (pre_eoy_Y / pre_soy_Y) − 1, i.e. the pre-tax
          intra-year growth. Loss years pass through unchanged.
        - Within year Y, intra-year scaling:
              post_eq[t] = post_eq_start_of_Y * (pre_eq[t] / pre_eq_start_of_Y)

        This means the post-tax curve compounds on a REDUCED capital
        base after each tax payment, rather than tracking pre-tax minus
        cumulative tax. The two methods diverge over multiple years;
        reduced-base is the honest "what you actually have" view.

    Benchmark series:
        - Read close prices for ``bench_sym`` over [first_date, last_date].
        - Rescale so benchmark_equity[0] == pre_tax_equity[0] (= starting
          capital). LTCG isn't applied here — that's a one-shot adjustment
          shown in the summary tile, not a running drag.
    """
    import os
    import sqlite3
    from datetime import date as _date
    from services.api.schemas import StrictWfEquityCurve

    empty = StrictWfEquityCurve()
    if not os.path.exists(paper_path):
        return empty
    try:
        c = sqlite3.connect(paper_path)
        rows = c.execute(
            "SELECT trade_date, equity FROM paper_equity "
            "WHERE run_id=? AND snapshot_kind='close_5pm_ct' ORDER BY trade_date",
            [run_id],
        ).fetchall()
        c.close()
    except sqlite3.Error:
        return empty
    if not rows:
        return empty

    rate = _STRICT_WF_TAX_RATES.get(universe, 0.0)
    max_d = _date.fromisoformat(str(rows[-1][0]))

    # Per-year first / last equity (pre-tax).
    soy_eq: dict[int, float] = {}
    eoy_eq: dict[int, float] = {}
    for d_str, eq in rows:
        y = int(str(d_str)[:4])
        if y not in soy_eq:
            soy_eq[y] = float(eq)
        eoy_eq[y] = float(eq)

    # Per-year reduced-base post-tax multiple. For each completed year:
    #   yearly_post_factor = 1 + r * (1 - rate)   if r > 0 (gain → taxed)
    #                      = 1 + r                if r <= 0 (loss → no tax)
    # Compound these to get post_eq at the start of each year.
    sorted_years = sorted(soy_eq.keys())
    # post_eq_start_of_year[y] = post-tax equity at the first datapoint of year y.
    # Year 1 starts at pre-tax starting capity (= pretax[0]); subsequent years are
    # reduced by cumulative tax of completed prior years.
    post_eq_start_of_year: dict[int, float] = {}
    starting = float(rows[0][1])
    cumul_multiple_post = 1.0  # running post-tax multiplier ($1 of starting → this much)
    for i, y in enumerate(sorted_years):
        # post-tax equity at start of year y = starting * cumul_multiple_post
        post_eq_start_of_year[y] = starting * cumul_multiple_post
        # If this year is complete, apply its yearly post-tax factor for the next year.
        if max_d >= _date(y, 12, 28):
            soy = soy_eq[y]
            eoy = eoy_eq[y]
            r = (eoy / soy - 1) if soy > 0 else 0
            factor = (1 + r * (1 - rate)) if r > 0 else (1 + r)
            cumul_multiple_post *= factor

    # Walk daily rows; post_eq[t] = post_eq_start_of_year[year(t)] *
    # (pre_eq[t] / pre_eq_start_of_year[year(t)]).
    dates: list[str] = []
    pretax: list[float] = []
    posttax: list[float] = []
    for d_str, eq in rows:
        y = int(str(d_str)[:4])
        intra_year_factor = (float(eq) / soy_eq[y]) if soy_eq[y] > 0 else 1.0
        post_eq_today = post_eq_start_of_year[y] * intra_year_factor
        dates.append(str(d_str))
        pretax.append(round(float(eq), 4))
        posttax.append(round(post_eq_today, 4))

    # Benchmark equity — rescale close prices to start at pretax[0].
    bench_equity: list[float] = []
    try:
        first_d = _date.fromisoformat(dates[0])
        last_d = max_d
        bench_rows = duck.execute(
            "SELECT bar_date, close FROM ohlcv_daily WHERE symbol = ? "
            "AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
            [bench_sym, first_d, last_d],
        ).fetchall()
        if bench_rows:
            # Build a date → close map, then sample at each strategy date
            # (forward-filling between bench bars so weekend/holiday strategy
            # dates pick up the prior bench close).
            bclose: dict[str, float] = {str(d): float(p) for d, p in bench_rows}
            # First close for normalization — could be later than dates[0]
            # if benchmark has no row exactly on dates[0]; use the earliest
            # available within the window.
            bench_first = float(bench_rows[0][1])
            scale = pretax[0] / bench_first if bench_first > 0 else 0.0
            last_close = bench_first
            for d_str in dates:
                if d_str in bclose:
                    last_close = bclose[d_str]
                bench_equity.append(round(last_close * scale, 4))
    except Exception:  # noqa: BLE001 — benchmark omitted is acceptable
        bench_equity = []

    # Single-shot LTCG dot at the last chart date — what the B&H investor
    # would actually keep if they liquidated today. Sits below the
    # benchmark line whenever the benchmark has gained; equals the line
    # otherwise (no tax on losses).
    bench_post_ltcg: float | None = None
    if bench_equity:
        ltcg = _STRICT_WF_BENCH_LTCG.get(universe, 0.0)
        if ltcg > 0:
            gain = bench_equity[-1] - bench_equity[0]
            tax = gain * ltcg if gain > 0 else 0.0
            bench_post_ltcg = round(bench_equity[-1] - tax, 4)

    return StrictWfEquityCurve(
        dates=dates,
        equity_pre_tax=pretax,
        equity_post_tax=posttax,
        benchmark_equity=bench_equity,
        benchmark_post_ltcg_endpoint=bench_post_ltcg,
    )


def _strict_monthly_excess(
    paper_path: str,
    run_id: str,
    duck: duckdb.DuckDBPyConnection,
    bench_sym: str,
) -> list["StrictWfMonthlyExcessCell"]:
    """Per-(year,month) strategy/benchmark/excess returns for the heatmap.

    For each month, return = (last_close_of_month / first_close_of_month) - 1.
    This is the conventional "what was this month's return" view; not the
    prior-month-anchored compound used by the cum tile. The heatmap cells
    therefore don't compound to match the cum number — but each cell is a
    standalone, intuitive monthly performance figure.
    """
    import os
    import sqlite3
    from services.api.schemas import StrictWfMonthlyExcessCell

    if not os.path.exists(paper_path):
        return []
    try:
        c = sqlite3.connect(paper_path)
        rows = c.execute(
            "SELECT trade_date, equity FROM paper_equity "
            "WHERE run_id=? AND snapshot_kind='close_5pm_ct' ORDER BY trade_date",
            [run_id],
        ).fetchall()
        c.close()
    except sqlite3.Error:
        return []
    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["trade_date", "equity"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ym"] = df["trade_date"].dt.to_period("M")
    strat = df.groupby("ym").agg(
        first=("equity", "first"), last=("equity", "last")
    )
    strat["ret"] = (strat["last"] / strat["first"] - 1) * 100

    # Benchmark over the same range.
    first_d = df["trade_date"].min().date()
    last_d = df["trade_date"].max().date()
    bench_rows = duck.execute(
        "SELECT bar_date, close FROM ohlcv_daily WHERE symbol = ? "
        "AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
        [bench_sym, first_d, last_d],
    ).fetchall()
    bench_by_ym: dict = {}
    if bench_rows:
        bdf = pd.DataFrame(bench_rows, columns=["bar_date", "close"])
        bdf["bar_date"] = pd.to_datetime(bdf["bar_date"])
        bdf["ym"] = bdf["bar_date"].dt.to_period("M")
        bg = bdf.groupby("ym").agg(first=("close", "first"), last=("close", "last"))
        bg["ret"] = (bg["last"] / bg["first"] - 1) * 100
        bench_by_ym = bg["ret"].to_dict()

    out: list[StrictWfMonthlyExcessCell] = []
    for ym, row in strat.iterrows():
        s_ret = float(row["ret"])
        b_ret = bench_by_ym.get(ym)
        b_ret_f = float(b_ret) if b_ret is not None else None
        excess = s_ret - b_ret_f if b_ret_f is not None else None
        out.append(StrictWfMonthlyExcessCell(
            year=int(ym.year),
            month=int(ym.month),
            strategy_pct=s_ret,
            benchmark_pct=b_ret_f,
            excess_pct=excess,
        ))
    return out


def _strict_cum_after_tax_multiple(
    universe: str, max_d, years: list[StrictWfYearPoint]
) -> float | None:
    """Compound after-tax equity multiple by walking years left→right.

    Per-year factor:
        - in-progress year, or losing year → (1 + r)
        - completed year with positive r   → (1 + r·(1 − tax_rate))

    Equivalent to deducting the prior year's tax bill from capital on
    Jan 1, then compounding the next year on the reduced base. Returns
    None when no tax rate applies or there's no year data.
    """
    from datetime import date as _date

    rate = _STRICT_WF_TAX_RATES.get(universe, 0.0)
    if rate <= 0 or not years or max_d is None:
        return None
    m = 1.0
    for y in years:
        r = y.strategy_return_pct / 100.0
        year_complete = max_d >= _date(y.year, 12, 28)
        if year_complete and r > 0:
            m *= 1 + r * (1 - rate)
        else:
            m *= 1 + r
    return m


def get_strict_wf_month_detail(
    duck: duckdb.DuckDBPyConnection, universe: str, year: int, month: int
) -> StrictWfMonthDetail:
    """Drill-down for a single heatmap cell. Pure quantitative derivation
    from paper_equity + paper_positions + ohlcv_daily.

    - Daily strategy returns from paper_equity close_5pm_ct snapshots
    - Daily benchmark returns from ohlcv_daily close prices
    - Best / worst 3 days by EXCESS (strategy − benchmark) pct
    - Top 10 holdings by average position-weight over the month
    """
    import math
    import os
    import sqlite3
    from datetime import date as _date
    from calendar import monthrange

    cfg = _STRICT_WF_PATHS.get(universe)
    if cfg is None:
        raise ValueError(f"no strict-WF config for universe={universe!r}")

    bench_sym, _bench_label, _currency = _BENCHMARK.get(
        universe, ("SPY", "SPY B&H", "USD")
    )
    paper_path = cfg["paper"]
    run_id = f"strict_wf_live_{universe.lower()}"

    # Month bounds.
    last_day_of_month = monthrange(year, month)[1]
    start_d = _date(year, month, 1)
    end_d = _date(year, month, last_day_of_month)

    detail = StrictWfMonthDetail(universe=universe, year=year, month=month)

    if not os.path.exists(paper_path):
        return detail

    # Strategy daily equity over the month.
    try:
        c = sqlite3.connect(paper_path)
        rows = c.execute(
            "SELECT trade_date, equity FROM paper_equity "
            "WHERE run_id=? AND snapshot_kind='close_5pm_ct' "
            "AND trade_date >= ? AND trade_date <= ? "
            "ORDER BY trade_date",
            [run_id, start_d.isoformat(), end_d.isoformat()],
        ).fetchall()
        c.close()
    except sqlite3.Error:
        return detail
    if not rows:
        return detail

    dates = [str(r[0]) for r in rows]
    equity = [float(r[1]) for r in rows]
    # Daily strategy returns: i = 0 has no prior, so first daily pct uses
    # NaN — but we still emit the row with strategy_pct=None for chart
    # continuity. Subsequent days are (eq[i]/eq[i-1] - 1) * 100.
    strat_daily: list[float | None] = [None]
    for i in range(1, len(equity)):
        if equity[i - 1] > 0:
            strat_daily.append((equity[i] / equity[i - 1] - 1) * 100)
        else:
            strat_daily.append(None)

    # Benchmark daily closes over the same window.
    bench_rows = duck.execute(
        "SELECT bar_date, close FROM ohlcv_daily "
        "WHERE symbol = ? AND bar_date >= ? AND bar_date <= ? ORDER BY bar_date",
        [bench_sym, start_d, end_d],
    ).fetchall()
    bench_by_date: dict[str, float] = {
        str(d): float(c) for d, c in bench_rows
    }
    bench_dates_sorted = sorted(bench_by_date)
    bench_daily_pct_by_date: dict[str, float] = {}
    for i in range(1, len(bench_dates_sorted)):
        prev = bench_by_date[bench_dates_sorted[i - 1]]
        cur = bench_by_date[bench_dates_sorted[i]]
        if prev > 0:
            bench_daily_pct_by_date[bench_dates_sorted[i]] = (cur / prev - 1) * 100

    daily_points: list[StrictWfDailyPoint] = []
    for i, d_str in enumerate(dates):
        s = strat_daily[i]
        b = bench_daily_pct_by_date.get(d_str)
        e = (s - b) if (s is not None and b is not None) else None
        daily_points.append(StrictWfDailyPoint(
            date=d_str,
            strategy_pct=s,
            benchmark_pct=b,
            excess_pct=e,
        ))
    detail.daily = daily_points
    detail.n_days = len(dates)

    # Month headline: first → last close.
    first_eq = equity[0]
    last_eq = equity[-1]
    if first_eq > 0:
        detail.strategy_pct = (last_eq / first_eq - 1) * 100
    # Benchmark month headline: first → last close in the month.
    if bench_dates_sorted:
        b_first = bench_by_date[bench_dates_sorted[0]]
        b_last = bench_by_date[bench_dates_sorted[-1]]
        if b_first > 0:
            detail.benchmark_pct = (b_last / b_first - 1) * 100
    if detail.strategy_pct is not None and detail.benchmark_pct is not None:
        detail.excess_pct = detail.strategy_pct - detail.benchmark_pct

    # Risk stats from intra-month daily returns.
    rets = [r for r in strat_daily[1:] if r is not None]
    if len(rets) > 1:
        mu = sum(rets) / len(rets)
        sigma = (sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
        if sigma > 0:
            detail.sharpe = (mu / sigma) * math.sqrt(252)
            detail.vol_pct = sigma * math.sqrt(252)  # annualized stdev in pct points
    peak = equity[0]
    dd_max = 0.0
    for e_v in equity:
        peak = max(peak, e_v)
        if peak > 0:
            dd_max = max(dd_max, (peak - e_v) / peak)
    detail.max_dd_pct = dd_max * 100

    # Best / worst 3 days by EXCESS pct.
    days_with_excess = [d for d in daily_points if d.excess_pct is not None]
    days_with_excess.sort(key=lambda d: d.excess_pct, reverse=True)
    detail.best_days = days_with_excess[:3]
    detail.worst_days = days_with_excess[-3:][::-1]  # most negative first

    # Top holdings: aggregate paper_positions over the month.
    try:
        c = sqlite3.connect(paper_path)
        pos_rows = c.execute(
            "SELECT trade_date, symbol, qty, entry_price "
            "FROM paper_positions "
            "WHERE run_id=? AND trade_date >= ? AND trade_date <= ?",
            [run_id, start_d.isoformat(), end_d.isoformat()],
        ).fetchall()
        c.close()
    except sqlite3.Error:
        pos_rows = []
    # Holdings aggregation. The strategy uses a rolling-tranche rebalance:
    # one new lot per trading day, each lot held for ~5 days. So multiple
    # lots can co-exist for the same symbol on a given day. To compute
    # "average weight in portfolio", we must:
    #   1. Sum all lot weights per (symbol, date) → that day's TOTAL
    #      symbol weight.
    #   2. Average those daily totals over the days the symbol was held.
    # The previous version averaged per-LOT weight, which under-counted
    # by ~3-5x when overlapping tranches existed.
    if pos_rows and dates:
        eq_by_date = dict(zip(dates, equity))
        from collections import defaultdict
        # (symbol, date) → sum of lot weights on that day
        sym_date_weight: dict[tuple[str, str], float] = defaultdict(float)
        for trade_date, symbol, qty, entry_price in pos_rows:
            td_str = str(trade_date)
            if td_str not in eq_by_date or eq_by_date[td_str] <= 0:
                continue
            mv = float(qty) * float(entry_price)
            sym_date_weight[(symbol, td_str)] += mv / eq_by_date[td_str]
        # symbol → list of daily total weights (length = days_held)
        sym_daily_weights: dict[str, list[float]] = defaultdict(list)
        for (sym, _td), w in sym_date_weight.items():
            sym_daily_weights[sym].append(w)
        holdings: list[StrictWfHolding] = []
        for sym, weights in sym_daily_weights.items():
            if not weights:
                continue
            avg_w = sum(weights) / len(weights)
            holdings.append(StrictWfHolding(
                symbol=sym,
                days_held=len(weights),
                avg_weight_pct=avg_w * 100,
            ))
        # Sort by days_held desc, breaking ties by avg_weight desc.
        holdings.sort(key=lambda h: (h.days_held, h.avg_weight_pct), reverse=True)
        detail.top_holdings = holdings[:10]

    return detail


def get_strict_wf_status(
    duck: duckdb.DuckDBPyConnection, universe: str, variant: str = "baseline"
) -> StrictWfResponse:
    """Live snapshot of a strict-WF run. Cached in memory by predictions.sqlite
    mtime so repeated requests during the same retrain are O(1).

    ``variant`` selects WHICH WF run to read (default 'baseline' = the locked
    V1 run). Experiment variants read from their own sqlite files, so picking
    one never mutates the baseline.
    """
    import os
    cfg = _resolve_strict_wf_cfg(universe, variant)
    available = _available_strict_wf_variants(universe)
    gate_decs = _strict_wf_gate_decisions(universe, variant)

    bench_sym, bench_label, currency = _BENCHMARK.get(
        universe, ("SPY", "SPY B&H", "USD")
    )

    preds_path = cfg["preds"]
    paper_path = cfg["paper"]
    expected = int(cfg["expected_retrains"])

    progress = _strict_wf_progress(preds_path, expected)

    if not os.path.exists(preds_path):
        return StrictWfResponse(
            universe=universe,
            benchmark_symbol=bench_sym,
            benchmark_label=bench_label,
            currency=currency,
            progress=progress,
            years=[],
            summary=StrictWfSummary(),
            variant=variant,
            available_variants=available,
            gate_decisions=gate_decs,
        )

    # Cache by (universe, variant, mtime)
    mtime = os.path.getmtime(preds_path)
    cache_key = (universe, variant, mtime)
    cached = _STRICT_WF_CACHE.get(cache_key)
    if cached is not None:
        # Refresh just the progress timestamp (so ETA stays current)
        cached.progress = progress
        return cached

    run_id = _replay_engine_for_strict(universe, preds_path, paper_path, cfg["commission"])
    if run_id is None:
        return StrictWfResponse(
            universe=universe,
            benchmark_symbol=bench_sym,
            benchmark_label=bench_label,
            currency=currency,
            progress=progress,
            years=[],
            summary=StrictWfSummary(),
            variant=variant,
            available_variants=available,
            gate_decisions=gate_decs,
        )

    years, year_window = _strict_wf_per_year(paper_path, run_id)
    # Benchmark column uses the strategy's actual trading window for
    # each year (not the full calendar year). For partial in-progress
    # years this means we compare partial-year strategy to partial-year
    # SPY — apples-to-apples, no inflated full-year SPY comparator.
    bench_by_year = _benchmark_year_returns_pct_in_window(
        duck, bench_sym, year_window
    )
    # Companion metrics for the "stress context" columns on the years
    # table: SPY's own intra-window drawdown and VIX peak. Both use the
    # same year_window as the SPY return so all four columns share an
    # apples-to-apples date span. VIX reads from a standalone parquet —
    # zero contact with market.duckdb (the live WF holds that lock).
    bench_dd_by_year = _benchmark_year_maxdd_pct_in_window(
        duck, bench_sym, year_window
    )
    vix_peak_by_year = _vix_peak_by_year(year_window)
    for y in years:
        b = bench_by_year.get(y.year)
        y.benchmark_return_pct = b
        if b is not None:
            y.excess_pct = y.strategy_return_pct - b
        y.benchmark_max_dd_pct = bench_dd_by_year.get(y.year)
        y.vix_peak = vix_peak_by_year.get(y.year)

    # Tax-adjusted strategy return for EVERY year (in-progress years
    # are taxed as if the strategy closes today). Frontend uses this
    # for the After Tax column AND the Excess (A/T) = after_tax − bench
    # column.
    _strict_apply_tax(universe, paper_path, run_id, years)

    n_years = max(len(years), 1)

    # Build the equity curve first so we can derive the cum summary from
    # its endpoints. This guarantees the dashboard's summary tiles match
    # the chart exactly — same end-to-end equity, same WF window for the
    # benchmark, no "year compound vs end-to-end" drift, no "full
    # calendar year SPY vs partial WF window" mismatch.
    equity_curve = _strict_build_equity_curve(
        universe, paper_path, run_id, duck, bench_sym
    )
    bench_ltcg = _STRICT_WF_BENCH_LTCG.get(universe, 0.0)

    # Strategy pre-tax cum = end-to-end equity ratio. Falls back to the
    # old yearly-compound math if the equity curve is empty.
    if equity_curve.equity_pre_tax and equity_curve.equity_pre_tax[0] > 0:
        strat_cum = (
            equity_curve.equity_pre_tax[-1] / equity_curve.equity_pre_tax[0]
        )
    else:
        strat_cum = 1.0
        for y in years:
            strat_cum *= (1 + y.strategy_return_pct / 100)

    # Strategy after-tax cum = end-to-end ratio of the reduced-base
    # post-tax line. Hides itself (None) if no year has completed yet —
    # without a completed year there's no tax to apply, and showing the
    # same number as pre-tax is just noise.
    has_completed_year = any(
        y.strategy_return_after_tax_pct is not None for y in years
    )
    strat_after_tax_pct: float | None = None
    strat_after_tax_multiple: float | None = None
    strat_after_tax_annualized: float | None = None
    if has_completed_year and equity_curve.equity_post_tax and \
            equity_curve.equity_post_tax[0] > 0:
        strat_after_tax_multiple = (
            equity_curve.equity_post_tax[-1] / equity_curve.equity_post_tax[0]
        )
        strat_after_tax_pct = (strat_after_tax_multiple - 1) * 100
        strat_after_tax_annualized = (
            strat_after_tax_multiple ** (1 / n_years) - 1
        ) * 100

    # Benchmark cum = end-to-end ratio of bench_equity over the WF window.
    # The benchmark series in equity_curve is already restricted to
    # [first_date, last_date] of the WF — so this answers "what did SPY
    # do over the same date span the strategy actually traded".
    if equity_curve.benchmark_equity and equity_curve.benchmark_equity[0] > 0:
        bench_cum = (
            equity_curve.benchmark_equity[-1] / equity_curve.benchmark_equity[0]
        )
    else:
        bench_cum = 1.0
    bench_cum_pct_pre = (bench_cum - 1) * 100
    # SPY post-LTCG: one-shot liquidation at end of WF window. Same
    # convention as the post-LTCG dot on the chart.
    bench_cum_pct_post: float | None = None
    if bench_ltcg > 0:
        if bench_cum_pct_pre > 0:
            bench_cum_pct_post = bench_cum_pct_pre * (1 - bench_ltcg)
        else:
            bench_cum_pct_post = bench_cum_pct_pre

    summary = StrictWfSummary(
        starting_capital=1000.0,
        strategy_cum_return_pct=(strat_cum - 1) * 100,
        strategy_cum_return_after_tax_pct=strat_after_tax_pct,
        strategy_annualized_after_tax_pct=strat_after_tax_annualized,
        strategy_multiple_after_tax=strat_after_tax_multiple,
        benchmark_cum_return_pct=bench_cum_pct_pre,
        benchmark_cum_return_after_tax_pct=bench_cum_pct_post,
        strategy_annualized_pct=(strat_cum ** (1 / n_years) - 1) * 100,
        benchmark_annualized_pct=(bench_cum ** (1 / n_years) - 1) * 100,
        n_years=float(n_years),
        strategy_multiple=strat_cum,
    )
    monthly_excess = _strict_monthly_excess(paper_path, run_id, duck, bench_sym)

    # Display-only rescale to the configured starting capital. Done at
    # the very end so internal compounding math stays on the $1,000 base
    # the paper engine simulates with.
    target_capital = _STRICT_WF_STARTING_CAPITAL.get(universe, 1000.0)
    if target_capital != 1000.0 and equity_curve.equity_pre_tax:
        scale = target_capital / 1000.0
        summary.starting_capital = target_capital
        equity_curve.equity_pre_tax = [
            round(v * scale, 2) for v in equity_curve.equity_pre_tax
        ]
        equity_curve.equity_post_tax = [
            round(v * scale, 2) for v in equity_curve.equity_post_tax
        ]
        equity_curve.benchmark_equity = [
            round(v * scale, 2) for v in equity_curve.benchmark_equity
        ]
        if equity_curve.benchmark_post_ltcg_endpoint is not None:
            equity_curve.benchmark_post_ltcg_endpoint = round(
                equity_curve.benchmark_post_ltcg_endpoint * scale, 2
            )

    resp = StrictWfResponse(
        universe=universe,
        benchmark_symbol=bench_sym,
        benchmark_label=bench_label,
        currency=currency,
        progress=progress,
        years=years,
        summary=summary,
        equity_curve=equity_curve,
        monthly_excess=monthly_excess,
        variant=variant,
        available_variants=available,
        gate_decisions=gate_decs,
    )
    # Cap cache size at a few entries to avoid leak across many mtime changes.
    if len(_STRICT_WF_CACHE) > 8:
        _STRICT_WF_CACHE.clear()
    _STRICT_WF_CACHE[cache_key] = resp
    return resp
