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

    order = "DESC" if direction == "long" else "ASC"
    rows = sqlite_conn.execute(
        f"""
        SELECT symbol, predicted_return, predicted_quintile,
               top_quintile_proba, bottom_quintile_proba,
               model_version_regression, model_version_classification
        FROM predictions_log
        WHERE universe = ? AND as_of = ?
        ORDER BY predicted_return {order}
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
# defaults (US 25% blended STCG, 15% LTCG). India numbers reflect the
# post-July-2024 budget: 20% STCG, 12.5% LTCG.
_TAX_RATES = {
    "SP500": {"strategy_stcg": 0.25, "benchmark_ltcg": 0.15},
    "NIFTY100": {"strategy_stcg": 0.20, "benchmark_ltcg": 0.125},
}

# Per-universe benchmark mapping.
_BENCHMARK = {
    "SP500": ("SPY", "SPY B&H", "USD"),
    "NIFTY100": ("NIFTYBEES", "NIFTY 50 ETF (NIFTYBEES) B&H", "INR"),
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
    paper_db = "data/processed/india_phase_a/analysis.sqlite" if universe == "NIFTY100" \
        else "data/processed/walkforward_10yr/analysis.sqlite"
    preds_db = "data/processed/india_phase_a/walkforward/predictions.sqlite" if universe == "NIFTY100" \
        else "data/processed/walkforward_10yr/predictions.sqlite"

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
            commission_model="india_zerodha" if universe == "NIFTY100" else "ibkr_lite",
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
        "expected_retrains": 132,  # 11 years × 12 monthly retrains
    },
    "NIFTY100": {
        "preds": "data/processed/wf_nifty100_strict/predictions.sqlite",
        "paper": "data/processed/wf_nifty100_strict/analysis_live.sqlite",
        "commission": "india_zerodha",
        # 124 = monthly retrains over the strict NIFTY100 window
        # 2016-01-01 → 2026-04-30 (10y 4m). SP500 is 132 because that
        # WF spans 2014-01-01 → 2024-12-31 (11y).
        "expected_retrains": 124,
    },
}


# In-memory cache keyed by (universe, preds_mtime).
_STRICT_WF_CACHE: dict[tuple[str, float], StrictWfResponse] = {}


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
    # Always rebuild — the calling layer caches by mtime so this is cheap on cache hit.
    if os.path.exists(paper_path):
        os.remove(paper_path)
    cfg = StrategyConfig(
        run_id=run_id,
        universe=universe,
        predictions_sqlite_path=preds_path,
        paper_db_path=paper_path,
        commission_model=commission,
    )
    backtest(cfg)
    return run_id


def _strict_wf_per_year(paper_path: str, run_id: str) -> list[StrictWfYearPoint]:
    import math, sqlite3
    c = sqlite3.connect(paper_path)
    df = pd.read_sql_query(
        "SELECT trade_date, equity FROM paper_equity "
        "WHERE run_id=? AND snapshot_kind='close_5pm_ct' ORDER BY trade_date",
        c, params=[run_id],
    )
    c.close()
    if df.empty:
        return []
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    out: list[StrictWfYearPoint] = []
    for year, g in df.groupby("year"):
        eqs = g["equity"].to_numpy()
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
    return out


def _benchmark_year_returns_pct(
    duck: duckdb.DuckDBPyConnection, symbol: str
) -> dict[int, float]:
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
    # Heuristic for "running": modified in last 2 hours
    progress.is_running = (datetime.now(timezone.utc) - mtime).total_seconds() < 7200
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
    c.close()
    # Compute avg retrain pace from mtime windows. Cheapest approximation:
    # if more than 1 retrain done, divide elapsed since first retrain by N-1.
    # We don't have first-retrain time, so just use a fixed ~75 min default.
    if progress.retrains_complete > 0 and progress.retrains_complete < expected_total:
        avg_min = 75.0  # heuristic; matches observed ~73-78 min/retrain
        progress.avg_retrain_minutes = avg_min
        remaining = expected_total - progress.retrains_complete
        eta_seconds = remaining * avg_min * 60
        progress.eta_completion_utc = (
            datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
        ).isoformat()
    return progress


# Tax rates used by the Live WF year table (display-only — the pre-tax
# pipeline is unchanged). 30% blanket US short-term, 15% India short-term.
_STRICT_WF_TAX_RATES = {"SP500": 0.30, "NIFTY100": 0.15}

# LTCG rates applied to the benchmark cumulative return (B&H investor
# who sells once at the end of the window).
#   SP500    → 15% (US federal LTCG, mid-bracket; Texas resident → no
#               state income tax add-on).
#   NIFTY100 → 12.5% (India LTCG on listed equity).
_STRICT_WF_BENCH_LTCG = {"SP500": 0.15, "NIFTY100": 0.125}


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
    """Set strategy_return_after_tax_pct on years whose calendar year has
    fully elapsed in the WF data; leave it None for the in-progress year."""
    from datetime import date as _date

    rate = _STRICT_WF_TAX_RATES.get(universe, 0.0)
    if rate <= 0 or not years:
        return
    max_d = _strict_max_trade_date(paper_path, run_id)
    if max_d is None:
        return
    for y in years:
        # A year counts as "complete" once the last trade date in the WF
        # has reached late December of that year (Dec 28 catches all
        # calendars whose last trading day falls Dec 28-31).
        if max_d >= _date(y.year, 12, 28):
            if y.strategy_return_pct > 0:
                y.strategy_return_after_tax_pct = y.strategy_return_pct * (1 - rate)
            else:
                # Loss: pass through (no carryforward modeled here).
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

    Post-tax recipe:
        - For each completed calendar year, tax_y = max(0, eoy_eq - soy_eq) * rate
          where soy_eq/eoy_eq are pre-tax (paper-engine) equities.
        - post_tax(t) = pre_tax(t) − sum(tax_y for completed years y < t.year).
        - Losing years contribute zero. The in-progress year is never taxed.

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

    # Tax owed for each COMPLETED year (becomes effective on Jan 1 of y+1).
    year_tax: dict[int, float] = {}
    for y, first in soy_eq.items():
        if max_d >= _date(y, 12, 28):
            gain = eoy_eq[y] - first
            year_tax[y] = gain * rate if gain > 0 else 0.0

    # Walk rows; cumul_tax at date d = sum of taxes for all completed
    # years strictly before d.year.
    dates: list[str] = []
    pretax: list[float] = []
    posttax: list[float] = []
    # Precompute the running cumul keyed by year for O(1) lookup.
    sorted_completed = sorted(year_tax)
    running = 0.0
    cumul_by_starting_year: dict[int, float] = {}
    for y in sorted_completed:
        # Tax for year y takes effect at Jan 1 of year y+1.
        cumul_by_starting_year[y + 1] = running + year_tax[y]
        running += year_tax[y]
    last_year_seen = -1
    cur_cumul = 0.0
    for d_str, eq in rows:
        y = int(str(d_str)[:4])
        if y != last_year_seen:
            # Update cumul if we crossed into a new year that has prior-year tax accrued.
            # Largest completed-year-boundary <= y.
            candidates = [yy for yy in cumul_by_starting_year if yy <= y]
            cur_cumul = max((cumul_by_starting_year[yy] for yy in candidates), default=0.0)
            last_year_seen = y
        dates.append(str(d_str))
        pretax.append(round(float(eq), 4))
        posttax.append(round(float(eq) - cur_cumul, 4))

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

    return StrictWfEquityCurve(
        dates=dates,
        equity_pre_tax=pretax,
        equity_post_tax=posttax,
        benchmark_equity=bench_equity,
    )


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


def get_strict_wf_status(
    duck: duckdb.DuckDBPyConnection, universe: str
) -> StrictWfResponse:
    """Live snapshot of a strict-WF run. Cached in memory by predictions.sqlite
    mtime so repeated requests during the same retrain are O(1)."""
    import os
    cfg = _STRICT_WF_PATHS.get(universe)
    if cfg is None:
        raise ValueError(f"no strict-WF config for universe={universe!r}")

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
        )

    # Cache by mtime
    mtime = os.path.getmtime(preds_path)
    cache_key = (universe, mtime)
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
        )

    years = _strict_wf_per_year(paper_path, run_id)
    bench_by_year = _benchmark_year_returns_pct(duck, bench_sym)
    for y in years:
        b = bench_by_year.get(y.year)
        y.benchmark_return_pct = b
        if b is not None:
            y.excess_pct = y.strategy_return_pct - b

    # Tax-adjusted strategy return — populated only when the calendar
    # year has fully elapsed in the data (we look at the latest paper
    # equity row). Pipeline stays pre-tax; this is a display-only column.
    # Rates: 30% US short-term, 15% India short-term. Losses pass through
    # unchanged (no carryforward modeled at this level).
    _strict_apply_tax(universe, paper_path, run_id, years)

    # Cumulative summary
    strat_cum = 1.0
    bench_cum = 1.0
    for y in years:
        strat_cum *= (1 + y.strategy_return_pct / 100)
        if y.benchmark_return_pct is not None:
            bench_cum *= (1 + y.benchmark_return_pct / 100)
    n_years = max(len(years), 1)
    # After-tax cumulative — reuses the same per-year strategy_return_pct
    # but with capital reduced by each completed year's tax. See
    # _strict_cum_after_tax_multiple for the per-year factor.
    max_d_at = _strict_max_trade_date(paper_path, run_id)
    m_after_tax = _strict_cum_after_tax_multiple(universe, max_d_at, years)
    # Benchmark cum after LTCG — one-shot liquidation at the window end.
    # Applied only when bench_cum is positive (no tax on losses).
    bench_ltcg = _STRICT_WF_BENCH_LTCG.get(universe, 0.0)
    bench_cum_pct_pre = (bench_cum - 1) * 100
    bench_cum_pct_post: float | None = None
    if bench_ltcg > 0:
        if bench_cum_pct_pre > 0:
            bench_cum_pct_post = bench_cum_pct_pre * (1 - bench_ltcg)
        else:
            bench_cum_pct_post = bench_cum_pct_pre
    summary = StrictWfSummary(
        starting_capital=1000.0,
        strategy_cum_return_pct=(strat_cum - 1) * 100,
        strategy_cum_return_after_tax_pct=(
            (m_after_tax - 1) * 100 if m_after_tax is not None else None
        ),
        strategy_annualized_after_tax_pct=(
            (m_after_tax ** (1 / n_years) - 1) * 100 if m_after_tax is not None else None
        ),
        strategy_multiple_after_tax=m_after_tax,
        benchmark_cum_return_pct=bench_cum_pct_pre,
        benchmark_cum_return_after_tax_pct=bench_cum_pct_post,
        strategy_annualized_pct=(strat_cum ** (1 / n_years) - 1) * 100,
        benchmark_annualized_pct=(bench_cum ** (1 / n_years) - 1) * 100,
        n_years=float(n_years),
        strategy_multiple=strat_cum,
    )

    equity_curve = _strict_build_equity_curve(
        universe, paper_path, run_id, duck, bench_sym
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
    )
    # Cap cache size at a few entries to avoid leak across many mtime changes.
    if len(_STRICT_WF_CACHE) > 8:
        _STRICT_WF_CACHE.clear()
    _STRICT_WF_CACHE[cache_key] = resp
    return resp
