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
    OHLCVPoint,
    OHLCVResponse,
    PerformanceResponse,
    StockHistoryResponse,
    TopPick,
    TopPicksResponse,
    UniverseInfo,
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
