"""Predictions log: insert, settle (when horizon closes), and summarize.

Settlement rules:
    - A prediction made on date T with horizon=H settles at T + H trading
      bars (NOT calendar days). We look up the symbol's OHLCV and take the
      Hth bar AFTER T. Calendar weekends + holidays are skipped naturally.
    - We never settle predictions whose horizon hasn't closed yet —
      settle_predictions takes a target date `as_of` and only touches rows
      where the realized close is actually available in the data store.
    - Realized close is sourced from ohlcv_daily (the same source used by
      training labels). yfinance is intentionally NOT used here so settled
      returns and training labels share a price source.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import duckdb
import pandas as pd

from packages.common.config import settings
from packages.common.logging import log
from packages.inference.db import get_sqlite_conn
from packages.ingestion.storage import get_conn as get_duck_conn

_PRED_COLS = (
    "universe",
    "symbol",
    "as_of",
    "horizon_days",
    "predicted_return",
    "predicted_quintile",
    "top_quintile_proba",
    "bottom_quintile_proba",
    "model_version_regression",
    "model_version_classification",
)


# ---------------------------------------------------------------------------
# Insert / upsert
# ---------------------------------------------------------------------------


def log_predictions(
    predictions: pd.DataFrame,
    *,
    sqlite_path: str | None = None,
) -> int:
    """Idempotent upsert into predictions_log.

    Re-running on the same as_of with newer models updates the prediction
    columns but preserves any already-realized fields (realized_return,
    realized_quintile, settled_at). Returns rows touched.
    """
    if predictions.empty:
        return 0

    missing = [c for c in _PRED_COLS if c not in predictions.columns]
    if missing:
        raise ValueError(f"log_predictions: predictions missing columns {missing}")

    rows = [
        tuple(
            (r["as_of"].isoformat() if isinstance(r["as_of"], date) else str(r["as_of"]))
            if c == "as_of"
            else (None if pd.isna(r[c]) else r[c])
            for c in _PRED_COLS
        )
        for _, r in predictions.iterrows()
    ]

    sql = f"""
        INSERT INTO predictions_log ({", ".join(_PRED_COLS)})
        VALUES ({", ".join("?" for _ in _PRED_COLS)})
        ON CONFLICT (universe, symbol, as_of, horizon_days) DO UPDATE SET
            predicted_return = excluded.predicted_return,
            predicted_quintile = excluded.predicted_quintile,
            top_quintile_proba = excluded.top_quintile_proba,
            bottom_quintile_proba = excluded.bottom_quintile_proba,
            model_version_regression = excluded.model_version_regression,
            model_version_classification = excluded.model_version_classification,
            -- Refresh created_at on upsert so MAX(created_at) reflects the
            -- LATEST write (08:35 CT morning preliminary OR 17:00 CT post-
            -- close final). Without this the timestamp is stuck at first
            -- write and you can't tell from the DB whether you're looking
            -- at the preliminary or final batch.
            created_at = excluded.created_at
    """

    with get_sqlite_conn(sqlite_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def _close_at_offset(
    symbol: str,
    pred_date: date,
    n_trading_bars: int,
    *,
    duckdb_path: str | None = None,
    duck_conn: "duckdb.DuckDBPyConnection | None" = None,
) -> tuple[float, float, date] | None:
    """Return (close_at_pred, close_at_pred_plus_horizon, realized_date) or None
    if not enough future bars exist yet.

    ``n_trading_bars`` counts BARS in the symbol's OHLCV series, not calendar
    days — same convention as the training labels.

    Pass ``duck_conn`` to reuse one connection across many calls — important
    when settling thousands of predictions in one go, since each fresh
    ``duckdb.connect`` acquires a Windows file lock that races with the API
    server's read-only opens. Calls that don't pass a connection still work
    (opens a read-only one internally).
    """
    # Pull a generous window ahead — we need pred_date itself plus N bars.
    window_end = pred_date + timedelta(days=n_trading_bars * 2 + 21)
    if duck_conn is not None:
        df = duck_conn.execute(
            """
            SELECT bar_date, close
            FROM ohlcv_daily
            WHERE symbol = ? AND bar_date >= ? AND bar_date <= ?
            ORDER BY bar_date
            """,
            [symbol, pred_date, window_end],
        ).df()
    else:
        # Fallback path — opens a read-only connection (we only SELECT).
        # Callers settling many predictions should pass an explicit conn.
        import duckdb as _duckdb
        db_path = duckdb_path or settings.duckdb_path
        c = _duckdb.connect(db_path, read_only=True)
        try:
            df = c.execute(
                """
                SELECT bar_date, close
                FROM ohlcv_daily
                WHERE symbol = ? AND bar_date >= ? AND bar_date <= ?
                ORDER BY bar_date
                """,
                [symbol, pred_date, window_end],
            ).df()
        finally:
            c.close()
    if df.empty:
        return None

    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
    df = df.drop_duplicates(subset=["bar_date"], keep="last")
    on_pred = df[df["bar_date"] == pred_date]
    if on_pred.empty:
        return None
    after = df[df["bar_date"] > pred_date].reset_index(drop=True)
    if len(after) < n_trading_bars:
        return None
    realized_row = after.iloc[n_trading_bars - 1]
    return (
        float(on_pred["close"].iloc[0]),
        float(realized_row["close"]),
        realized_row["bar_date"],
    )


def settle_predictions(
    as_of: date,
    horizon_days: int = 5,
    *,
    sqlite_path: str | None = None,
    duckdb_path: str | None = None,
) -> int:
    """Settle predictions whose horizon has closed by ``as_of``.

    For each unsettled prediction whose pred_date + horizon trading bars
    is now available in OHLCV, compute realized log return and update.
    Then bucket realized_returns cross-sectionally per (universe, as_of,
    horizon_days) into quintiles for the realized_quintile column.
    """
    cutoff_pred_date = as_of - timedelta(days=horizon_days)

    with get_sqlite_conn(sqlite_path) as conn:
        rows = conn.execute(
            """
            SELECT id, universe, symbol, as_of
            FROM predictions_log
            WHERE as_of <= ?
              AND horizon_days = ?
              AND realized_return IS NULL
            """,
            [cutoff_pred_date.isoformat(), horizon_days],
        ).fetchall()

    if not rows:
        return 0

    settled = 0
    settled_groups: set[tuple[str, str]] = set()
    now = datetime.utcnow()

    # Share ONE read-only DuckDB connection across all `_close_at_offset`
    # calls. Without this, settling N predictions opens N write-mode DuckDB
    # connections, each acquiring the Windows file lock — every fresh open
    # is a potential collision with the API server's reads, and on a busy
    # day at least one inevitably loses. Read-only is correct here: we
    # only SELECT from ohlcv_daily.
    duck_db_path = duckdb_path or settings.duckdb_path
    duck_conn = duckdb.connect(duck_db_path, read_only=True)
    try:
        with get_sqlite_conn(sqlite_path) as conn:
            for id_, universe, symbol, pred_date_str in rows:
                pred_date = (
                    pred_date_str
                    if isinstance(pred_date_str, date)
                    else date.fromisoformat(str(pred_date_str))
                )
                triple = _close_at_offset(
                    symbol, pred_date, horizon_days,
                    duckdb_path=duckdb_path, duck_conn=duck_conn,
                )
                if triple is None:
                    continue
                close0, close_h, _ = triple
                if close0 <= 0 or close_h <= 0:
                    continue
                realized = math.log(close_h / close0)
                conn.execute(
                    """
                    UPDATE predictions_log
                    SET realized_return = ?, settled_at = ?
                    WHERE id = ?
                    """,
                    [realized, now, id_],
                )
                settled += 1
                settled_groups.add((universe, pred_date.isoformat()))
            conn.commit()

            # Cross-sectional realized quintile per (universe, as_of, horizon_days).
            for universe, pred_date_str in settled_groups:
                grp = conn.execute(
                    """
                    SELECT id, realized_return
                    FROM predictions_log
                    WHERE universe = ? AND as_of = ? AND horizon_days = ?
                      AND realized_return IS NOT NULL
                    """,
                    [universe, pred_date_str, horizon_days],
                ).fetchall()
                if len(grp) < 5:
                    continue
                df = pd.DataFrame(grp, columns=["id", "realized_return"])
                try:
                    df["q"] = pd.qcut(
                        df["realized_return"], 5, labels=False, duplicates="drop"
                    )
                except ValueError:
                    continue
                df = df.dropna(subset=["q"])
                for _id, q in zip(df["id"], df["q"], strict=True):
                    conn.execute(
                        "UPDATE predictions_log SET realized_quintile = ? WHERE id = ?",
                        [int(q), int(_id)],
                    )
            conn.commit()
    finally:
        duck_conn.close()

    log.info(f"settle_predictions: settled={settled} groups={len(settled_groups)}")
    return settled


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------


def performance_summary(
    universe: str,
    lookback_days: int = 90,
    *,
    sqlite_path: str | None = None,
) -> dict:
    """Aggregate stats over the last ``lookback_days`` of SETTLED predictions.

    `lookback_days` is interpreted as TRADING days (since `as_of` in
    predictions_log holds trading days, never weekends/holidays).
    Internally we convert to a calendar-day cutoff using a 7/5 ratio
    (≈ 1 trading week per 1 calendar week) so the SQL filter doesn't
    miss the earliest trading day in the window.
    """
    # Calendar-day cutoff sized to comfortably cover `lookback_days`
    # trading days. ceil(lookback_days * 7 / 5) is the tightest upper
    # bound. Add a small +3-day buffer for holiday clusters.
    calendar_span = (lookback_days * 7 + 4) // 5 + 3
    cutoff = (date.today() - timedelta(days=calendar_span)).isoformat()

    with get_sqlite_conn(sqlite_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM predictions_log
            WHERE universe = ?
              AND realized_return IS NOT NULL
              AND as_of >= ?
            """,
            conn,
            params=[universe, cutoff],
        )

    if df.empty:
        return {
            "universe": universe,
            "lookback_days": lookback_days,
            "n_predictions": 0,
            "n_settled": 0,
            "mean_daily_ic": float("nan"),
            "std_daily_ic": float("nan"),
            "t_stat_daily_ic": float("nan"),
            "hit_rate": float("nan"),
            "decile_spread": float("nan"),
            "calibration_table": [],
        }

    from packages.modeling.evaluate import (
        decile_spread,
        evaluate_predictions,
    )

    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    daily_metrics = evaluate_predictions(
        y_true_returns=df["realized_return"],
        y_pred_returns=df["predicted_return"],
        bar_dates=df["as_of"],
    )
    spread = decile_spread(
        df["realized_return"].to_numpy(), df["predicted_return"].to_numpy()
    )

    cal_table: list[dict] = []
    if df["top_quintile_proba"].notna().any() and df["realized_quintile"].notna().any():
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        df["proba_bin"] = pd.cut(df["top_quintile_proba"], bins=bins, include_lowest=True)
        n_classes_top = int(df["realized_quintile"].max()) if df["realized_quintile"].notna().any() else 4
        grouped = df.groupby("proba_bin", observed=True).agg(
            n=("id", "count"),
            observed_top_rate=(
                "realized_quintile",
                lambda x: float((x == n_classes_top).mean()) if len(x) else float("nan"),
            ),
            mean_proba=("top_quintile_proba", "mean"),
        )
        for bin_label, row in grouped.iterrows():
            cal_table.append(
                {
                    "proba_bin": str(bin_label),
                    "mean_proba": float(row["mean_proba"]),
                    "observed_top_rate": float(row["observed_top_rate"]),
                    "n": int(row["n"]),
                }
            )

    return {
        "universe": universe,
        "lookback_days": lookback_days,
        "n_predictions": len(df),
        "n_settled": len(df),
        "mean_daily_ic": daily_metrics["mean_daily_ic"],
        "std_daily_ic": daily_metrics["std_daily_ic"],
        "t_stat_daily_ic": daily_metrics["t_stat_daily_ic"],
        "mean_daily_rank_ic": daily_metrics["mean_daily_rank_ic"],
        "hit_rate": daily_metrics["hit_rate"],
        "decile_spread": spread,
        "calibration_table": cal_table,
    }
