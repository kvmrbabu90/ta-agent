"""Master dataset assembly: join features + forward returns + quintile labels.

The output is a long-format DataFrame keyed by (symbol, bar_date) with:
    - all feature columns from packages.features.pipeline
    - fwd_return_{H}d                  (regression target)
    - fwd_quintile_{H}d                (classification target, 0..n_classes-1)
    - in_universe                      (bool, point-in-time membership flag)

We retain rows where ``in_universe == False`` so callers can distinguish
"not in the universe on date T" (correct) from "in the universe but no label
yet" (insufficient future data). Modeling code MUST filter to in_universe=True
AND label-not-null before training.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from packages.common.config import PROCESSED_DIR
from packages.common.logging import log
from packages.features.pipeline import build_feature_matrix
from packages.ingestion.storage import get_conn, get_ohlcv
from packages.labels.targets import (
    compute_forward_log_return,
    compute_quintile_labels,
)


def _load_membership_for_universe(
    universe: str, *, duckdb_path: str | None = None
) -> pd.DataFrame:
    with get_conn(duckdb_path) as conn:
        return conn.execute(
            "SELECT universe, symbol, start_date, end_date FROM index_membership WHERE universe = ?",
            [universe],
        ).df()


def _build_in_universe_flag(
    panel: pd.DataFrame, membership: pd.DataFrame
) -> pd.Series:
    """True for (symbol, bar_date) rows where the symbol was in the universe on that date."""
    if membership.empty:
        return pd.Series(False, index=panel.index)
    intervals = membership.copy()
    intervals["start_date"] = pd.to_datetime(intervals["start_date"]).dt.date
    intervals["end_date"] = pd.to_datetime(
        intervals["end_date"].fillna(pd.Timestamp.max.date())
    ).dt.date

    merged = panel[["symbol", "bar_date"]].merge(
        intervals[["symbol", "start_date", "end_date"]], on="symbol", how="left"
    )
    bar_dt = pd.to_datetime(merged["bar_date"]).dt.date
    in_window = (intervals_safe(merged, "start_date") <= bar_dt) & (
        bar_dt <= intervals_safe(merged, "end_date")
    )
    is_member = (
        in_window.groupby(
            [merged["symbol"], merged["bar_date"]]
        ).max()
        .rename("is_member")
        .reset_index()
    )
    out = (
        panel[["symbol", "bar_date"]]
        .merge(is_member, on=["symbol", "bar_date"], how="left")["is_member"]
        .fillna(False)
        .astype(bool)
    )
    return out


def intervals_safe(df: pd.DataFrame, col: str) -> pd.Series:
    """Safe date coercion for membership intervals (handles NaN merge artifacts)."""
    return pd.to_datetime(df[col]).dt.date


def _load_universe_ohlcv(
    universe: str,
    start: date,
    end: date,
    *,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """OHLCV for every symbol that was a member of ``universe`` at any point in [start, end].

    Pulls (symbol, exchange) pairs from membership and filters bars by both.
    Symbol-only fetch causes cross-exchange collisions (e.g. HAL = Halliburton
    on NYSE and Hindustan Aeronautics on NSE), which corrupts forward returns.
    """
    with get_conn(duckdb_path) as conn:
        sym_exch_pairs = conn.execute(
            """
            SELECT DISTINCT symbol, exchange
            FROM index_membership
            WHERE universe = ?
              AND start_date <= ?
              AND (end_date IS NULL OR end_date >= ?)
            """,
            [universe, end, start],
        ).fetchall()
    if not sym_exch_pairs:
        return pd.DataFrame(columns=["symbol", "bar_date", "close"])

    rows: list[pd.DataFrame] = []
    for sym, exch in sym_exch_pairs:
        bars = get_ohlcv(sym, start=start, end=end, exchange=exch)
        if not bars.empty:
            rows.append(bars[["symbol", "bar_date", "close"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "bar_date", "close"])
    return pd.concat(rows, axis=0, ignore_index=True)


def _print_sanity_stats(df: pd.DataFrame, fwd_col: str, q_col: str) -> None:
    total = len(df)
    in_universe = int(df["in_universe"].sum())
    labeled = int(df[q_col].notna().sum())
    log.info(f"dataset rows total={total:,} in_universe={in_universe:,} labeled={labeled:,}")
    if labeled > 0:
        m = df[df[q_col].notna()]
        log.info(
            f"forward-return: mean={m[fwd_col].mean():.5f} std={m[fwd_col].std():.5f}"
        )
        balance = m[q_col].value_counts(normalize=True).sort_index()
        balance_str = ", ".join(f"q{int(k)}={v:.1%}" for k, v in balance.items())
        log.info(f"quintile balance: {balance_str}")


def build_training_dataset(
    universe: str,
    start: date,
    end: date,
    *,
    horizon_days: int = 5,
    n_classes: int = 5,
    duckdb_path: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Assemble the final (features + targets) training panel for a universe / window."""
    fwd_col = f"fwd_return_{horizon_days}d"
    q_col = f"fwd_quintile_{horizon_days}d"

    # 1. Features (PIT-masked inside the pipeline).
    features = build_feature_matrix(universe, start, end, duckdb_path=duckdb_path)
    if features.empty:
        log.warning("build_training_dataset: empty feature matrix")
        return pd.DataFrame()

    # 2. Forward returns from raw closes.
    closes = _load_universe_ohlcv(universe, start, end, duckdb_path=duckdb_path)
    fwd = compute_forward_log_return(closes, horizon_days=horizon_days)

    # 3. Quintile labels (PIT-respecting).
    membership = _load_membership_for_universe(universe, duckdb_path=duckdb_path)
    membership["universe"] = universe  # ensure col present even if filtered earlier
    quintiles = compute_quintile_labels(
        fwd, membership, universe, horizon_days=horizon_days, n_classes=n_classes
    )

    # 4. Stitch it all together.
    panel = features.copy()
    panel["bar_date"] = pd.to_datetime(panel["bar_date"]).dt.date
    quintiles["bar_date"] = pd.to_datetime(quintiles["bar_date"]).dt.date
    panel = panel.merge(
        quintiles[["symbol", "bar_date", fwd_col, q_col]],
        on=["symbol", "bar_date"],
        how="left",
    )

    panel["in_universe"] = _build_in_universe_flag(panel, membership).values

    # Sanity stats
    _print_sanity_stats(panel, fwd_col, q_col)

    # Persist if requested.
    if output_path is None:
        output_path = (
            PROCESSED_DIR / f"training_{universe.lower()}_{start}_{end}.parquet"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    log.info(f"build_training_dataset: wrote {len(panel):,} rows → {output_path}")
    return panel
