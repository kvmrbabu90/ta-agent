"""Feature pipeline orchestrator.

build_feature_matrix(universe, start, end, ...) →
    1. Load OHLCV for every symbol that was a member of ``universe`` at any
       point during [start, end]. (Survivorship-bias-free.)
    2. For each symbol, compute the 6 per-symbol feature groups.
    3. Concatenate into a long-format panel.
    4. Compute panel-wide cross-sectional and regime features on that panel.
    5. Mask features for any (symbol, date) where the symbol was NOT a
       member of the universe on that date — using point-in-time membership.

The masking step is non-negotiable. Without it, a stock that joined the
index on date T would carry features computed from its pre-T history into
cross-sectional ranks, biasing the universe panel.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from packages.common.logging import log
from packages.features.base import FeatureGroup, PanelFeatureGroup
from packages.features.cross_sectional import CrossSectionalFeatures
from packages.features.microstructure import MicrostructureFeatures
from packages.features.momentum import MomentumFeatures
from packages.features.price import PriceFeatures
from packages.features.regime import RegimeFeatures
from packages.features.trend import TrendFeatures
from packages.features.volatility import VolatilityFeatures
from packages.features.volume import VolumeFeatures
from packages.ingestion.storage import get_conn, get_ohlcv

# Order matters: panel groups (cross-sectional, regime) read columns produced
# by the per-symbol groups. Within a phase, order is irrelevant.
PER_SYMBOL_GROUPS: list[FeatureGroup] = [
    PriceFeatures(),
    TrendFeatures(),
    MomentumFeatures(),
    VolatilityFeatures(),
    VolumeFeatures(),
    MicrostructureFeatures(),
]

PANEL_GROUPS: list[PanelFeatureGroup] = [
    CrossSectionalFeatures(),
    RegimeFeatures(),
]


def _all_symbols_in_window(
    universe: str, start: date, end: date, *, duckdb_path: str | None = None
) -> list[str]:
    """All symbols that were members of ``universe`` at any point in [start, end]."""
    with get_conn(duckdb_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT symbol
            FROM index_membership
            WHERE universe = ?
              AND start_date <= ?
              AND (end_date IS NULL OR end_date >= ?)
            ORDER BY symbol
            """,
            [universe, end, start],
        ).fetchall()
    return [r[0] for r in rows]


def _membership_intervals(
    universe: str, *, duckdb_path: str | None = None
) -> pd.DataFrame:
    """All (symbol, start_date, end_date) rows for the universe."""
    with get_conn(duckdb_path) as conn:
        return conn.execute(
            """
            SELECT symbol, start_date, end_date
            FROM index_membership
            WHERE universe = ?
            """,
            [universe],
        ).df()


def _is_member_mask(
    panel: pd.DataFrame, intervals: pd.DataFrame
) -> pd.Series:
    """True for rows where (symbol, bar_date) was an active membership."""
    if intervals.empty:
        return pd.Series(False, index=panel.index)
    intervals = intervals.copy()
    intervals["end_date"] = intervals["end_date"].fillna(pd.Timestamp.max.date())
    merged = panel[["symbol", "bar_date"]].merge(intervals, on="symbol", how="left")
    bar_dt = pd.to_datetime(merged["bar_date"]).dt.date
    sd = pd.to_datetime(merged["start_date"]).dt.date
    ed = pd.to_datetime(merged["end_date"]).dt.date
    mask = (sd <= bar_dt) & (bar_dt <= ed)
    # If a symbol has multiple intervals, OR them by (symbol, bar_date).
    out = mask.groupby([merged["symbol"], merged["bar_date"]]).max()
    out = panel[["symbol", "bar_date"]].merge(
        out.rename("is_member").reset_index(), on=["symbol", "bar_date"], how="left"
    )["is_member"]
    return out.fillna(False).astype(bool).values


def _per_symbol_features(symbol: str, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Run all per-symbol groups and concatenate features for one symbol."""
    if ohlcv.empty:
        return pd.DataFrame()
    ohlcv = ohlcv.sort_values("bar_date").reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for group in PER_SYMBOL_GROUPS:
        piece = group.compute(ohlcv)
        # Each group returns DataFrame indexed by bar_date.
        piece = piece.reset_index(drop=True)
        pieces.append(piece)
    feat = pd.concat(pieces, axis=1)
    feat.insert(0, "bar_date", ohlcv["bar_date"].values)
    feat.insert(0, "symbol", symbol)
    return feat


def build_feature_matrix(
    universe: str,
    start: date,
    end: date,
    *,
    use_pit_membership: bool = True,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """Assemble the long-format feature panel for ``universe`` over [start, end].

    Returns columns: symbol, bar_date, plus ~50 feature columns named like
    ``price__log_return_5d``, ``trend__macd_histogram``, ``cross_sectional__xs_rank_rsi_14``.
    """
    symbols = _all_symbols_in_window(universe, start, end, duckdb_path=duckdb_path)
    if not symbols:
        log.warning(f"build_feature_matrix: no symbols for {universe} in [{start}..{end}]")
        return pd.DataFrame()

    log.info(
        f"build_feature_matrix: {universe} [{start}..{end}] "
        f"({len(symbols)} symbols)"
    )

    rows: list[pd.DataFrame] = []
    for sym in symbols:
        ohlcv = get_ohlcv(sym, start=start, end=end)
        if ohlcv.empty:
            continue
        feat = _per_symbol_features(sym, ohlcv)
        rows.append(feat)

    if not rows:
        log.warning("build_feature_matrix: no per-symbol features computed (empty OHLCV)")
        return pd.DataFrame()

    panel = pd.concat(rows, axis=0, ignore_index=True)

    # Run panel groups; they return (symbol, bar_date, panel_features...). Merge.
    for pg in PANEL_GROUPS:
        pf = pg.compute(panel)
        panel = panel.merge(pf, on=["symbol", "bar_date"], how="left")

    if use_pit_membership:
        intervals = _membership_intervals(universe, duckdb_path=duckdb_path)
        mask = _is_member_mask(panel, intervals)
        feature_cols = [c for c in panel.columns if c not in ("symbol", "bar_date")]
        panel.loc[~mask, feature_cols] = pd.NA

    return panel
