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

# Import macro + earnings to trigger their extension registration.
# packages.features.interactions is intentionally NOT imported here:
# cross-seed validation showed +122% rank-IC on seed 42 but -156% on
# seed 44, and +earnings+interactions combined was destructively negative
# (-139% rank-IC seed 42). Code is preserved at packages/features/interactions.py
# for potential future use with deeper trees or different feature_fraction.
# To re-enable, uncomment the line below — it self-imports macro to enforce
# the correct registration order (macro must register before interactions).
import packages.features.earnings  # noqa: F401
import packages.features.macro  # noqa: F401

# import packages.features.interactions  # noqa: F401  # disabled, see comment above
from packages.common.logging import log
from packages.features.base import FeatureGroup, PanelFeatureGroup
from packages.features.cross_sectional import CrossSectionalFeatures
from packages.features.extensions import get_active_extensions
from packages.features.market_structure import MarketStructureFeatures
from packages.features.microstructure import MicrostructureFeatures
from packages.features.momentum import MomentumFeatures
from packages.features.price import PriceFeatures
from packages.features.regime import RegimeFeatures
from packages.features.swings import SwingFeatures
from packages.features.trend import TrendFeatures
from packages.features.volatility import VolatilityFeatures
from packages.features.volume import VolumeFeatures
from packages.features.volume_profile import VolumeProfileFeatures
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
    MarketStructureFeatures(),
    VolumeProfileFeatures(),
    SwingFeatures(),
]

# Always-on panel groups. Conditional ones are pulled in via the extension
# registry (see packages/features/extensions.py).
_BASE_PANEL_GROUPS: list[PanelFeatureGroup] = [
    CrossSectionalFeatures(),
    RegimeFeatures(),
]


def _resolve_panel_groups(duckdb_path: str | None) -> list[PanelFeatureGroup]:
    groups: list[PanelFeatureGroup] = list(_BASE_PANEL_GROUPS)
    for kind, group in get_active_extensions(duckdb_path=duckdb_path):
        if kind == "panel":
            groups.append(group)
    return groups


def _resolve_per_symbol_groups(duckdb_path: str | None) -> list[FeatureGroup]:
    groups: list[FeatureGroup] = list(PER_SYMBOL_GROUPS)
    for kind, group in get_active_extensions(duckdb_path=duckdb_path):
        if kind == "per_symbol":
            groups.append(group)
    return groups


def _all_symbols_in_window(
    universe: str, start: date, end: date, *, duckdb_path: str | None = None
) -> list[tuple[str, str]]:
    """(symbol, exchange) pairs that were members of ``universe`` at any point in [start, end].

    Returning the exchange is non-negotiable: passing it to ``get_ohlcv``
    prevents cross-exchange ticker collisions (e.g. HAL = Halliburton on NYSE
    and Hindustan Aeronautics on NSE) from polluting per-symbol feature
    windows.
    """
    with get_conn(duckdb_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT symbol, exchange
            FROM index_membership
            WHERE universe = ?
              AND start_date <= ?
              AND (end_date IS NULL OR end_date >= ?)
            ORDER BY symbol
            """,
            [universe, end, start],
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


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


def _per_symbol_features(
    symbol: str,
    ohlcv: pd.DataFrame,
    groups: list[FeatureGroup],
) -> pd.DataFrame:
    """Run all per-symbol groups and concatenate features for one symbol."""
    if ohlcv.empty:
        return pd.DataFrame()
    ohlcv = ohlcv.sort_values("bar_date").reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for group in groups:
        piece = group.compute(ohlcv)
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
    sym_exch_pairs = _all_symbols_in_window(universe, start, end, duckdb_path=duckdb_path)
    if not sym_exch_pairs:
        log.warning(f"build_feature_matrix: no symbols for {universe} in [{start}..{end}]")
        return pd.DataFrame()

    log.info(
        f"build_feature_matrix: {universe} [{start}..{end}] "
        f"({len(sym_exch_pairs)} symbols)"
    )

    per_symbol_groups = _resolve_per_symbol_groups(duckdb_path)

    rows: list[pd.DataFrame] = []
    for sym, exch in sym_exch_pairs:
        ohlcv = get_ohlcv(sym, start=start, end=end, exchange=exch)
        if ohlcv.empty:
            continue
        feat = _per_symbol_features(sym, ohlcv, per_symbol_groups)
        rows.append(feat)

    if not rows:
        log.warning("build_feature_matrix: no per-symbol features computed (empty OHLCV)")
        return pd.DataFrame()

    panel = pd.concat(rows, axis=0, ignore_index=True)

    # Run panel groups; they return (symbol, bar_date, panel_features...). Merge.
    panel_groups = _resolve_panel_groups(duckdb_path)
    for pg in panel_groups:
        pf = pg.compute(panel)
        panel = panel.merge(pf, on=["symbol", "bar_date"], how="left")

    if use_pit_membership:
        intervals = _membership_intervals(universe, duckdb_path=duckdb_path)
        mask = _is_member_mask(panel, intervals)
        feature_cols = [c for c in panel.columns if c not in ("symbol", "bar_date")]
        panel.loc[~mask, feature_cols] = pd.NA

    return panel
