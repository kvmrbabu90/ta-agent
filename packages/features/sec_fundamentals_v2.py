"""Phase D2 fundamental features: TTM aggregation + cross-sectional ranks.

Phase D's MVP used raw FY values directly — they were too persistent (one
update per year per company) and rank-IC seed-variance killed any signal.
Phase D2 fixes the two known weaknesses:

(1) **TTM aggregation.** Instead of FY values that update once a year, we
    sum the trailing four QUARTERLY observations to get rolling TTM
    (trailing twelve months). This refreshes 4× per year and the model
    sees changes much closer to when they happen. PIT-correct: TTM at row
    T uses only quarterly observations with filed_date <= T.

(2) **Cross-sectional ranks.** Raw factor values aren't what works in
    equity quant — RANKS WITHIN THE UNIVERSE PER DATE are. ROE of 18% is
    only meaningful relative to today's universe. We add rank columns
    (pct-rank within bar_date) for every TTM ratio.

Output features (10):
    Raw TTM ratios (5):
        sec_fundamentals_v2__roe_ttm
        sec_fundamentals_v2__net_margin_ttm
        sec_fundamentals_v2__asset_turnover_ttm
        sec_fundamentals_v2__gross_margin_ttm
        sec_fundamentals_v2__debt_to_equity   (point-in-time, no TTM — balance sheet)
    Cross-sectional ranks (5): same names suffixed with `_xs_rank`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.common.logging import log
from packages.features.base import PanelFeatureGroup
from packages.features.extensions import FeatureExtension, register_extension
from packages.ingestion.sec_fundamentals import (
    has_fundamentals_data,
    load_fundamentals,
)

_REQUIRED_CONCEPTS = (
    "NetIncomeLoss",
    "StockholdersEquity",
    "Assets",
    "LongTermDebt",
    "GrossProfit",
    "Revenues",
)
_RAW_RATIOS = (
    "roe_ttm",
    "net_margin_ttm",
    "asset_turnover_ttm",
    "gross_margin_ttm",
    "debt_to_equity",
)
_FEATURE_COLS = tuple(_RAW_RATIOS) + tuple(f"{r}_xs_rank" for r in _RAW_RATIOS)


def _ttm_value_per_symbol(
    panel_dates: np.ndarray, sym_quarters: pd.DataFrame
) -> np.ndarray:
    """For one symbol, compute TTM (trailing 4 quarterly filings) at each
    panel_date. PIT-correct: only quarters with filed_date <= panel_date count.

    Vectorized implementation:
      1. Dedup quarterly observations by period_end (keep latest filed_date —
         catches restatements). The result is one row per fiscal quarter.
      2. Sort by period_end ascending; pre-compute rolling 4-quarter sum.
      3. Track the filed_date of each quarter; for each panel_date T, take
         the index of the most-recent quarter whose filed_date <= T, and
         return the precomputed TTM[idx].

    Avoids the previous O(panel_dates × dedup × sort) nested loop, which
    blew up to >15 minutes on 826 symbols × 3650 dates × 6 concepts.
    """
    n = len(panel_dates)
    if sym_quarters.empty:
        return np.full(n, np.nan)

    quarterly = sym_quarters[sym_quarters["fp"].isin(["Q1", "Q2", "Q3", "Q4"])]
    if quarterly.empty:
        return _latest_value_per_symbol(panel_dates, sym_quarters)

    # Dedup by period_end, keeping the latest restatement (groupby -> last).
    quarterly = quarterly.sort_values("filed_date")
    dedup = (
        quarterly.groupby("period_end", as_index=False)
        .agg(filed_date=("filed_date", "last"), value=("value", "last"))
        .sort_values("period_end")
        .reset_index(drop=True)
    )
    if len(dedup) < 4:
        return np.full(n, np.nan)

    # Pre-compute rolling 4-quarter sum per fiscal-quarter index.
    values_arr = dedup["value"].to_numpy()
    period_ends_arr = dedup["period_end"].to_numpy()
    filed_dates_arr = dedup["filed_date"].to_numpy()
    ttm = np.full(len(values_arr), np.nan)
    if len(values_arr) >= 4:
        # Sum of [i-3, i] inclusive for i >= 3.
        cumsum = np.cumsum(values_arr)
        ttm[3:] = cumsum[3:] - np.concatenate([[0.0], cumsum[:-4]])

    # Each quarter's TTM is "available" only after its filed_date. For each
    # panel_date T, find the most-recent quarter with filed_date <= T and
    # return its TTM value.
    # We need filed_dates_arr to be sorted for searchsorted. Sort once.
    sort_idx = np.argsort(filed_dates_arr)
    sorted_filed = filed_dates_arr[sort_idx]
    sorted_ttm = ttm[sort_idx]
    quarter_idx = np.searchsorted(sorted_filed, panel_dates, side="right") - 1
    out = np.full(n, np.nan)
    valid = quarter_idx >= 0
    if valid.any():
        out[valid] = sorted_ttm[quarter_idx[valid]]
    # The unused period_ends_arr remains a useful debugging breadcrumb when
    # diffing this against the slow loop's output. Suppress lint via underscore.
    _ = period_ends_arr
    return out


def _latest_value_per_symbol(
    panel_dates: np.ndarray, sym_obs: pd.DataFrame
) -> np.ndarray:
    """Latest single value (any fp) with filed_date <= panel_date."""
    n = len(panel_dates)
    if sym_obs.empty:
        return np.full(n, np.nan)
    sorted_obs = sym_obs.sort_values("filed_date").reset_index(drop=True)
    filed_dates = sorted_obs["filed_date"].to_numpy()
    values = sorted_obs["value"].to_numpy()
    out = np.full(n, np.nan)
    idx = np.searchsorted(filed_dates, panel_dates, side="right") - 1
    mask_valid = idx >= 0
    if mask_valid.any():
        out[mask_valid] = values[idx[mask_valid]]
    return out


def _safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Divide elementwise, returning NaN where denom <= 0 or NaN."""
    out = np.full_like(numer, np.nan, dtype=float)
    mask = (denom > 0) & np.isfinite(numer) & np.isfinite(denom)
    out[mask] = numer[mask] / denom[mask]
    return out


class SecFundamentalsV2Features(PanelFeatureGroup):
    name = "sec_fundamentals_v2"

    def __init__(self, duckdb_path: str | None = None) -> None:
        self._duckdb_path = duckdb_path

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)
        out = panel[["symbol", "bar_date"]].copy()

        if panel.empty:
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        symbols = panel["symbol"].unique().tolist()
        # Pull both quarterly and FY data — TTM uses Q1-Q4, debt_to_equity
        # uses the most recent point-in-time value (any fp).
        fundamentals = load_fundamentals(
            symbols, concepts=list(_REQUIRED_CONCEPTS),
            duckdb_path=self._duckdb_path,
        )
        if fundamentals.empty:
            log.info(f"{self.name}: no rows for panel symbols; emitting NaN")
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        fundamentals = fundamentals.copy()
        fundamentals["filed_date"] = pd.to_datetime(fundamentals["filed_date"]).dt.date
        fundamentals["period_end"] = pd.to_datetime(fundamentals["period_end"]).dt.date

        # Build per-(symbol, concept) observations once.
        by_sc = fundamentals.groupby(["symbol", "concept"])

        # Compute raw TTM ratios per symbol.
        out_pieces: list[pd.DataFrame] = []
        for sym in symbols:
            sub = panel[panel["symbol"] == sym][["symbol", "bar_date"]].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("bar_date").reset_index(drop=True)
            panel_dates_arr = pd.to_datetime(sub["bar_date"]).dt.date.to_numpy()

            def _get(concept: str, ttm: bool = True, sym=sym, dates=panel_dates_arr) -> np.ndarray:
                try:
                    grp = by_sc.get_group((sym, concept))
                except KeyError:
                    return np.full(len(dates), np.nan)
                if ttm:
                    return _ttm_value_per_symbol(dates, grp)
                return _latest_value_per_symbol(dates, grp)

            ni_ttm = _get("NetIncomeLoss", ttm=True)
            rev_ttm = _get("Revenues", ttm=True)
            gp_ttm = _get("GrossProfit", ttm=True)
            # Equity / Assets / Debt are STOCK measures (point-in-time, balance sheet),
            # not flow measures — we use the most recent value, not a TTM sum.
            eq = _get("StockholdersEquity", ttm=False)
            assets = _get("Assets", ttm=False)
            debt = _get("LongTermDebt", ttm=False)

            sub[f"{self.name}__roe_ttm"] = _safe_div(ni_ttm, eq)
            sub[f"{self.name}__net_margin_ttm"] = _safe_div(ni_ttm, rev_ttm)
            sub[f"{self.name}__asset_turnover_ttm"] = _safe_div(rev_ttm, assets)
            sub[f"{self.name}__gross_margin_ttm"] = _safe_div(gp_ttm, rev_ttm)
            sub[f"{self.name}__debt_to_equity"] = _safe_div(debt, eq)
            out_pieces.append(sub)

        if not out_pieces:
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        merged = pd.concat(out_pieces, axis=0, ignore_index=True)
        out = out.merge(merged, on=["symbol", "bar_date"], how="left")

        # Cross-sectional pct-rank within each bar_date.
        for ratio in _RAW_RATIOS:
            raw_col = f"{self.name}__{ratio}"
            rank_col = f"{self.name}__{ratio}_xs_rank"
            if raw_col not in out.columns:
                out[rank_col] = pd.NA
                continue
            # rank(pct=True) within each bar_date; NaN inputs stay NaN.
            out[rank_col] = (
                out.groupby("bar_date", observed=True, sort=False)[raw_col]
                .rank(pct=True)
                .values
            )
        return out


# --- Extension registration --------------------------------------------------


class _SecFundamentalsV2Extension(FeatureExtension):
    name = "sec_fundamentals_v2"
    kind = "panel"

    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        return has_fundamentals_data(duckdb_path=duckdb_path)

    def make_group(
        self, *, duckdb_path: str | None = None
    ) -> SecFundamentalsV2Features:
        return SecFundamentalsV2Features(duckdb_path=duckdb_path)


register_extension(_SecFundamentalsV2Extension())
