"""Fundamental factor features from SEC EDGAR (free, official).

Reads ``sec_fundamentals`` (populated by scripts.refresh_sec_fundamentals).
Causal: every value at row T uses ONLY observations with filed_date <= T.
We deliberately use ANNUAL (FY) values to dodge the 10-Q vs 10-K
calendar-mismatch trap — a stock at bar_date T uses the most recent FY
report whose 10-K was filed on or before T (typically 30-90 days after
fiscal year end).

Five features (no market-cap dependency to keep the MVP simple):

    sec_fundamentals__roe                NetIncomeLoss / StockholdersEquity   (FY)
    sec_fundamentals__debt_to_equity     LongTermDebt / StockholdersEquity    (FY)
    sec_fundamentals__gross_margin       GrossProfit / Revenues               (FY)
    sec_fundamentals__net_margin         NetIncomeLoss / Revenues             (FY)
    sec_fundamentals__asset_turnover     Revenues / Assets                    (FY)

When a company hasn't filed a 10-K yet (early panel rows), or some concept
is missing, the relevant feature is NaN. LightGBM handles NaN natively.

Cross-sectional ranks of these factors are the canonical "value/quality"
factor zoo (Fama-French + AQR-style) — among the most replicated equity
anomalies in academic finance. Whether they still work in 2026 on SP500
is what we're about to find out.
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

_REQUIRED_CONCEPTS = ("NetIncomeLoss", "StockholdersEquity", "Assets",
                      "LongTermDebt", "GrossProfit", "Revenues")
_FEATURE_COLS = (
    "roe",
    "debt_to_equity",
    "gross_margin",
    "net_margin",
    "asset_turnover",
)


def _latest_fy_per_concept_per_symbol(
    symbols: list[str],
    panel_dates: np.ndarray,
    fundamentals: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """Return {concept: {symbol: array-len-N values aligned to panel_dates}}.

    For each (symbol, concept), at each panel_date T, return the most recent
    value where filed_date <= T AND fp == 'FY'. NaN if no qualifying row.
    """
    out: dict[str, dict[str, np.ndarray]] = {c: {} for c in _REQUIRED_CONCEPTS}
    if fundamentals.empty:
        for c in _REQUIRED_CONCEPTS:
            for s in symbols:
                out[c][s] = np.full(len(panel_dates), np.nan)
        return out

    fy_only = fundamentals[fundamentals["fp"] == "FY"].copy()
    fy_only["filed_date"] = pd.to_datetime(fy_only["filed_date"]).dt.date
    fy_only = fy_only.sort_values(["symbol", "concept", "filed_date"]).reset_index(drop=True)

    by_sym_concept = fy_only.groupby(["symbol", "concept"])

    for sym in symbols:
        for concept in _REQUIRED_CONCEPTS:
            key = (sym, concept)
            try:
                grp = by_sym_concept.get_group(key)
            except KeyError:
                out[concept][sym] = np.full(len(panel_dates), np.nan)
                continue
            filed_dates = grp["filed_date"].to_numpy()
            values = grp["value"].to_numpy()
            # For each panel_date T, find latest filed_date <= T (causal).
            idx = np.searchsorted(filed_dates, panel_dates, side="right") - 1
            arr = np.full(len(panel_dates), np.nan)
            mask_valid = idx >= 0
            if mask_valid.any():
                arr[mask_valid] = values[idx[mask_valid]]
            out[concept][sym] = arr
    return out


def _safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Divide elementwise, returning NaN where denom <= 0 or NaN."""
    out = np.full_like(numer, np.nan, dtype=float)
    mask = (denom > 0) & np.isfinite(numer) & np.isfinite(denom)
    out[mask] = numer[mask] / denom[mask]
    return out


class SecFundamentalsFeatures(PanelFeatureGroup):
    name = "sec_fundamentals"

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
        fundamentals = load_fundamentals(
            symbols, concepts=list(_REQUIRED_CONCEPTS),
            fp=["FY"], duckdb_path=self._duckdb_path,
        )
        if fundamentals.empty:
            log.info("sec_fundamentals: no FY rows for panel symbols; emitting NaN")
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        # Build a per-symbol-per-concept lookup aligned to the symbol's panel dates.
        # For efficiency, group panel rows by symbol and process one symbol at a time.
        out_pieces: list[pd.DataFrame] = []
        for sym in symbols:
            sub = panel[panel["symbol"] == sym][["symbol", "bar_date"]].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("bar_date").reset_index(drop=True)
            panel_dates_arr = pd.to_datetime(sub["bar_date"]).dt.date.to_numpy()

            # Pull latest FY value per concept (aligned to this symbol's bar_dates).
            per_concept = _latest_fy_per_concept_per_symbol(
                [sym], panel_dates_arr, fundamentals[fundamentals["symbol"] == sym]
            )
            ni = per_concept["NetIncomeLoss"][sym]
            eq = per_concept["StockholdersEquity"][sym]
            assets = per_concept["Assets"][sym]
            debt = per_concept["LongTermDebt"][sym]
            gp = per_concept["GrossProfit"][sym]
            rev = per_concept["Revenues"][sym]

            sub[f"{self.name}__roe"] = _safe_div(ni, eq)
            sub[f"{self.name}__debt_to_equity"] = _safe_div(debt, eq)
            sub[f"{self.name}__gross_margin"] = _safe_div(gp, rev)
            sub[f"{self.name}__net_margin"] = _safe_div(ni, rev)
            sub[f"{self.name}__asset_turnover"] = _safe_div(rev, assets)
            out_pieces.append(sub)

        if not out_pieces:
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out
        merged = pd.concat(out_pieces, axis=0, ignore_index=True)
        out = out.merge(merged, on=["symbol", "bar_date"], how="left")
        return out


# --- Extension registration --------------------------------------------------


class _SecFundamentalsExtension(FeatureExtension):
    name = "sec_fundamentals"
    kind = "panel"

    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        return has_fundamentals_data(duckdb_path=duckdb_path)

    def make_group(
        self, *, duckdb_path: str | None = None
    ) -> SecFundamentalsFeatures:
        return SecFundamentalsFeatures(duckdb_path=duckdb_path)


register_extension(_SecFundamentalsExtension())
