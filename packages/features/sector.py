"""GICS sector one-hot features.

Joins each (symbol, bar_date) row with the symbol's GICS sector and
emits one-hot indicators for each canonical sector.

Data source:
    `data/processed/sectors_{universe}.parquet` — refreshed by
    `scripts/refresh_sectors.py`. The parquet is the canonical store for
    sector data in this codebase (the alternative DuckDB-backed path at
    `packages/ingestion/sector_metadata.py` was built first then
    de-duplicated against the existing parquet).

Why one-hot rather than integer-encoded:
    Our pipeline coerces all features to float. Integer-encoded sector
    would produce tree splits like 'sector_id < 4', which is meaningless
    arbitrary bisection of an unordered list. One-hot avoids this — each
    split is 'is_technology', 'is_financial_services', etc., which is
    semantically meaningful.

Why this is a PANEL extension (not per-symbol):
    Sector is per-symbol but TIME-INVARIANT. Per-symbol feature groups
    expect time-varying outputs from one symbol's OHLCV. Doing the
    symbol→sector join in panel space is cheaper (one merge vs N
    per-symbol lookups) and matches how cross-sectional / macro /
    earnings extensions work.

Validation gate (per the project's reject-by-default convention):
    See `scripts/validate_sector_features.py`. Pattern from
    `validate_phase_a.py`: 3 random seeds × 5-fold purged CV on the
    same train/test slice with and without sector features. Ship if:
      - Mean rank-IC delta > 0
      - Worst per-seed delta > 0 (no seed regresses)
      - Decile spread does not sign-flip negative
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from packages.common.config import PROCESSED_DIR
from packages.features.base import PanelFeatureGroup
from packages.features.extensions import FeatureExtension, register_extension

# Canonical GICS sector names. Stable column set guarantees the train-time
# feature_cols list matches inference-time, even when a member rotation
# adds a stock from a sector we hadn't seen before (it just falls into
# the "Unknown" bucket).
CANONICAL_SECTORS = [
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Basic Materials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Unknown",
]

# yfinance occasionally uses slightly different labels than the canonical
# GICS names; normalize at load time so the feature set stays stable.
_LABEL_NORMALIZE = {
    "Financials": "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials",
    "Information Technology": "Technology",
    "Communications Services": "Communication Services",
    "UNKNOWN": "Unknown",
}


def _sector_col(sector: str) -> str:
    """Make a column-friendly version of a sector name."""
    return sector.lower().replace(" ", "_").replace("&", "and")


def _sectors_path(universe: str = "SP500") -> Path:
    return Path(PROCESSED_DIR) / f"sectors_{universe.lower()}.parquet"


def _load_sectors(universe: str = "SP500") -> dict[str, str]:
    """Load (symbol -> canonical sector) from the parquet. Empty dict if missing."""
    p = _sectors_path(universe)
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return {}
    if df.empty or "symbol" not in df.columns or "sector" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for sym, sector in zip(df["symbol"], df["sector"], strict=True):
        if not sym:
            continue
        normalized = _LABEL_NORMALIZE.get(str(sector).strip(), str(sector).strip())
        if normalized not in CANONICAL_SECTORS:
            normalized = "Unknown"
        out[sym] = normalized
    return out


def has_sector_data(universe: str = "SP500") -> bool:
    """Cheap probe used by the FeatureExtension to gate activation."""
    p = _sectors_path(universe)
    return p.exists() and p.stat().st_size > 0


class SectorFeatures(PanelFeatureGroup):
    name = "sector"

    def __init__(self, duckdb_path: str | None = None) -> None:
        # duckdb_path is in the contract for FeatureExtension.make_group but
        # we don't use it — sector data lives in parquet.
        del duckdb_path
        # Cache the lookup to avoid re-reading the parquet on every call.
        # Negligible memory (503 entries) and the file rarely changes.
        self._sector_by_symbol: dict[str, str] | None = None

    def _ensure_loaded(self) -> dict[str, str]:
        if self._sector_by_symbol is None:
            self._sector_by_symbol = _load_sectors("SP500")
        return self._sector_by_symbol

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)
        out = panel[["symbol", "bar_date"]].copy()

        # Always emit the same column set, even if data missing — keeps
        # the train/inference feature_cols lists stable across runs.
        sector_cols = [f"{self.name}__is_{_sector_col(s)}" for s in CANONICAL_SECTORS]
        if panel.empty:
            for c in sector_cols:
                out[c] = 0.0
            return out

        sector_by_symbol = self._ensure_loaded()
        if not sector_by_symbol:
            for c in sector_cols:
                out[c] = 0.0
            out[f"{self.name}__is_unknown"] = 1.0
            return out

        # Vectorized one-hot. Default unknown symbols to "Unknown" so the
        # bucket is a learnable signal, not silently zeroed across all cols.
        symbol_arr = out["symbol"].to_numpy()
        for sector in CANONICAL_SECTORS:
            col = f"{self.name}__is_{_sector_col(sector)}"
            out[col] = [
                1.0 if sector_by_symbol.get(s, "Unknown") == sector else 0.0
                for s in symbol_arr
            ]
        return out


# --- Extension registration --------------------------------------------------


class _SectorExtension(FeatureExtension):
    name = "sector"
    kind = "panel"

    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        return has_sector_data("SP500")

    def make_group(self, *, duckdb_path: str | None = None) -> SectorFeatures:
        return SectorFeatures(duckdb_path=duckdb_path)


register_extension(_SectorExtension())


__all__ = ["SectorFeatures", "CANONICAL_SECTORS"]
