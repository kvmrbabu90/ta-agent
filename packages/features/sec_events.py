"""8-K and major-form filing features (free historical events from SEC EDGAR).

Reads the ``edgar_filings`` table populated by ``scripts.refresh_edgar_filings``.
Causal: every value at row T uses ONLY filings with filing_date <= T.

Features (6):
    sec_events__days_since_last_8k          (capped at 90)
    sec_events__days_since_last_8k_2_02     (earnings-related 8-K — Item 2.02)
    sec_events__days_since_last_8k_5_02     (officer changes — Item 5.02)
    sec_events__count_8k_last_30d           (event density)
    sec_events__days_since_last_10q         (capped at 120)
    sec_events__days_since_last_10k         (capped at 400 — annual report)

When no filings exist for a symbol (CIK not in SEC ticker map, etc.), all
features are NaN — LightGBM handles this natively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.common.logging import log
from packages.features.base import PanelFeatureGroup
from packages.features.extensions import FeatureExtension, register_extension
from packages.ingestion.sec_filings import has_filings_data, load_filings

_FEATURE_COLS = (
    "days_since_last_8k",
    "days_since_last_8k_2_02",
    "days_since_last_8k_5_02",
    "count_8k_last_30d",
    "days_since_last_10q",
    "days_since_last_10k",
)

_DAYS_CAP_8K = 90
_DAYS_CAP_10Q = 120
_DAYS_CAP_10K = 400
_COUNT_WINDOW = 30


def _filter_8k_by_item(filings: pd.DataFrame, item_code: str) -> pd.DataFrame:
    """Return the 8-K rows whose comma-separated items list includes item_code."""
    is_8k = filings["form_type"] == "8-K"
    has_item = filings["items"].fillna("").str.contains(
        rf"(?:^|,){item_code}(?:,|$)", regex=True
    )
    return filings[is_8k & has_item]


def _days_since_per_symbol(
    panel: pd.DataFrame, filings: pd.DataFrame, *, cap_days: int
) -> np.ndarray:
    """Vectorized causal "days since last filing" per (symbol, bar_date).

    For each panel row at (sym, T), returns (T - max(filing_date for sym
    where filing_date <= T)). Bounded above by cap_days; values past the cap
    return NaN.
    """
    out = np.full(len(panel), np.nan)
    if filings.empty:
        return out

    panel_dates = pd.to_datetime(panel["bar_date"]).dt.date.to_numpy()
    panel_syms = panel["symbol"].to_numpy()

    by_sym = filings.sort_values(["symbol", "filing_date"]).groupby("symbol")
    sym_to_dates: dict[str, np.ndarray] = {
        sym: pd.to_datetime(grp["filing_date"]).dt.date.to_numpy()
        for sym, grp in by_sym
    }

    for i in range(len(panel)):
        sym = panel_syms[i]
        dates = sym_to_dates.get(sym)
        if dates is None or len(dates) == 0:
            continue
        # Find latest filing_date <= panel_date.
        idx = np.searchsorted(dates, panel_dates[i], side="right") - 1
        if idx < 0:
            continue
        delta = (panel_dates[i] - dates[idx]).days
        if 0 <= delta <= cap_days:
            out[i] = float(delta)
    return out


def _count_in_window_per_symbol(
    panel: pd.DataFrame, filings: pd.DataFrame, *, window_days: int
) -> np.ndarray:
    """Count of filings in (T - window_days, T] per (symbol, bar_date)."""
    out = np.zeros(len(panel))
    if filings.empty:
        return out

    panel_dates = pd.to_datetime(panel["bar_date"]).dt.date.to_numpy()
    panel_syms = panel["symbol"].to_numpy()

    by_sym = filings.sort_values(["symbol", "filing_date"]).groupby("symbol")
    sym_to_dates: dict[str, np.ndarray] = {
        sym: pd.to_datetime(grp["filing_date"]).dt.date.to_numpy()
        for sym, grp in by_sym
    }

    for i in range(len(panel)):
        sym = panel_syms[i]
        dates = sym_to_dates.get(sym)
        if dates is None or len(dates) == 0:
            continue
        bar = panel_dates[i]
        # Count filings strictly after (bar - window_days) and <= bar.
        from datetime import timedelta as _td
        lo_idx = np.searchsorted(dates, bar - _td(days=window_days), side="right")
        hi_idx = np.searchsorted(dates, bar, side="right")
        out[i] = float(hi_idx - lo_idx)
    return out


class SecEventsFeatures(PanelFeatureGroup):
    name = "sec_events"

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
        filings = load_filings(symbols, duckdb_path=self._duckdb_path)
        if filings.empty:
            log.info("sec_events: no filings rows for panel symbols; emitting zero/NaN")
            # Helpers handle empty filings correctly: days_since -> NaN, count -> 0.
            # We let them run rather than short-circuiting.

        # Filter by form-type once; pass each subset to the helpers.
        filings_8k = filings[filings["form_type"] == "8-K"]
        filings_10q = filings[filings["form_type"] == "10-Q"]
        filings_10k = filings[filings["form_type"] == "10-K"]
        filings_8k_2_02 = _filter_8k_by_item(filings, "2.02")
        filings_8k_5_02 = _filter_8k_by_item(filings, "5.02")

        out[f"{self.name}__days_since_last_8k"] = _days_since_per_symbol(
            panel, filings_8k, cap_days=_DAYS_CAP_8K
        )
        out[f"{self.name}__days_since_last_8k_2_02"] = _days_since_per_symbol(
            panel, filings_8k_2_02, cap_days=_DAYS_CAP_8K
        )
        out[f"{self.name}__days_since_last_8k_5_02"] = _days_since_per_symbol(
            panel, filings_8k_5_02, cap_days=_DAYS_CAP_8K
        )
        out[f"{self.name}__count_8k_last_30d"] = _count_in_window_per_symbol(
            panel, filings_8k, window_days=_COUNT_WINDOW
        )
        out[f"{self.name}__days_since_last_10q"] = _days_since_per_symbol(
            panel, filings_10q, cap_days=_DAYS_CAP_10Q
        )
        out[f"{self.name}__days_since_last_10k"] = _days_since_per_symbol(
            panel, filings_10k, cap_days=_DAYS_CAP_10K
        )
        return out


# --- Extension registration --------------------------------------------------


class _SecEventsExtension(FeatureExtension):
    name = "sec_events"
    kind = "panel"

    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        return has_filings_data(duckdb_path=duckdb_path)

    def make_group(self, *, duckdb_path: str | None = None) -> SecEventsFeatures:
        return SecEventsFeatures(duckdb_path=duckdb_path)


register_extension(_SecEventsExtension())
