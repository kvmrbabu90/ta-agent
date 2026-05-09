"""Earnings-window features broadcast to the (symbol, bar_date) panel.

Reads from the ``earnings_calendar`` table populated by
``scripts.refresh_earnings``. Causal — every value at row T uses ONLY
earnings rows with announce_date <= T (for "since" features) or
announce_date > T (for "to" features, which are forward-looking but the
date itself is publicly known in advance via the company calendar).

Features (4):
    earnings__days_since_last_earnings   Bounded at 90 (older than that = NaN).
                                         Captures post-earnings drift / reversal window.
    earnings__days_to_next_earnings      Bounded at 60. Captures pre-earnings runup
                                         and elevated implied vol.
    earnings__last_surprise_pct          Most recent earnings surprise (%).
    earnings__last_surprise_z_252        Z-score of last_surprise_pct vs the symbol's
                                         trailing 252-trading-day distribution of past
                                         surprises (causal — uses only quarters before T).

When earnings data is missing for a symbol entirely (e.g. delisted, never reported),
all four features are NaN — LightGBM handles this natively.

Causality contract:
    For row at (symbol S, bar_date T):
      - last_earnings = max(announce_date for S where announce_date <= T)
      - next_earnings = min(announce_date for S where announce_date > T)
    Both are anchored to T-relative dates, never peeking past T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.common.logging import log
from packages.features.base import PanelFeatureGroup
from packages.features.extensions import FeatureExtension, register_extension
from packages.ingestion.earnings import has_earnings_data, load_earnings

_DAYS_SINCE_BOUND = 90
_DAYS_TO_BOUND = 60
_FEATURE_COLS = (
    "days_since_last_earnings",
    "days_to_next_earnings",
    "last_surprise_pct",
    "last_surprise_z_252",
)


class EarningsFeatures(PanelFeatureGroup):
    name = "earnings"

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
        earnings = load_earnings(symbols, duckdb_path=self._duckdb_path)
        if earnings.empty:
            log.info("earnings: no earnings rows for panel symbols; emitting NaN")
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        # Coerce announce_date to pure date for safe comparison with bar_date.
        earnings["announce_date"] = pd.to_datetime(earnings["announce_date"]).dt.date
        earnings = earnings.sort_values(["symbol", "announce_date"]).reset_index(drop=True)

        # Compute trailing 252-trading-day surprise z-score per symbol BEFORE the join.
        # Use "previous 8 surprises (~ 2 years quarterly)" as the rolling window —
        # 252 calendar days isn't perfectly aligned with quarterly cadence so we
        # use 8 prior reports as a proxy (causal: shift(1) before rolling).
        earnings["surprise_lag1"] = earnings.groupby("symbol")["surprise_pct"].shift(1)
        # Rolling mean/std of past surprises, computed per symbol.
        # min_periods=4 → need at least 4 prior quarters before issuing a non-NaN z.
        roll = earnings.groupby("symbol")["surprise_lag1"].rolling(8, min_periods=4)
        earnings["surprise_mean_8"] = roll.mean().reset_index(level=0, drop=True)
        earnings["surprise_std_8"] = roll.std().reset_index(level=0, drop=True)
        earnings = earnings.drop(columns=["surprise_lag1"])

        out_pieces: list[pd.DataFrame] = []
        for sym in symbols:
            sym_panel = panel[panel["symbol"] == sym][["symbol", "bar_date"]].copy()
            if sym_panel.empty:
                continue
            sym_earnings = earnings[earnings["symbol"] == sym]
            if sym_earnings.empty:
                # No earnings data: emit NaN for this symbol.
                for c in _FEATURE_COLS:
                    sym_panel[f"{self.name}__{c}"] = np.nan
                out_pieces.append(sym_panel)
                continue
            sym_panel = sym_panel.sort_values("bar_date").reset_index(drop=True)
            bar_dates = pd.to_datetime(sym_panel["bar_date"]).dt.date.to_numpy()
            announce_dates = sym_earnings["announce_date"].to_numpy()
            surprises = sym_earnings["surprise_pct"].to_numpy()
            surp_means = sym_earnings["surprise_mean_8"].to_numpy()
            surp_stds = sym_earnings["surprise_std_8"].to_numpy()

            n = len(bar_dates)
            days_since = np.full(n, np.nan)
            days_to = np.full(n, np.nan)
            last_surprise = np.full(n, np.nan)
            last_z = np.full(n, np.nan)

            # For each bar_date, find idx of latest announce_date <= bar_date and
            # earliest announce_date > bar_date (np.searchsorted, ascending dates).
            idx_le = np.searchsorted(announce_dates, bar_dates, side="right") - 1
            idx_gt = np.searchsorted(announce_dates, bar_dates, side="right")

            for i in range(n):
                # days_since (last earnings)
                if 0 <= idx_le[i] < len(announce_dates):
                    last_d = announce_dates[idx_le[i]]
                    delta = (bar_dates[i] - last_d).days
                    if 0 <= delta <= _DAYS_SINCE_BOUND:
                        days_since[i] = float(delta)
                    last_surprise_val = surprises[idx_le[i]]
                    if pd.notna(last_surprise_val):
                        last_surprise[i] = float(last_surprise_val)
                        s_mean = surp_means[idx_le[i]]
                        s_std = surp_stds[idx_le[i]]
                        if pd.notna(s_mean) and pd.notna(s_std) and s_std > 0:
                            last_z[i] = float((last_surprise_val - s_mean) / s_std)
                # days_to (next earnings)
                if 0 <= idx_gt[i] < len(announce_dates):
                    next_d = announce_dates[idx_gt[i]]
                    delta = (next_d - bar_dates[i]).days
                    if 0 <= delta <= _DAYS_TO_BOUND:
                        days_to[i] = float(delta)

            sym_panel[f"{self.name}__days_since_last_earnings"] = days_since
            sym_panel[f"{self.name}__days_to_next_earnings"] = days_to
            sym_panel[f"{self.name}__last_surprise_pct"] = last_surprise
            sym_panel[f"{self.name}__last_surprise_z_252"] = last_z
            out_pieces.append(sym_panel)

        if not out_pieces:
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out
        merged = pd.concat(out_pieces, axis=0, ignore_index=True)
        # Re-merge to preserve original (symbol, bar_date) order from the panel.
        out = out.merge(merged, on=["symbol", "bar_date"], how="left")
        return out


# --- Extension registration --------------------------------------------------


class _EarningsExtension(FeatureExtension):
    name = "earnings"
    kind = "panel"

    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        return has_earnings_data(duckdb_path=duckdb_path)

    def make_group(self, *, duckdb_path: str | None = None) -> EarningsFeatures:
        return EarningsFeatures(duckdb_path=duckdb_path)


register_extension(_EarningsExtension())
