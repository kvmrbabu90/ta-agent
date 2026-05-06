"""Macro features broadcast to (symbol, bar_date) panel.

Reads from the ``macro_daily`` table. If no macro data is present, the
feature group should not be registered (see pipeline conditional logic);
when it is registered with empty data, all features come back NaN.

Features:
    macro__vix_level_z_252  — z-score of VIX close vs. trailing 252-day window
                              (excluding today, so causal)
    macro__vix_chg_5d       — 5-day change in VIX (absolute, not %)
    macro__fx_ret_5d        — 5-day log return of USD/INR
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import PanelFeatureGroup
from packages.ingestion.macro import load_macro_series


def _trailing_zscore(s: pd.Series, window: int) -> pd.Series:
    """Causal z-score: exclude the current observation when computing
    the rolling mean and std so today's value isn't in its own baseline."""
    prior = s.shift(1)
    mean = prior.rolling(window=window, min_periods=max(20, window // 4)).mean()
    std = prior.rolling(window=window, min_periods=max(20, window // 4)).std()
    return (s - mean) / std.replace(0.0, np.nan)


class MacroFeatures(PanelFeatureGroup):
    name = "macro"

    def __init__(self, duckdb_path: str | None = None) -> None:
        self._duckdb_path = duckdb_path

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)
        out = panel[["symbol", "bar_date"]].copy()

        if panel.empty:
            for col in ("vix_level_z_252", "vix_chg_5d", "fx_ret_5d"):
                out[f"{self.name}__{col}"] = pd.NA
            return out

        # Window we need from macro_daily: full panel range plus 252-day warmup.
        bar_dates = pd.to_datetime(panel["bar_date"]).dt.date
        start = bar_dates.min() - pd.Timedelta(days=400)
        end = bar_dates.max()

        vix = load_macro_series("vix", start, end, duckdb_path=self._duckdb_path)
        fx = load_macro_series("usd_inr", start, end, duckdb_path=self._duckdb_path)

        # Build a per-date macro frame indexed by datetime.
        idx = pd.DatetimeIndex(sorted(set(vix.index) | set(fx.index)))
        macro = pd.DataFrame(index=idx)
        if not vix.empty:
            macro["vix_level_z_252"] = _trailing_zscore(vix.reindex(idx).ffill(), 252).values
            macro["vix_chg_5d"] = vix.reindex(idx).ffill().diff(5).values
        else:
            macro["vix_level_z_252"] = np.nan
            macro["vix_chg_5d"] = np.nan
        if not fx.empty:
            log_fx = np.log(fx.reindex(idx).ffill().where(lambda s: s > 0))
            macro["fx_ret_5d"] = log_fx.diff(5).values
        else:
            macro["fx_ret_5d"] = np.nan

        macro = macro.reset_index().rename(columns={"index": "bar_date"})
        # Forward-fill macro values across stock-holiday gaps before merge.
        macro = macro.sort_values("bar_date").ffill()

        # Coerce both sides to datetime64 to dodge merge dtype mismatches when
        # the panel uses datetime64[us] but macro_daily was stored as DATE.
        macro["bar_date"] = pd.to_datetime(macro["bar_date"])
        out_join = out.copy()
        out_join["bar_date"] = pd.to_datetime(out_join["bar_date"])

        merged = out_join.merge(macro, on="bar_date", how="left")
        merged["bar_date"] = out["bar_date"].values  # restore original dtype
        rename = {
            c: f"{self.name}__{c}"
            for c in ("vix_level_z_252", "vix_chg_5d", "fx_ret_5d")
        }
        return merged.rename(columns=rename)
