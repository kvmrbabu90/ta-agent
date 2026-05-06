"""Cross-sectional features — ranks and z-scores within universe-on-date.

These rely on the per-symbol features already being on the panel. Pipeline
orders the work so per-symbol groups run first.
"""

from __future__ import annotations

import pandas as pd

from packages.features.base import PanelFeatureGroup

# (input column on panel  ->  output cross-sectional feature)
_RANK_INPUTS: dict[str, str] = {
    "price__log_return_5d": "xs_rank_log_return_5d",
    "price__log_return_20d": "xs_rank_log_return_20d",
    "momentum__rsi_14": "xs_rank_rsi_14",
    "volume__volume_zscore_20": "xs_rank_volume_zscore_20",
    "price__realized_vol_20d": "xs_rank_realized_vol_20d",
}

_ZSCORE_INPUTS: dict[str, str] = {
    "price__log_return_5d": "xs_zscore_log_return_5d",
}


def _xs_rank(series: pd.Series) -> pd.Series:
    """Rank within group, normalized to [0, 1]; NaNs propagate."""
    return series.rank(pct=True)


def _xs_zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([float("nan")] * len(series), index=series.index)
    return (series - mean) / std


class CrossSectionalFeatures(PanelFeatureGroup):
    name = "cross_sectional"

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)
        out = panel[["symbol", "bar_date"]].copy()

        # Compute ranks/z-scores per bar_date — the only legal cross-section.
        # ``transform`` preserves the original index alignment.
        for src_col, out_name in _RANK_INPUTS.items():
            if src_col not in panel.columns:
                out[f"{self.name}__{out_name}"] = pd.NA
                continue
            out[f"{self.name}__{out_name}"] = (
                panel.groupby("bar_date", observed=True, sort=False)[src_col]
                .transform(_xs_rank)
                .values
            )

        for src_col, out_name in _ZSCORE_INPUTS.items():
            if src_col not in panel.columns:
                out[f"{self.name}__{out_name}"] = pd.NA
                continue
            out[f"{self.name}__{out_name}"] = (
                panel.groupby("bar_date", observed=True, sort=False)[src_col]
                .transform(_xs_zscore)
                .values
            )

        return out
