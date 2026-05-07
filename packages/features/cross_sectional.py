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


class CrossSectionalFeatures(PanelFeatureGroup):
    name = "cross_sectional"

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)
        out = panel[["symbol", "bar_date"]].copy()

        # Use pandas' built-in groupby methods rather than UDFs through
        # ``.transform(func)`` — the UDF path has a known shape-handling
        # edge case (`'Series' object has no attribute 'columns'`) that
        # surfaces on small universes / sparse groups, and the built-ins
        # are also significantly faster.
        for src_col, out_name in _RANK_INPUTS.items():
            if src_col not in panel.columns:
                out[f"{self.name}__{out_name}"] = pd.NA
                continue
            out[f"{self.name}__{out_name}"] = (
                panel.groupby("bar_date", observed=True, sort=False)[src_col]
                .rank(pct=True)
                .values
            )

        for src_col, out_name in _ZSCORE_INPUTS.items():
            if src_col not in panel.columns:
                out[f"{self.name}__{out_name}"] = pd.NA
                continue
            grp = panel.groupby("bar_date", observed=True, sort=False)[src_col]
            mean = grp.transform("mean")
            std = grp.transform("std")
            zscore = (panel[src_col] - mean) / std.where(std != 0)
            out[f"{self.name}__{out_name}"] = zscore.values

        return out
