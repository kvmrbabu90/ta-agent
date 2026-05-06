"""Regime features — universe-wide aggregates broadcast to every symbol."""

from __future__ import annotations

import pandas as pd

from packages.features.base import PanelFeatureGroup


class RegimeFeatures(PanelFeatureGroup):
    name = "regime"

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)

        if "price__log_return_1d" not in panel.columns:
            raise ValueError(
                "regime features require price__log_return_1d on the panel"
            )

        out = panel[["symbol", "bar_date"]].copy()

        # Per-date universe aggregates
        per_date = (
            panel.groupby("bar_date", observed=True, sort=True)
            .agg(
                market_return_1d=("price__log_return_1d", "mean"),
                market_breadth_advance_decline=(
                    "price__log_return_1d",
                    lambda s: (s > 0).mean(),
                ),
            )
            .reset_index()
        )

        per_date["market_return_5d"] = per_date["market_return_1d"].rolling(
            5, min_periods=5
        ).sum()
        per_date["market_realized_vol_20d"] = per_date["market_return_1d"].rolling(
            20, min_periods=20
        ).std()

        # Cross-sectional dispersion: stdev of 20d returns across the universe.
        if "price__log_return_20d" in panel.columns:
            dispersion = (
                panel.groupby("bar_date", observed=True, sort=True)["price__log_return_20d"]
                .std()
                .reset_index(name="market_dispersion_20d")
            )
        else:
            dispersion = pd.DataFrame({"bar_date": per_date["bar_date"], "market_dispersion_20d": pd.NA})

        regime_panel = per_date.merge(dispersion, on="bar_date", how="left")

        # Broadcast back to (symbol, bar_date) and prefix.
        merged = out.merge(regime_panel, on="bar_date", how="left")
        rename = {
            c: f"{self.name}__{c}"
            for c in (
                "market_return_1d",
                "market_return_5d",
                "market_breadth_advance_decline",
                "market_realized_vol_20d",
                "market_dispersion_20d",
            )
        }
        merged = merged.rename(columns=rename)

        return merged
