"""Per-bar microstructure features (bar shape, gap)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import FeatureGroup

_EPS = 1e-9


class MicrostructureFeatures(FeatureGroup):
    name = "microstructure"

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        open_ = ohlcv["open"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        close = ohlcv["close"].astype(float)

        out = pd.DataFrame(index=ohlcv["bar_date"].values)

        rng = (high - low).replace(0.0, np.nan)
        out[f"{self.name}__close_location"] = ((close - low) / (rng + _EPS)).values
        out[f"{self.name}__body_pct"] = ((close - open_).abs() / (rng + _EPS)).values

        upper_wick = high - close.where(close > open_, open_)
        lower_wick = close.where(close < open_, open_) - low
        out[f"{self.name}__upper_wick_pct"] = (upper_wick / (rng + _EPS)).values
        out[f"{self.name}__lower_wick_pct"] = (lower_wick / (rng + _EPS)).values
        # Wick balance: ratio of upper to lower wick. Skewed candles indicate
        # rejection from one side. Bounded with +EPS so a zero lower wick
        # produces a finite (large) value rather than inf.
        out[f"{self.name}__wick_ratio"] = (upper_wick / (lower_wick + _EPS)).values

        out[f"{self.name}__gap_pct"] = (open_ / close.shift(1) - 1.0).values

        return out
