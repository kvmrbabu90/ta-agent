"""Price-derived features (returns, realized vol, drawdown)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import FeatureGroup


class PriceFeatures(FeatureGroup):
    name = "price"

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        close = ohlcv["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        log_ret_1d = log_close.diff()

        out = pd.DataFrame(index=ohlcv["bar_date"].values)
        out[f"{self.name}__log_return_1d"] = log_ret_1d.values
        out[f"{self.name}__log_return_5d"] = log_close.diff(5).values
        out[f"{self.name}__log_return_10d"] = log_close.diff(10).values
        out[f"{self.name}__log_return_20d"] = log_close.diff(20).values

        # Realized volatility = rolling stdev of 1-day log returns.
        for w in (5, 20, 60):
            out[f"{self.name}__realized_vol_{w}d"] = (
                log_ret_1d.rolling(window=w, min_periods=w).std().values
            )

        # Max drawdown over a trailing 20d window: close / running_max - 1.
        rolling_max = close.rolling(window=20, min_periods=20).max()
        out[f"{self.name}__max_drawdown_20d"] = (close / rolling_max - 1.0).values

        return out
