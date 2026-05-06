"""Market-structure features — cross-bar pattern signals.

These differ from microstructure (per-bar shape) and from price/trend
(rolling smoothed statistics) by asking a binary question about the relative
position of recent bars: did we just print a higher high? a lower low?

All features are strictly trailing — the rolling extreme is taken over
*prior* bars (``shift(1)`` then rolling-max), never including today.
"""

from __future__ import annotations

import pandas as pd

from packages.features.base import FeatureGroup


def _is_higher_high(high: pd.Series, period: int) -> pd.Series:
    """1 if today's high > the max high over the prior `period` bars."""
    prev_max = high.shift(1).rolling(period, min_periods=period).max()
    return (high > prev_max).astype(float)


def _is_lower_low(low: pd.Series, period: int) -> pd.Series:
    """1 if today's low < the min low over the prior `period` bars."""
    prev_min = low.shift(1).rolling(period, min_periods=period).min()
    return (low < prev_min).astype(float)


class MarketStructureFeatures(FeatureGroup):
    name = "market_structure"

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)

        out = pd.DataFrame(index=ohlcv["bar_date"].values)
        out[f"{self.name}__hh_5"] = _is_higher_high(high, 5).values
        out[f"{self.name}__hh_10"] = _is_higher_high(high, 10).values
        out[f"{self.name}__ll_5"] = _is_lower_low(low, 5).values
        out[f"{self.name}__ll_10"] = _is_lower_low(low, 10).values

        return out
