"""Volatility features (ATR, Bollinger, Garman-Klass, Parkinson)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import FeatureGroup


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


class VolatilityFeatures(FeatureGroup):
    name = "volatility"

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        open_ = ohlcv["open"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        close = ohlcv["close"].astype(float)

        out = pd.DataFrame(index=ohlcv["bar_date"].values)

        atr_14 = _atr(high, low, close, 14)
        out[f"{self.name}__atr_14"] = atr_14.values
        out[f"{self.name}__atr_14_pct"] = (atr_14 / close).values

        # Bollinger (20, 2).
        sma20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std()
        upper = sma20 + 2.0 * std20
        lower = sma20 - 2.0 * std20
        width = (upper - lower) / sma20
        pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
        out[f"{self.name}__bollinger_width_20"] = width.values
        out[f"{self.name}__bollinger_pct_b_20"] = pct_b.values

        # Garman-Klass: σ² = 0.5 (ln H/L)² − (2 ln 2 − 1)(ln C/O)², averaged
        # over a 20d window. Reported as annualized-ish stdev (sqrt of mean).
        ln_hl_sq = np.log(high.where(high > 0) / low.where(low > 0)) ** 2
        ln_co_sq = np.log(close.where(close > 0) / open_.where(open_ > 0)) ** 2
        gk_term = 0.5 * ln_hl_sq - (2.0 * np.log(2.0) - 1.0) * ln_co_sq
        gk_var = gk_term.rolling(20, min_periods=20).mean()
        out[f"{self.name}__garman_klass_vol_20"] = np.sqrt(gk_var.clip(lower=0.0)).values

        # Parkinson: σ² = (1/(4 ln 2)) * mean((ln H/L)²) over 20d.
        park_var = (1.0 / (4.0 * np.log(2.0))) * ln_hl_sq.rolling(20, min_periods=20).mean()
        out[f"{self.name}__parkinson_vol_20"] = np.sqrt(park_var.clip(lower=0.0)).values

        return out
