"""Momentum features (RSI, Stochastic, Williams %R, ROC, momentum quality)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import FeatureGroup


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI. Strictly causal (uses past gains/losses only)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    h = high.rolling(period, min_periods=period).max()
    lo = low.rolling(period, min_periods=period).min()
    return 100.0 * (close - lo) / (h - lo).replace(0.0, np.nan)


def _williams_r(close: pd.Series, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    h = high.rolling(period, min_periods=period).max()
    lo = low.rolling(period, min_periods=period).min()
    return -100.0 * (h - close) / (h - lo).replace(0.0, np.nan)


def _rolling_corr_with_time(s: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between ``s`` and an integer time index.

    High positive = price climbed monotonically over the window; near zero =
    no trend; high negative = monotonic decline. A measure of trend quality.
    """
    t = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    return s.rolling(window=window, min_periods=window).corr(t)


class MomentumFeatures(FeatureGroup):
    name = "momentum"

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)

        out = pd.DataFrame(index=ohlcv["bar_date"].values)
        out[f"{self.name}__rsi_14"] = _rsi(close, 14).values
        out[f"{self.name}__rsi_28"] = _rsi(close, 28).values

        stoch_k = _stoch_k(close, high, low, 14)
        out[f"{self.name}__stoch_k_14"] = stoch_k.values
        out[f"{self.name}__stoch_d_14"] = (
            stoch_k.rolling(3, min_periods=3).mean().values
        )

        out[f"{self.name}__roc_5"] = ((close / close.shift(5) - 1.0) * 100.0).values
        out[f"{self.name}__roc_20"] = ((close / close.shift(20) - 1.0) * 100.0).values

        out[f"{self.name}__williams_r_14"] = _williams_r(close, high, low, 14).values
        out[f"{self.name}__momentum_quality_60"] = _rolling_corr_with_time(close, 60).values

        return out
