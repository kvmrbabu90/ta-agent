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


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Commodity Channel Index. Mean absolute deviation in the denominator
    keeps it bounded against typical-price moves."""
    typ = (high + low + close) / 3.0
    sma = typ.rolling(period, min_periods=period).mean()
    mad = typ.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (typ - sma) / (0.015 * mad.replace(0.0, np.nan))


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
        rsi_7 = _rsi(close, 7)
        rsi_14 = _rsi(close, 14)
        rsi_21 = _rsi(close, 21)
        rsi_28 = _rsi(close, 28)
        out[f"{self.name}__rsi_7"] = rsi_7.values
        out[f"{self.name}__rsi_14"] = rsi_14.values
        out[f"{self.name}__rsi_21"] = rsi_21.values
        out[f"{self.name}__rsi_28"] = rsi_28.values
        out[f"{self.name}__rsi_overbought"] = (rsi_14 > 70.0).astype(float).values
        out[f"{self.name}__rsi_oversold"] = (rsi_14 < 30.0).astype(float).values

        stoch_k = _stoch_k(close, high, low, 14)
        out[f"{self.name}__stoch_k_14"] = stoch_k.values
        out[f"{self.name}__stoch_d_14"] = (
            stoch_k.rolling(3, min_periods=3).mean().values
        )

        out[f"{self.name}__roc_5"] = ((close / close.shift(5) - 1.0) * 100.0).values
        out[f"{self.name}__roc_20"] = ((close / close.shift(20) - 1.0) * 100.0).values

        out[f"{self.name}__williams_r_14"] = _williams_r(close, high, low, 14).values
        out[f"{self.name}__momentum_quality_60"] = _rolling_corr_with_time(close, 60).values

        # MACD crossings (binary). The cross fires on the bar when sign flips.
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        diff = macd - macd_signal
        diff_prev = diff.shift(1)
        out[f"{self.name}__macd_cross_up"] = ((diff > 0) & (diff_prev <= 0)).astype(float).values
        out[f"{self.name}__macd_cross_down"] = ((diff < 0) & (diff_prev >= 0)).astype(float).values

        out[f"{self.name}__cci_14"] = _cci(high, low, close, 14).values
        out[f"{self.name}__cci_20"] = _cci(high, low, close, 20).values

        return out
