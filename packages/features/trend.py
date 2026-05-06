"""Trend features (moving averages, MACD, ADX, distance-from-high)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import FeatureGroup


def _ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. ``adjust=False`` keeps it strictly causal."""
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — same as EMA with alpha = 1/period (causal)."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx_components(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ADX, +DI, -DI) for the given period using Wilder's smoothing."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = _wilder_smooth(dx, period)
    return adx, plus_di, minus_di


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return _adx_components(high, low, close, period)[0]


def _aroon(high: pd.Series, low: pd.Series, period: int = 25) -> tuple[pd.Series, pd.Series]:
    """Aroon Up/Down: how recently the trailing window's extreme occurred.

    aroon_up    = 100 * (period - bars_since_high) / period
    aroon_down  = 100 * (period - bars_since_low)  / period
    """
    def _bars_since_argmax(arr: np.ndarray) -> float:
        return float(period - 1 - int(np.argmax(arr)))

    def _bars_since_argmin(arr: np.ndarray) -> float:
        return float(period - 1 - int(np.argmin(arr)))

    bars_since_high = high.rolling(period, min_periods=period).apply(_bars_since_argmax, raw=True)
    bars_since_low = low.rolling(period, min_periods=period).apply(_bars_since_argmin, raw=True)
    aroon_up = 100.0 * (period - bars_since_high) / period
    aroon_down = 100.0 * (period - bars_since_low) / period
    return aroon_up, aroon_down


class TrendFeatures(FeatureGroup):
    name = "trend"

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)

        out = pd.DataFrame(index=ohlcv["bar_date"].values)

        sma5 = close.rolling(5, min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=20).mean()
        sma50 = close.rolling(50, min_periods=50).mean()
        sma200 = close.rolling(200, min_periods=200).mean()

        out[f"{self.name}__sma_ratio_5_20"] = (sma5 / sma20 - 1.0).values
        out[f"{self.name}__sma_ratio_20_50"] = (sma20 / sma50 - 1.0).values
        out[f"{self.name}__sma_ratio_50_200"] = (sma50 / sma200 - 1.0).values

        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        out[f"{self.name}__ema_ratio_12_26"] = (ema12 / ema26 - 1.0).values

        macd = ema12 - ema26
        macd_signal = _ema(macd, 9)
        out[f"{self.name}__macd"] = macd.values
        out[f"{self.name}__macd_signal"] = macd_signal.values
        out[f"{self.name}__macd_histogram"] = (macd - macd_signal).values

        adx14, plus_di14, minus_di14 = _adx_components(high, low, close, 14)
        out[f"{self.name}__adx_14"] = adx14.values
        out[f"{self.name}__di_plus_14"] = plus_di14.values
        out[f"{self.name}__di_minus_14"] = minus_di14.values
        out[f"{self.name}__di_diff_14"] = (plus_di14 - minus_di14).values

        out[f"{self.name}__price_to_sma200_ratio"] = (close / sma200 - 1.0).values

        rolling_high_252 = high.rolling(window=252, min_periods=252).max()
        out[f"{self.name}__distance_from_52w_high"] = (close / rolling_high_252 - 1.0).values

        # Aroon (25-period). Distinct family from MACD/ADX — measures recency
        # of the rolling-window extremes rather than smoothed price differences.
        aroon_up, aroon_down = _aroon(high, low, 25)
        out[f"{self.name}__aroon_up_25"] = aroon_up.values
        out[f"{self.name}__aroon_down_25"] = aroon_down.values
        out[f"{self.name}__aroon_oscillator_25"] = (aroon_up - aroon_down).values

        return out
