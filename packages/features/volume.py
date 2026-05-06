"""Volume features (z-score, OBV slope, VWAP deviation, dollar-volume z-score, trend)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import FeatureGroup


def _zscore_rolling(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std()
    return (s - mean) / std.replace(0.0, np.nan)


def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """Slope of the linear regression of ``y`` on time over a trailing window."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    sx2 = ((x - x_mean) ** 2).sum()

    def _slope(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        return ((x - x_mean) * (arr - arr.mean())).sum() / sx2

    return y.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """Money Flow Index — volume-weighted RSI on the typical-price flow."""
    typ = (high + low + close) / 3.0
    money_flow = typ * volume
    direction = np.sign(typ.diff())
    pos_flow = money_flow.where(direction > 0, 0.0)
    neg_flow = money_flow.where(direction < 0, 0.0)
    pos_sum = pos_flow.rolling(period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(period, min_periods=period).sum()
    mfr = pos_sum / neg_sum.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + mfr))


def _cmf(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """Chaikin Money Flow — volume-weighted close-location, summed over period."""
    money_flow_mult = ((close - low) - (high - close)) / (high - low).replace(0.0, np.nan)
    money_flow_vol = money_flow_mult * volume
    return (
        money_flow_vol.rolling(period, min_periods=period).sum()
        / volume.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    )


def _force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Elder's Force Index = EMA(close.diff() * volume, period)."""
    raw = close.diff() * volume
    return raw.ewm(span=period, adjust=False, min_periods=period).mean()


class VolumeFeatures(FeatureGroup):
    name = "volume"

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        close = ohlcv["close"].astype(float)
        volume = ohlcv["volume"].astype(float)

        out = pd.DataFrame(index=ohlcv["bar_date"].values)

        out[f"{self.name}__volume_zscore_20"] = _zscore_rolling(volume, 20).values

        # OBV: cumulative sum of signed volume, where sign is the direction
        # of close vs the prior close. Then take its 20d regression slope.
        sign = np.sign(close.diff().fillna(0.0))
        obv = (sign * volume).cumsum()
        out[f"{self.name}__obv_slope_20"] = _rolling_slope(obv, 20).values
        out[f"{self.name}__obv_zscore_20"] = _zscore_rolling(obv, 20).values

        # 20-day VWAP and current close's deviation from it.
        dollar_vol = close * volume
        vwap20 = (
            dollar_vol.rolling(20, min_periods=20).sum()
            / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan)
        )
        out[f"{self.name}__vwap_deviation_20"] = (close / vwap20 - 1.0).values
        out[f"{self.name}__vwap_above"] = (close > vwap20).astype(float).values

        out[f"{self.name}__dollar_volume_zscore_20"] = _zscore_rolling(dollar_vol, 20).values

        # Volume trend: rolling correlation of log(volume) with time over 60d.
        log_volume = np.log(volume.where(volume > 0))
        t = pd.Series(np.arange(len(log_volume), dtype=float), index=log_volume.index)
        out[f"{self.name}__volume_trend_60"] = (
            log_volume.rolling(60, min_periods=60).corr(t).values
        )

        # Money-flow / pressure family.
        out[f"{self.name}__mfi_14"] = _mfi(high, low, close, volume, 14).values
        out[f"{self.name}__cmf_20"] = _cmf(high, low, close, volume, 20).values
        out[f"{self.name}__force_index_13"] = _force_index(close, volume, 13).values

        return out
