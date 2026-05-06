"""Daily-bar approximation of volume profile (POC, VAH, VAL).

True volume profile is built from intraday tick or minute data — we don't
have that. Approximation: each daily bar's volume is distributed UNIFORMLY
across [low, high]. Sum across the past 50 bars and bin by price. The bin
with maximum volume is the Point of Control (POC); the smallest interval
around it containing 70% of total volume defines Value Area High (VAH) and
Value Area Low (VAL).

This is rough but captures the right qualitative signal: where in the
recent price range did the most trading happen.

Features (all expressed as % distance from current close):
    poc_distance_50        (close - poc) / close
    value_area_high_50     (close - vah) / close
    value_area_low_50      (close - val) / close

Window = 50 bars; warmup = 50 bars; per-bar cost is O(window * n_bins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import FeatureGroup

_WINDOW = 50
_N_BINS = 20
_VALUE_AREA_FRACTION = 0.70


def _value_area(bin_volume: np.ndarray, value_fraction: float) -> tuple[int, int]:
    """Walk outward from POC bin until cumulative volume >= value_fraction * total.
    Returns (lower_bin_idx, upper_bin_idx) inclusive."""
    total = bin_volume.sum()
    if total == 0.0:
        return -1, -1
    target = value_fraction * total
    poc = int(np.argmax(bin_volume))
    lo = hi = poc
    cumulative = bin_volume[poc]
    n = len(bin_volume)
    while cumulative < target and (lo > 0 or hi < n - 1):
        left = bin_volume[lo - 1] if lo > 0 else -np.inf
        right = bin_volume[hi + 1] if hi < n - 1 else -np.inf
        if left >= right and lo > 0:
            lo -= 1
            cumulative += bin_volume[lo]
        elif hi < n - 1:
            hi += 1
            cumulative += bin_volume[hi]
        else:
            break
    return lo, hi


def _profile_for_window(
    high: np.ndarray, low: np.ndarray, volume: np.ndarray, n_bins: int
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Return (poc_price, vah_price, val_price) for the given window."""
    w_high = float(high.max())
    w_low = float(low.min())
    if w_high <= w_low:
        return None, None, None
    edges = np.linspace(w_low, w_high, n_bins + 1)
    bin_vol = np.zeros(n_bins, dtype=float)
    for i in range(len(high)):
        bar_low, bar_high, bar_vol = float(low[i]), float(high[i]), float(volume[i])
        if bar_high <= bar_low or bar_vol <= 0.0:
            continue
        lo_idx = max(0, int(np.searchsorted(edges, bar_low, side="right") - 1))
        hi_idx = min(n_bins - 1, int(np.searchsorted(edges, bar_high, side="left")))
        n_spanned = max(1, hi_idx - lo_idx + 1)
        per_bin = bar_vol / n_spanned
        bin_vol[lo_idx : hi_idx + 1] += per_bin
    poc_idx = int(np.argmax(bin_vol))
    poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0
    val_idx, vah_idx = _value_area(bin_vol, _VALUE_AREA_FRACTION)
    if val_idx < 0:
        return poc_price, None, None
    val_price = float(edges[val_idx])
    vah_price = float(edges[vah_idx + 1])
    return poc_price, vah_price, val_price


class VolumeProfileFeatures(FeatureGroup):
    name = "volume_profile"

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        high = ohlcv["high"].astype(float).values
        low = ohlcv["low"].astype(float).values
        close = ohlcv["close"].astype(float).values
        volume = ohlcv["volume"].astype(float).values
        n = len(close)

        poc_dist = np.full(n, np.nan)
        vah_dist = np.full(n, np.nan)
        val_dist = np.full(n, np.nan)

        for t in range(_WINDOW - 1, n):
            sl = slice(t - _WINDOW + 1, t + 1)
            poc, vah, val = _profile_for_window(
                high[sl], low[sl], volume[sl], _N_BINS
            )
            c = close[t]
            if c <= 0.0 or poc is None:
                continue
            poc_dist[t] = (c - poc) / c
            if vah is not None:
                vah_dist[t] = (c - vah) / c
            if val is not None:
                val_dist[t] = (c - val) / c

        out = pd.DataFrame(index=ohlcv["bar_date"].values)
        out[f"{self.name}__poc_distance_50"] = poc_dist
        out[f"{self.name}__value_area_high_50"] = vah_dist
        out[f"{self.name}__value_area_low_50"] = val_dist
        return out
