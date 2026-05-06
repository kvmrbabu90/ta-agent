"""Swing-point and Fibonacci features with strict causal confirmation.

Swing detection convention (causal):
    Bar t is a SWING HIGH if high[t] is the maximum over high[t-K..t+K].
    Bar t is a SWING LOW  if low[t]  is the minimum  over low[t-K..t+K].

    The catch: at bar t we do NOT yet know whether t is a swing — we have
    to wait K bars. So a swing at index t only becomes USABLE at bar t+K.
    For any feature evaluated at bar b, we filter the swing set to those
    where (swing_idx + K) <= b. This eliminates look-ahead by construction.

Features (12), with K=5 by default:
    Position (4):
        swings__dist_to_swing_high
        swings__dist_to_swing_low
        swings__swing_high_age
        swings__swing_low_age

    Touch counts over trailing 50 bars (2):
        swings__swing_high_touch_count_50
        swings__swing_low_touch_count_50

    Fibonacci retracement distances (6):
        swings__fib_dist_236, _382, _500, _618, _786
        swings__nearest_fib_dist
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import FeatureGroup

_CONFIRMATION_K = 5
_TOUCH_WINDOW = 50
_TOUCH_TOLERANCE = 0.005  # 0.5%
_FIB_RATIOS: tuple[tuple[str, float], ...] = (
    ("236", 0.236),
    ("382", 0.382),
    ("500", 0.500),
    ("618", 0.618),
    ("786", 0.786),
)


def _detect_swings(
    values: np.ndarray, k: int, kind: str
) -> list[int]:
    """Return indices of confirmed swing highs (kind='high') or lows ('low')."""
    n = len(values)
    out: list[int] = []
    for t in range(k, n - k):
        window = values[t - k : t + k + 1]
        v = values[t]
        if kind == "high":
            if v == window.max():
                out.append(t)
        else:
            if v == window.min():
                out.append(t)
    return out


def _last_le(idxs: list[int], confirmed_at_or_before: int) -> int | None:
    """Largest swing index whose confirmation bar (idx + K) is <= ``confirmed_at_or_before``."""
    if not idxs:
        return None
    # Binary search for the largest idx with idx + K <= confirmed_at_or_before.
    target = confirmed_at_or_before - _CONFIRMATION_K
    # Find largest idx in idxs where idx <= target.
    lo, hi = 0, len(idxs) - 1
    best = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if idxs[mid] <= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return idxs[best] if best >= 0 else None


class SwingFeatures(FeatureGroup):
    name = "swings"

    def __init__(self, k: int = _CONFIRMATION_K) -> None:
        self.k = k

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(ohlcv)
        high = ohlcv["high"].astype(float).values
        low = ohlcv["low"].astype(float).values
        close = ohlcv["close"].astype(float).values
        n = len(close)

        swing_high_idxs = _detect_swings(high, self.k, "high")
        swing_low_idxs = _detect_swings(low, self.k, "low")

        dist_high = np.full(n, np.nan)
        dist_low = np.full(n, np.nan)
        age_high = np.full(n, np.nan)
        age_low = np.full(n, np.nan)
        touch_high = np.full(n, np.nan)
        touch_low = np.full(n, np.nan)
        fib_dist: dict[str, np.ndarray] = {
            label: np.full(n, np.nan) for label, _ in _FIB_RATIOS
        }
        nearest_fib = np.full(n, np.nan)

        for b in range(n):
            sh_idx = _last_le(swing_high_idxs, b)
            sl_idx = _last_le(swing_low_idxs, b)
            c = close[b]
            if c <= 0.0:
                continue

            if sh_idx is not None:
                sh_price = high[sh_idx]
                dist_high[b] = (sh_price - c) / c
                age_high[b] = float(b - sh_idx)
            if sl_idx is not None:
                sl_price = low[sl_idx]
                dist_low[b] = (c - sl_price) / c
                age_low[b] = float(b - sl_idx)

            # Touch counts over trailing window (causal — uses past bars only).
            window_start = max(0, b - _TOUCH_WINDOW + 1)
            if sh_idx is not None and sh_idx <= b:
                sh_price = high[sh_idx]
                if sh_price > 0:
                    high_window = high[window_start : b + 1]
                    touch_high[b] = float(
                        np.sum(np.abs(high_window - sh_price) / sh_price <= _TOUCH_TOLERANCE)
                    )
            if sl_idx is not None and sl_idx <= b:
                sl_price = low[sl_idx]
                if sl_price > 0:
                    low_window = low[window_start : b + 1]
                    touch_low[b] = float(
                        np.sum(np.abs(low_window - sl_price) / sl_price <= _TOUCH_TOLERANCE)
                    )

            # Fibonacci retracements need both a confirmed swing high and low.
            if sh_idx is None or sl_idx is None:
                continue
            sh_price = high[sh_idx]
            sl_price = low[sl_idx]
            if sh_price <= sl_price:
                continue
            range_ = sh_price - sl_price
            min_dist = np.inf
            for label, ratio in _FIB_RATIOS:
                level = sl_price + ratio * range_
                d = (c - level) / c
                fib_dist[label][b] = d
                ad = abs(d)
                if ad < min_dist:
                    min_dist = ad
            if np.isfinite(min_dist):
                nearest_fib[b] = min_dist

        out = pd.DataFrame(index=ohlcv["bar_date"].values)
        out[f"{self.name}__dist_to_swing_high"] = dist_high
        out[f"{self.name}__dist_to_swing_low"] = dist_low
        out[f"{self.name}__swing_high_age"] = age_high
        out[f"{self.name}__swing_low_age"] = age_low
        out[f"{self.name}__swing_high_touch_count_50"] = touch_high
        out[f"{self.name}__swing_low_touch_count_50"] = touch_low
        for label, _ in _FIB_RATIOS:
            out[f"{self.name}__fib_dist_{label}"] = fib_dist[label]
        out[f"{self.name}__nearest_fib_dist"] = nearest_fib

        return out
