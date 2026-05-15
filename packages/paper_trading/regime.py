"""Market regime detector — gates new entries based on SPY's distance
from its 50-day moving average.

Why this exists:
    Mean-reversion strategies are regime-dependent. They work in choppy
    sideways markets and break when the broad market is trending hard
    (either direction). Buying losers in a strong-trend down market just
    keeps catching falling knives; shorting winners in a strong-trend up
    market gets squeezed.

Mechanics:
    1. Compute SPY's distance from its 50-day SMA, normalized by 20-day
       realized volatility of SPY daily returns. Result is a z-score.
    2. Apply a sizing multiplier based on |z|:
         |z| < 1.0    →  1.0× (normal regime, sideways/mild trend)
         |z| < 1.5    →  0.75× (mild trending, scale down)
         |z| < 2.0    →  0.5×  (clear trend, half size)
         |z| >= 2.0   →  0.25× (strong trend, quarter size)

    The multiplier scales the SLICE BUDGET in the engine, so existing
    positions aren't liquidated — only new lots get smaller. This is
    deliberate: yanking positions on a regime flip would whipsaw worse
    than the regime change itself.

    Falls back to 1.0× (no scaling) when SPY data is missing.

Scope:
    Audit telemetry is persisted alongside the run so we can replay the
    regime call after the fact. We do NOT auto-disable the strategy in
    extreme regimes — half-size is the most aggressive scale-down, on the
    grounds that conviction-weighted sizing already starves bad signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import duckdb

from packages.common.config import settings
from packages.common.logging import log

# Multiplier band thresholds (|z| breakpoints).
_BAND_THRESHOLDS = [
    (1.0, 1.00, "normal"),
    (1.5, 0.75, "mild_trend"),
    (2.0, 0.50, "trend"),
    (math.inf, 0.25, "strong_trend"),
]

_SMA_WINDOW = 50
_VOL_WINDOW = 20


@dataclass
class RegimeReading:
    """Regime state on a given day. Persisted in paper_runs for audit."""
    on_date: date
    spy_close: float
    spy_sma50: float
    spy_distance: float       # close - sma50 (raw $)
    spy_vol_20d: float        # std of daily log returns over last 20 bars
    z_score: float            # spy_distance / (spy_close × spy_vol_20d)
    multiplier: float
    band: str

    def to_dict(self) -> dict:
        return {
            "on_date": self.on_date.isoformat(),
            "spy_close": round(self.spy_close, 2),
            "spy_sma50": round(self.spy_sma50, 2),
            "spy_distance": round(self.spy_distance, 2),
            "spy_vol_20d": round(self.spy_vol_20d, 4),
            "z_score": round(self.z_score, 2),
            "multiplier": self.multiplier,
            "band": self.band,
        }


class RegimeGate:
    """Cached SPY-based regime gate.

    Build once per backtest (loads ~1y of SPY closes), then call
    `multiplier_for(d)` per trading day. The fetch is read-only so
    it doesn't conflict with the API or the writer pipeline.
    """

    def __init__(
        self,
        *,
        as_of_max: date,
        history_days: int = 400,
        duckdb_path: str | None = None,
    ):
        self._closes_by_date: dict[date, float] = {}
        self._sorted_dates: list[date] = []
        self._history_days = history_days
        self._load(as_of_max, duckdb_path)

    def _load(self, as_of_max: date, duckdb_path: str | None) -> None:
        p = duckdb_path or settings.duckdb_path
        try:
            conn = duckdb.connect(p, read_only=True)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"regime: SPY DuckDB open failed: {exc!r} (gate disabled)")
            return
        try:
            rows = conn.execute(
                """
                SELECT bar_date, close FROM ohlcv_daily
                WHERE symbol = 'SPY' AND bar_date <= ?
                  AND bar_date >= ?
                ORDER BY bar_date
                """,
                [as_of_max, as_of_max - timedelta(days=self._history_days)],
            ).fetchall()
        finally:
            conn.close()
        for d, c in rows:
            if c is None or c <= 0:
                continue
            d_native = d if isinstance(d, date) else date.fromisoformat(str(d))
            self._closes_by_date[d_native] = float(c)
        self._sorted_dates = sorted(self._closes_by_date.keys())
        if not self._sorted_dates:
            log.warning("regime: no SPY closes loaded (gate disabled — defaults to 1.0×)")
        else:
            log.info(
                f"regime: loaded {len(self._sorted_dates)} SPY closes "
                f"[{self._sorted_dates[0]} .. {self._sorted_dates[-1]}]"
            )

    def reading_for(self, on_date: date) -> RegimeReading | None:
        """Return the regime reading on or before `on_date`. None if not enough data."""
        if not self._sorted_dates:
            return None
        # Pick the latest SPY bar at or before on_date.
        idx = _bisect_right(self._sorted_dates, on_date) - 1
        if idx < _SMA_WINDOW:
            # Not enough history for a 50-day SMA.
            return None
        anchor = self._sorted_dates[idx]
        recent_dates = self._sorted_dates[idx - _SMA_WINDOW + 1 : idx + 1]
        recent_closes = [self._closes_by_date[d] for d in recent_dates]
        sma = sum(recent_closes) / len(recent_closes)
        close = recent_closes[-1]
        # 20-day realized vol of daily log returns.
        if idx < _VOL_WINDOW:
            return None
        vol_dates = self._sorted_dates[idx - _VOL_WINDOW : idx + 1]
        vol_closes = [self._closes_by_date[d] for d in vol_dates]
        log_rets = [
            math.log(vol_closes[k] / vol_closes[k - 1])
            for k in range(1, len(vol_closes))
            if vol_closes[k - 1] > 0
        ]
        if len(log_rets) < 2:
            return None
        mean = sum(log_rets) / len(log_rets)
        var = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        distance = close - sma
        # Z-score: # of "expected drift std deviations" the price is from
        # the SMA. The expected diffusion scale over an SMA-window of N
        # days is daily_vol × sqrt(N), so the natural denominator is
        #   close × daily_vol × sqrt(SMA_WINDOW)
        # NOT just close × daily_vol — the latter is the 1-day move scale,
        # which makes any decent uptrend look like a 5-10σ event. With
        # the sqrt(N) correction, |z|>1 means the current price is more
        # than one full SMA-window-worth of typical drift away from the
        # average, which is a meaningful regime signal.
        denom = max(close * sd * math.sqrt(_SMA_WINDOW), 1e-9)
        z = distance / denom
        mult, band = _multiplier_for_z(abs(z))
        return RegimeReading(
            on_date=anchor, spy_close=close, spy_sma50=sma,
            spy_distance=distance, spy_vol_20d=sd, z_score=z,
            multiplier=mult, band=band,
        )

    def multiplier_for(self, on_date: date) -> float:
        r = self.reading_for(on_date)
        return r.multiplier if r is not None else 1.0


def _multiplier_for_z(abs_z: float) -> tuple[float, str]:
    for thresh, mult, band in _BAND_THRESHOLDS:
        if abs_z < thresh:
            return mult, band
    return _BAND_THRESHOLDS[-1][1], _BAND_THRESHOLDS[-1][2]


def _bisect_right(seq: list, target) -> int:
    """Stdlib `bisect.bisect_right` analog avoiding the import for 4 lines."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if target < seq[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


__all__ = ["RegimeGate", "RegimeReading"]
