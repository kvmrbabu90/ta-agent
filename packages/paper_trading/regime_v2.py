"""Market regime detector v2 — multi-indicator composite.

Replaces the v1 single-indicator gate (SPY z-score from 50-day SMA),
which was empirically blind to slow bear markets like 2022. The May 2026
2022-stress walk-forward confirmed v1 fired at 1.0× (calm regime) through
the entire 2022 bear, providing zero defense.

Literature consensus (Faber 2007, Whaley 1993, Pollet-Wilson 2010,
AQR drawdown-recovery papers): **simple multi-indicator composites
dominate sophisticated single-model regime detectors** in out-of-sample
performance. Regime change is multi-causal; any single indicator misses
regimes that don't match its signature.

v2 combines FOUR indicators, each emitting a binary trigger:

  1. drawdown_from_52w_peak >= 10%
       AQR/industry standard. 10% is the empirical threshold where
       mean-reversion strategies typically start failing.

  2. spy_close < spy_sma200
       Faber (2007) tactical filter. Below the 200-day SMA marks the
       transition from "bull trend" to "bear trend". Faber demonstrated
       this alone cuts max drawdown ~50% across 100 years of asset
       classes with minimal return drag.

  3. vix >= 25
       Whaley (1993) elevated-volatility threshold. VIX 25 = market
       pricing 25% annualized vol = elevated; 30+ = stressed; 40+ = panic.

  4. cross_sectional_dispersion ANOMALY
       Pollet-Wilson (2010) showed average pairwise correlation among
       constituent stocks rises sharply near regime transitions. Above
       0.6 correlation = compressed dispersion = regime-shift warning.
       (NOT IMPLEMENTED IN v2 — deferred. Requires per-day pairwise
       correlation across the full universe; adds compute cost. Add
       when v2 is validated and we want more sensitivity.)

The composite SCORE is the count of triggered indicators. Map to sizing:

       0 triggers  → calm        → 1.00× slice budget
       1 trigger   → concern     → 0.75×
       2 triggers  → stress      → 0.50×
       3+ triggers → crisis      → 0.25×

We deliberately don't go to 0.0× (full halt) even in crisis. Some trades
will work in any regime; halving the position relative to scaling-down
is enough defense. Yanking everything would also break the strategy's
feature-freshness loop (drift detector etc.).

Validation targets (after the 10-year walk-forward lands):
  2017: must stay calm (low VIX, no DD, SPY > SMA-200)
  2018-Q4: must escalate to stress or crisis as the selloff develops
  2020-Q1 (COVID): must hit crisis fast (VIX > 60, DD > 20%)
  2022: must escalate to at least 'stress' for most of the year
  2024-2026: must mostly stay calm (matches v1's call on this period)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import duckdb

from packages.common.config import settings
from packages.common.logging import log

# Trigger thresholds. Hardcoded — these are designed to be readable
# constants, not tunable parameters. Cite literature in the docstring.
_DRAWDOWN_TRIGGER_PCT = 0.10        # AQR
_SMA200_WINDOW = 200                 # Faber
_DRAWDOWN_LOOKBACK_TRADING_DAYS = 252  # 52 weeks (1 trading year)
_VIX_TRIGGER_LEVEL = 25.0            # Whaley elevated threshold

# Score → (label, multiplier).
_BAND_TABLE = {
    0: ("calm", 1.00),
    1: ("concern", 0.75),
    2: ("stress", 0.50),
    3: ("crisis", 0.25),
    4: ("crisis", 0.25),  # 4 triggers also crisis (when dispersion is added)
}


@dataclass
class RegimeReadingV2:
    """Composite regime state on a given day. Persistable for audit."""
    on_date: date

    # Raw values
    spy_close: float
    spy_sma200: float | None
    spy_52w_high: float | None
    spy_drawdown_pct: float | None      # 0.10 = 10% off peak
    vix_level: float | None

    # Per-indicator booleans
    flag_drawdown: bool
    flag_below_sma200: bool
    flag_high_vix: bool

    score: int
    multiplier: float
    band: str

    def to_dict(self) -> dict:
        return {
            "on_date": self.on_date.isoformat(),
            "spy_close": round(self.spy_close, 2),
            "spy_sma200": round(self.spy_sma200, 2) if self.spy_sma200 else None,
            "spy_52w_high": round(self.spy_52w_high, 2) if self.spy_52w_high else None,
            "spy_drawdown_pct": (
                round(self.spy_drawdown_pct * 100, 2)
                if self.spy_drawdown_pct is not None else None
            ),
            "vix_level": round(self.vix_level, 2) if self.vix_level else None,
            "flag_drawdown": self.flag_drawdown,
            "flag_below_sma200": self.flag_below_sma200,
            "flag_high_vix": self.flag_high_vix,
            "score": self.score,
            "multiplier": self.multiplier,
            "band": self.band,
        }


class RegimeGateV2:
    """Composite multi-indicator regime gate. Drop-in replacement for
    RegimeGate (v1) with the same `multiplier_for(date)` interface but
    a richer underlying signal.

    Loads SPY closes + VIX levels once at construction; reads them with
    binary-search-style lookups in the hot path.
    """

    def __init__(
        self,
        *,
        as_of_max: date,
        duckdb_path: str | None = None,
    ):
        self._spy_closes: dict[date, float] = {}
        self._spy_dates: list[date] = []
        self._vix_levels: dict[date, float] = {}
        self._vix_dates: list[date] = []
        # Always load 1.5y of history before as_of_max so 200-SMA and
        # 52-week-peak computations have enough room.
        self._load(as_of_max, duckdb_path)

    def _load(self, as_of_max: date, duckdb_path: str | None) -> None:
        p = duckdb_path or settings.duckdb_path
        try:
            conn = duckdb.connect(p, read_only=True)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"regime_v2: DuckDB open failed: {exc!r} (gate disabled)")
            return
        try:
            history_floor = as_of_max - timedelta(days=600)  # ~2y warmup
            spy_rows = conn.execute(
                """
                SELECT bar_date, close FROM ohlcv_daily
                WHERE symbol = 'SPY' AND bar_date BETWEEN ? AND ?
                ORDER BY bar_date
                """,
                [history_floor, as_of_max],
            ).fetchall()
            try:
                vix_rows = conn.execute(
                    """
                    SELECT macro_date, value FROM macro_daily
                    WHERE series = 'vix' AND macro_date BETWEEN ? AND ?
                    ORDER BY macro_date
                    """,
                    [history_floor, as_of_max],
                ).fetchall()
            except duckdb.Error:
                # macro_daily schema or VIX missing — gate degrades gracefully:
                # VIX trigger stays at False, other two indicators still work.
                log.warning("regime_v2: VIX not available in macro_daily; gate degrades")
                vix_rows = []
        finally:
            conn.close()

        for d, c in spy_rows:
            if c is None or c <= 0:
                continue
            d_native = d if isinstance(d, date) else date.fromisoformat(str(d))
            self._spy_closes[d_native] = float(c)
        self._spy_dates = sorted(self._spy_closes.keys())

        for d, v in vix_rows:
            if v is None or v <= 0:
                continue
            d_native = d if isinstance(d, date) else date.fromisoformat(str(d))
            self._vix_levels[d_native] = float(v)
        self._vix_dates = sorted(self._vix_levels.keys())

        log.info(
            f"regime_v2: loaded {len(self._spy_dates)} SPY closes + "
            f"{len(self._vix_dates)} VIX levels "
            f"({self._spy_dates[0] if self._spy_dates else None} → "
            f"{self._spy_dates[-1] if self._spy_dates else None})"
        )

    def reading_for(self, on_date: date) -> RegimeReadingV2 | None:
        """Compute the composite regime reading on or before `on_date`."""
        if not self._spy_dates:
            return None
        idx = _bisect_right(self._spy_dates, on_date) - 1
        if idx < _SMA200_WINDOW - 1:
            return None
        anchor = self._spy_dates[idx]
        spy_close = self._spy_closes[anchor]

        # Indicator 1: drawdown from 52-week peak.
        peak_window_start = max(0, idx - _DRAWDOWN_LOOKBACK_TRADING_DAYS + 1)
        peak_window = self._spy_dates[peak_window_start: idx + 1]
        peak = max(self._spy_closes[d] for d in peak_window)
        drawdown_pct = (peak - spy_close) / peak if peak > 0 else None
        flag_drawdown = (
            drawdown_pct is not None and drawdown_pct >= _DRAWDOWN_TRIGGER_PCT
        )

        # Indicator 2: SPY < 200-day SMA.
        sma_window = self._spy_dates[idx - _SMA200_WINDOW + 1: idx + 1]
        sma200 = sum(self._spy_closes[d] for d in sma_window) / len(sma_window)
        flag_below_sma200 = spy_close < sma200

        # Indicator 3: VIX elevated. Find the latest VIX bar on or before
        # anchor; if VIX data is missing, the trigger stays False (graceful
        # degradation).
        vix_level = None
        flag_high_vix = False
        if self._vix_dates:
            vix_idx = _bisect_right(self._vix_dates, anchor) - 1
            if vix_idx >= 0:
                vix_level = self._vix_levels[self._vix_dates[vix_idx]]
                flag_high_vix = vix_level >= _VIX_TRIGGER_LEVEL

        score = int(flag_drawdown) + int(flag_below_sma200) + int(flag_high_vix)
        band, multiplier = _BAND_TABLE.get(score, _BAND_TABLE[3])

        return RegimeReadingV2(
            on_date=anchor,
            spy_close=spy_close,
            spy_sma200=sma200,
            spy_52w_high=peak,
            spy_drawdown_pct=drawdown_pct,
            vix_level=vix_level,
            flag_drawdown=flag_drawdown,
            flag_below_sma200=flag_below_sma200,
            flag_high_vix=flag_high_vix,
            score=score,
            multiplier=multiplier,
            band=band,
        )

    def multiplier_for(self, on_date: date) -> float:
        r = self.reading_for(on_date)
        return r.multiplier if r is not None else 1.0


def _bisect_right(seq: list, target) -> int:
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if target < seq[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


__all__ = ["RegimeGateV2", "RegimeReadingV2"]


# Silence unused-import linter for `math`; reserved for future indicators.
_ = math
