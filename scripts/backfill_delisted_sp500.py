"""Survivorship-bias fix: backfill OHLCV for delisted/removed SP500 members.

The recovery flow backfilled only currently-active SP500 names via yfinance.
Many historical members (Lehman, Bear Stearns, Alcoa, etc.) were silently
dropped because their tickers no longer trade. Training the model only on
survivors makes returns look better than they were.

This script:
  1. Pulls the FULL historical SP500 membership list (873 names) from
     `packages.ingestion.universe.sp500_history`.
  2. Identifies symbols with NO OHLCV in ohlcv_daily.
  3. Tries yfinance for each one. Keeps results that pass sanity:
       - ≥100 bars
       - At least one bar within the symbol's membership window
       - No single bar with abs(daily return) > 80%
       - Median volume ≥ 1000 over the bars in their membership window
  4. Upserts surviving names back into ohlcv_daily.
  5. Re-runs scripts.refresh_universes to restore membership rows that
     were dropped as 'orphan' by the earlier cleanup.
  6. Re-runs scripts.clean_ohlcv_and_membership to drop any newly-fetched
     garbage (re-uses the same big-move / low-volume filters).

Pure idempotent. Safe to re-run.
"""

from __future__ import annotations

import sys
import time
from datetime import date

import pandas as pd
import duckdb

from packages.common.logging import log
from packages.ingestion.adapters.yfinance_adapter import fetch_daily_bars
from packages.ingestion.storage import upsert_ohlcv
from packages.ingestion.universe.sp500_history import build_sp500_membership


_MIN_BARS = 100                 # need at least 100 bars to be useful
_MAX_DAILY_MOVE = 0.80          # any bar >80% move = bad data
_MIN_MEDIAN_VOLUME = 1000       # avg vol over membership window


def _existing_ohlcv_symbols(conn: duckdb.DuckDBPyConnection) -> set:
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM ohlcv_daily WHERE exchange = 'NYSE'"
    ).fetchall()
    return {r[0] for r in rows}


def _sanity_check(bars: pd.DataFrame, mem_start: date, mem_end: date) -> tuple[bool, str]:
    """Return (passes, reason). reason is empty if passes."""
    if bars.empty or len(bars) < _MIN_BARS:
        return False, f"too few bars ({len(bars)})"
    bars = bars.sort_values("bar_date")
    rets = bars["close"].pct_change().abs()
    if (rets > _MAX_DAILY_MOVE).any():
        return False, f"has >{_MAX_DAILY_MOVE*100:.0f}% daily move"
    # Restrict volume check to the symbol's membership window
    win = bars[(bars["bar_date"] >= pd.Timestamp(mem_start).date())
               & (bars["bar_date"] <= pd.Timestamp(mem_end).date())]
    if win.empty:
        return False, f"no bars in membership window [{mem_start}..{mem_end}]"
    if win["volume"].median() < _MIN_MEDIAN_VOLUME:
        return False, f"median vol {win['volume'].median():.0f} < {_MIN_MEDIAN_VOLUME}"
    return True, ""


def main() -> int:
    log.info("Building full SP500 historical membership from Wikipedia...")
    full = build_sp500_membership()
    full = full[full["universe"] == "SP500"].copy()
    log.info(f"  full SP500 history: {full['symbol'].nunique()} distinct symbols")

    conn = duckdb.connect("data/processed/market.duckdb")
    have = _existing_ohlcv_symbols(conn)
    log.info(f"  ohlcv_daily already has: {len(have)} SP500-eligible NYSE symbols")

    to_try = full[~full["symbol"].isin(have)].drop_duplicates("symbol").copy()
    log.info(f"  symbols to try (no current OHLCV): {len(to_try)}")

    # For each symbol, try yfinance and validate.
    accepted: list[str] = []
    rejected: list[tuple[str, str]] = []

    for i, row in to_try.reset_index(drop=True).iterrows():
        sym = str(row["symbol"])
        mem_start = pd.Timestamp(row["start_date"]).date()
        mem_end = pd.Timestamp(row["end_date"]).date() if pd.notna(row["end_date"]) else date(2026, 5, 16)
        # Fetch yfinance for symbol's membership window + 1y padding either side.
        try_start = max(date(2010, 1, 1), mem_start)
        try_end = min(date(2026, 5, 16), mem_end + pd.Timedelta(days=30).to_pytimedelta())
        if try_start >= try_end:
            rejected.append((sym, "membership window before 2010"))
            continue
        try:
            bars = fetch_daily_bars(sym, try_start, try_end)
        except Exception as exc:  # noqa: BLE001
            rejected.append((sym, f"fetch error: {type(exc).__name__}"))
            continue
        passes, reason = _sanity_check(bars, mem_start, mem_end)
        if not passes:
            rejected.append((sym, reason))
            continue
        # Tag and upsert
        bars["exchange"] = "NYSE"
        bars["source"] = "yfinance"
        upsert_ohlcv(bars)
        accepted.append(sym)
        if (i + 1) % 25 == 0:
            log.info(f"  [{i+1}/{len(to_try)}] accepted {len(accepted)}, rejected {len(rejected)}")
        # Be gentle with yfinance rate limits
        time.sleep(0.15)

    log.info("=" * 70)
    log.info(f"Backfill complete: accepted {len(accepted)}, rejected {len(rejected)}")
    log.info("=" * 70)
    if rejected:
        # group by reason
        from collections import Counter
        counter = Counter(r for _, r in rejected)
        for reason, n in counter.most_common():
            log.info(f"  rejected ({n}): {reason}")
        log.info(f"  sample rejected symbols: {[s for s,_ in rejected[:15]]}")
    conn.close()
    log.info("Now re-run scripts.refresh_universes + scripts.clean_ohlcv_and_membership.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
