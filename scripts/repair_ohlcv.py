"""Repair corrupted OHLCV bars by re-fetching from yfinance.

A long-standing bug in yfinance_adapter.daily_update meant the morning
8:35 CT pipeline tick could write partial-day bars (open from a prior
snapshot, high < open, etc.) that subsequent runs never re-fetched
(start=last+1 logic). This script identifies bars violating OHLC
invariants and refetches them from yfinance with end-of-day data.

Usage:
    python -m scripts.repair_ohlcv                       # all bad bars in 2026
    python -m scripts.repair_ohlcv --since 2026-05-01    # since a date
    python -m scripts.repair_ohlcv --dry-run             # report only

Run AFTER 17:00 ET / 16:00 CT so end-of-day bars are available from yfinance.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from collections import defaultdict

import duckdb

from packages.common.logging import log
from packages.ingestion.adapters.yfinance_adapter import fetch_daily_bars
from packages.ingestion.storage import get_conn, upsert_ohlcv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2026-01-01",
                   help="ISO date; repair bars on/after this. Default: 2026-01-01")
    p.add_argument("--dry-run", action="store_true",
                   help="Identify + log bad bars without refetching")
    args = p.parse_args()
    since = date.fromisoformat(args.since)

    log.info(f"repair_ohlcv: scanning ohlcv_daily for invariant violations since {since}")
    with get_conn() as duck:
        rows = duck.execute(
            """
            SELECT symbol, exchange, bar_date
            FROM ohlcv_daily
            WHERE bar_date >= ?
              AND (high < open OR high < close OR high < low
                   OR low > open OR low > close)
            ORDER BY bar_date, symbol
            """,
            [since],
        ).fetchall()

    if not rows:
        log.info("repair_ohlcv: no bars with invariant violations — nothing to do")
        return 0

    log.info(f"repair_ohlcv: found {len(rows):,} bad bars to repair")

    # Bucket by (symbol, exchange) so we refetch each symbol's bad-bar window
    # in one yfinance call.
    by_sym: dict[tuple[str, str], list[date]] = defaultdict(list)
    for sym, exch, d in rows:
        by_sym[(sym, exch)].append(d)

    if args.dry_run:
        for (sym, exch), dates in sorted(by_sym.items()):
            log.info(f"  {sym}:{exch}  {len(dates)} bad bars: {dates[:5]}{'...' if len(dates) > 5 else ''}")
        return 0

    repaired = failed = 0
    with get_conn() as duck:
        for (sym, exch), dates in sorted(by_sym.items()):
            start = min(dates)
            end = max(dates)
            try:
                df = fetch_daily_bars(sym, start, end, universe="SP500", exchange=exch)
                if df.empty:
                    log.warning(f"  {sym}: yfinance returned no bars for {start}..{end}")
                    failed += 1
                    continue
                n = upsert_ohlcv(df, conn=duck)
                log.info(f"  {sym}: refetched {n} bars for window {start}..{end}")
                repaired += n
            except Exception as e:
                log.error(f"  {sym}: repair failed: {e!r}")
                failed += 1

    log.info(f"repair_ohlcv: done. rows_written={repaired} failed_symbols={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
