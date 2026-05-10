"""Backfill the 11 SPDR sector ETFs into ohlcv_daily.

    python -m scripts.refresh_sector_etfs --start 2010-01-01 --end 2026-05-06

These ETFs are needed by SectorResidualFeatures to compute sector-demeaned
stock returns (subtract sector ETF return from stock return for each
(symbol, date) pair).

ETF list (the 11 SPDR Select Sector ETFs):
    XLK  Technology
    XLF  Financials
    XLE  Energy
    XLV  Health Care
    XLY  Consumer Discretionary
    XLP  Consumer Staples
    XLI  Industrials
    XLU  Utilities
    XLRE Real Estate
    XLB  Materials
    XLC  Communication Services (created 2018)

Stored in ohlcv_daily with exchange='NYSE' (where they trade) and
source='yfinance'. The downstream feature group reads them via the standard
get_ohlcv() path.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import click

from packages.common.logging import log
from packages.ingestion.adapters.yfinance_adapter import fetch_daily_bars
from packages.ingestion.storage import get_conn, upsert_ohlcv

SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLY":  "Consumer Cyclical",
    "XLP":  "Consumer Defensive",
    "XLI":  "Industrials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Basic Materials",
    "XLC":  "Communication Services",
}


@click.command()
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default=None, help="ISO date; defaults to today.")
@click.option(
    "--duckdb-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Override DuckDB path.",
)
def main(start: str, end: str | None, duckdb_path: Path | None) -> None:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end) if end else datetime.utcnow().date()
    log.info(f"backfilling {len(SECTOR_ETFS)} sector ETFs over [{s}..{e}]")

    succeeded = failed = rows_written = 0
    failures: list[dict] = []

    with get_conn(str(duckdb_path) if duckdb_path else None) as conn:
        for sym in SECTOR_ETFS:
            try:
                # Treat the ETF as a no-universe symbol; force NYSE exchange.
                df = fetch_daily_bars(sym, s, e, universe="", exchange="NYSE")
                if df.empty:
                    failed += 1
                    failures.append({"symbol": sym, "error": "no bars"})
                    continue
                n = upsert_ohlcv(df, conn=conn)
                rows_written += n
                succeeded += 1
                log.info(f"  {sym}: {n} bars")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append({"symbol": sym, "error": repr(exc)})
                log.error(f"failed {sym}: {exc!r}")

    log.info(
        f"sector ETF backfill complete: succeeded={succeeded} "
        f"failed={failed} rows_written={rows_written}"
    )
    click.echo(f"sector ETF backfill: {succeeded}/{len(SECTOR_ETFS)} ok ({rows_written} rows)")
    if failures:
        for f in failures:
            click.echo(f"  failed {f['symbol']}: {f['error']}")


if __name__ == "__main__":
    main()
