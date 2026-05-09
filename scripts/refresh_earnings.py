"""Backfill earnings calendar from yfinance.

    python -m scripts.refresh_earnings --universe SP500

Fetches Ticker.earnings_dates for every current member of ``universe`` and
writes to the ``earnings_calendar`` DuckDB table. Idempotent — safe to re-run.

yfinance Ticker.earnings_dates returns ~25 quarters (~6 years) per ticker,
which covers our recent panel. Pre-2020 rows will lack earnings features
(a NaN-tolerated downside). Upgrade to a paid Finnhub plan and swap the
adapter for ~10 years of clean data when needed.
"""

from __future__ import annotations

import click

from packages.common.logging import log
from packages.ingestion.earnings import backfill_earnings


@click.command()
@click.option("--universe", default="SP500", show_default=True)
def main(universe: str) -> None:
    log.info(f"refreshing earnings for {universe}")
    result = backfill_earnings(universe=universe)
    click.echo(
        f"earnings backfill: requested={result['requested']} "
        f"succeeded={result['succeeded']} failed={result['failed']} "
        f"rows_written={result['rows_written']}"
    )
    if result["failed"]:
        click.echo("first 10 failures:")
        for f in result["failures"][:10]:
            click.echo(f"  {f['symbol']}: {f['error']}")


if __name__ == "__main__":
    main()
