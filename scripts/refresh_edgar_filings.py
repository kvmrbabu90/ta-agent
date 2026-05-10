"""Backfill SEC EDGAR filings for the SP500 universe.

    python -m scripts.refresh_edgar_filings --universe SP500

Fetches the recent ~1000 filings for each current member of ``universe`` from
data.sec.gov and writes to the edgar_filings DuckDB table. Idempotent — safe
to re-run. Free, no API key required (just a polite User-Agent).
"""

from __future__ import annotations

import click

from packages.common.logging import log
from packages.ingestion.sec_filings import backfill_filings


@click.command()
@click.option("--universe", default="SP500", show_default=True)
def main(universe: str) -> None:
    log.info(f"refreshing EDGAR filings for {universe}")
    result = backfill_filings(universe=universe)
    click.echo(
        f"edgar backfill: requested={result['requested']} "
        f"succeeded={result['succeeded']} failed={result['failed']} "
        f"rows_written={result['rows_written']}"
    )
    if result["failed"]:
        click.echo("first 10 failures:")
        for f in result["failures"][:10]:
            click.echo(f"  {f['symbol']}: {f['error']}")


if __name__ == "__main__":
    main()
