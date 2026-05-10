"""Backfill SEC EDGAR fundamentals for the SP500 universe.

    python -m scripts.refresh_sec_fundamentals --universe SP500

Pulls quarterly + annual XBRL fact data from data.sec.gov/api/xbrl/companyfacts.
Free, no API key required (just a polite User-Agent).
"""

from __future__ import annotations

import click

from packages.common.logging import log
from packages.ingestion.sec_fundamentals import backfill_fundamentals


@click.command()
@click.option("--universe", default="SP500", show_default=True)
def main(universe: str) -> None:
    log.info(f"refreshing SEC fundamentals for {universe}")
    result = backfill_fundamentals(universe=universe)
    click.echo(
        f"sec_fundamentals backfill: requested={result['requested']} "
        f"succeeded={result['succeeded']} failed={result['failed']} "
        f"rows_written={result['rows_written']}"
    )
    if result["failed"]:
        click.echo("first 10 failures:")
        for f in result["failures"][:10]:
            click.echo(f"  {f['symbol']}: {f['error']}")


if __name__ == "__main__":
    main()
