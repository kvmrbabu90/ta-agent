"""CLI entrypoint for the Interactive Brokers OHLCV backfill / daily update.

Examples:
    python -m scripts.ib_backfill --universe SP500 --start 2014-01-01 --end 2024-12-31
    python -m scripts.ib_backfill --universe SP500 --daily

Requires TWS or IB Gateway running and accepting API connections on the
host/port configured in ``.env`` (IB_HOST, IB_PORT, IB_CLIENT_ID).
"""

from __future__ import annotations

import json
from datetime import datetime

import click

from packages.common.logging import log
from packages.ingestion.adapters.ib_adapter import backfill_universe, daily_update


@click.command()
@click.option(
    "--universe",
    default="SP500",
    show_default=True,
    help="Universe name (e.g. SP500). For India use the Kite adapter.",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Backfill start date (YYYY-MM-DD). Required unless --daily is set.",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Backfill end date (YYYY-MM-DD). Required unless --daily is set.",
)
@click.option(
    "--daily",
    is_flag=True,
    help="Run an incremental update instead of a full backfill.",
)
def main(
    universe: str,
    start: datetime | None,
    end: datetime | None,
    daily: bool,
) -> None:
    if daily:
        if start or end:
            raise click.UsageError("--start/--end are not compatible with --daily")
        result = daily_update(universe)
    else:
        if not (start and end):
            raise click.UsageError("--start and --end are required without --daily")
        result = backfill_universe(universe, start.date(), end.date())

    log.info(f"ib_backfill summary: {json.dumps({k: v for k, v in result.items() if k != 'failures'})}")
    click.echo(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
