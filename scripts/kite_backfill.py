"""CLI entrypoint for the Kite Connect (NSE / NIFTY 100) OHLCV backfill.

Examples:
    python -m scripts.kite_backfill --universe NIFTY100 --start 2014-01-01 --end 2024-12-31
    python -m scripts.kite_backfill --universe NIFTY100 --daily

Requires KITE_API_KEY, KITE_API_SECRET, and a fresh KITE_ACCESS_TOKEN in
``.env``. Tokens expire daily ~6am IST — run ``scripts.kite_login`` to refresh.
"""

from __future__ import annotations

import json
from datetime import datetime

import click

from packages.common.logging import log
from packages.ingestion.adapters.kite_adapter import backfill_universe, daily_update


@click.command()
@click.option(
    "--universe",
    default="NIFTY100",
    show_default=True,
    help="Universe name (e.g. NIFTY100). For US use the IB adapter.",
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

    log.info(
        f"kite_backfill summary: "
        f"{json.dumps({k: v for k, v in result.items() if k != 'failures'})}"
    )
    click.echo(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
