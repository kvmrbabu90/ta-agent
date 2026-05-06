"""CLI: refresh the macro_daily table (VIX, USD/INR, ...).

    python -m scripts.refresh_macro --start 2014-01-01 --end 2024-12-31
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import click

from packages.ingestion.macro import backfill_macro


@click.command()
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Backfill start (default: 10 years before today).",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Backfill end (default: today).",
)
def main(start: datetime | None, end: datetime | None) -> None:
    end_d = end.date() if end else date.today()
    start_d = start.date() if start else end_d - timedelta(days=365 * 10)
    result = backfill_macro(start_d, end_d)
    click.echo(result)


if __name__ == "__main__":
    main()
