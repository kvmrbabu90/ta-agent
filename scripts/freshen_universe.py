"""Refresh universe membership and backfill any newly-added symbols.

    python -m scripts.freshen_universe --universe SP500

Workflow:
    1. Rebuild membership (Wikipedia for SP500).
    2. Find members that have ZERO bars in ohlcv_daily and backfill them
       through IB.

This is the right tool to run after a quarterly index reconstitution.
"""

from __future__ import annotations

from datetime import date, timedelta

import click

from packages.common.logging import log
from packages.ingestion.storage import get_conn
from packages.ingestion.universe.membership import members_on, refresh_all_universes


def _new_members_without_bars(universe: str, members: list[str]) -> list[str]:
    if not members:
        return []
    placeholders = ",".join("?" * len(members))
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT symbol
            FROM ohlcv_daily
            WHERE symbol IN ({placeholders})
            """,
            list(members),
        ).fetchall()
    have = {r[0] for r in rows}
    return [s for s in members if s not in have]


@click.command()
@click.option(
    "--universe",
    type=click.Choice(["SP500"]),
    required=True,
)
@click.option(
    "--lookback-years",
    default=10,
    show_default=True,
    type=int,
    help="How far back to backfill new symbols.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Refresh membership and report what would be backfilled, but don't actually fetch.",
)
def main(universe: str, lookback_years: int, dry_run: bool) -> None:
    log.info("refreshing membership for all universes")
    refresh_all_universes()

    today = date.today()
    members_df = members_on(universe, today)
    members = members_df["symbol"].tolist()
    log.info(f"{universe}: {len(members)} current members")

    new_syms = _new_members_without_bars(universe, members)
    if not new_syms:
        click.echo(f"No new symbols to backfill for {universe}.")
        return

    click.echo(f"{len(new_syms)} symbol(s) need OHLCV backfill:")
    for s in new_syms:
        click.echo(f"  - {s}")

    if dry_run:
        click.echo("\n--dry-run set; skipping actual backfill.")
        return

    end = today
    start = end - timedelta(days=365 * lookback_years)
    from packages.ingestion.adapters.ib_adapter import backfill_universe

    result = backfill_universe(universe, start, end, symbols=new_syms)
    click.echo(f"\nBackfill summary: {result}")


if __name__ == "__main__":
    main()
