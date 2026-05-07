"""CLI: refresh point-in-time index membership tables.

Usage (from project root, with venv activated):
    python -m scripts.refresh_universes
    python -m scripts.refresh_universes --universe SP500
    python -m scripts.refresh_universes --as-of 2020-01-15 --show
"""

from __future__ import annotations

import click

from packages.common.logging import log
from packages.ingestion.storage import upsert_membership
from packages.ingestion.universe.membership import (
    members_on,
    refresh_all_universes,
)
from packages.ingestion.universe.nifty100_history import build_nifty100_membership
from packages.ingestion.universe.nifty100_pit import build_nifty100_pit_membership
from packages.ingestion.universe.sp500_history import build_sp500_membership


@click.command()
@click.option(
    "--universe",
    type=click.Choice(["SP500", "NIFTY100", "ALL"]),
    default="ALL",
    help="Which universe to refresh",
)
@click.option(
    "--as-of",
    default=None,
    help="Show members as of this date (YYYY-MM-DD) after refresh",
)
@click.option("--show/--no-show", default=False, help="Print members after refresh")
@click.option(
    "--pit-india/--no-pit-india",
    default=False,
    help=(
        "Use PIT reconstruction for NIFTY 100 from "
        "configs/universes/nifty100_changes.yaml (otherwise current snapshot only)."
    ),
)
def main(universe: str, as_of: str | None, show: bool, pit_india: bool) -> None:
    if universe == "ALL":
        counts = refresh_all_universes(pit_india=pit_india)
    elif universe == "SP500":
        df = build_sp500_membership()
        counts = {"SP500": upsert_membership(df)}
    else:
        df = (
            build_nifty100_pit_membership() if pit_india else build_nifty100_membership()
        )
        counts = {"NIFTY100": upsert_membership(df)}

    log.info(f"Wrote: {counts}")

    if show:
        from datetime import date as _date
        target = as_of or str(_date.today())
        for u in counts:
            members = members_on(u, target)
            log.info(f"{u} on {target}: {len(members)} members")
            click.echo(members.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
