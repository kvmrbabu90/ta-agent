"""Report PIT coverage of the NIFTY 100 change log.

    python -m scripts.audit_nifty_pit

Tells you:
  - how many events the YAML currently has
  - how many distinct semi-annual reconstitution windows are covered
  - the earliest reconstructable date (anything before that is back-projected
    to RELIABLE_START — i.e. survivorship-biased)
  - a list of "open intervals" (members with no add event found) — these are
    the names whose join-date is still unknown
"""

from __future__ import annotations

from datetime import date

import click

from packages.common.logging import log
from packages.ingestion.universe.nifty100_history import RELIABLE_START
from packages.ingestion.universe.nifty100_pit import (
    _load_changes,
    _validate_and_sort,
    build_nifty100_pit_membership,
)


@click.command()
def main() -> None:
    raw = _load_changes()
    events = _validate_and_sort(raw, today=date.today())

    log.info(f"loaded {len(events)} valid change events from YAML")

    if not events:
        click.echo(
            "PIT coverage: 0 events. NIFTY 100 history is currently equivalent "
            "to Phase A (current snapshot only).\n"
            "To improve: transcribe entries from NSE press releases at\n"
            "    https://www.niftyindices.com/reports/historical-data/equity-indices\n"
            "into configs/universes/nifty100_changes.yaml."
        )
        return

    df = build_nifty100_pit_membership()
    open_intervals = df[df["start_date"] == RELIABLE_START]
    closed_intervals = df[df["start_date"] != RELIABLE_START]

    earliest = min(e["effective_date"] for e in events)
    latest = max(e["effective_date"] for e in events)

    click.echo("PIT coverage report — NIFTY 100")
    click.echo("=" * 50)
    click.echo(f"  Total intervals reconstructed   : {len(df):>4d}")
    click.echo(f"  - With known start_date         : {len(closed_intervals):>4d}")
    click.echo(f"  - Back-projected to RELIABLE_START : {len(open_intervals):>4d}")
    click.echo()
    click.echo(f"  Earliest event date             : {earliest}")
    click.echo(f"  Latest event date               : {latest}")
    click.echo()
    click.echo("Symbols with unknown start_date (back-projected):")
    for sym in open_intervals["symbol"].sort_values().tolist()[:30]:
        click.echo(f"    - {sym}")
    if len(open_intervals) > 30:
        click.echo(f"    ... and {len(open_intervals) - 30} more")


if __name__ == "__main__":
    main()
