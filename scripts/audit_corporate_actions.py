"""CLI: audit a universe for inter-source price disagreements.

    python -m scripts.audit_corporate_actions --universe SP500 --lookback 365
"""

from __future__ import annotations

import click

from packages.ingestion.corporate_actions import audit_universe


@click.command()
@click.option("--universe", default="SP500", show_default=True)
@click.option(
    "--lookback",
    type=int,
    default=365,
    show_default=True,
    help="Days of history to inspect.",
)
@click.option(
    "--tolerance",
    type=float,
    default=1.0,
    show_default=True,
    help="Percent close-price disagreement above which to flag a row.",
)
@click.option(
    "--top",
    type=int,
    default=20,
    show_default=True,
    help="Show the top-N most-disagreeing symbols.",
)
def main(universe: str, lookback: int, tolerance: float, top: int) -> None:
    df = audit_universe(universe, lookback_days=lookback, tolerance_pct=tolerance)
    if df.empty:
        click.echo(f"No disagreements > {tolerance}% over the last {lookback} days.")
        return
    click.echo(f"Top {min(top, len(df))} disagreeing symbols in {universe}:")
    click.echo(df.head(top).to_string(index=False))


if __name__ == "__main__":
    main()
