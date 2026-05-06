"""CLI: build the technical-feature matrix for a universe and persist to parquet.

    python -m scripts.build_features --universe SP500 --start 2014-01-01 \\
        --end 2024-12-31 --output data/processed/features_sp500.parquet
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from packages.common.config import PROCESSED_DIR
from packages.common.logging import log
from packages.features import build_feature_matrix


@click.command()
@click.option("--universe", default="SP500", show_default=True)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output parquet path. Defaults to data/processed/features_<universe>.parquet.",
)
@click.option(
    "--no-pit", is_flag=True,
    help="Disable point-in-time membership masking (NOT RECOMMENDED — for debugging only).",
)
def main(
    universe: str,
    start: datetime,
    end: datetime,
    output: Path | None,
    no_pit: bool,
) -> None:
    out_path = output or (PROCESSED_DIR / f"features_{universe.lower()}.parquet")
    panel = build_feature_matrix(
        universe,
        start.date(),
        end.date(),
        use_pit_membership=not no_pit,
    )
    if panel.empty:
        log.warning("Feature matrix is empty; nothing to write.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    n_features = sum(1 for c in panel.columns if c not in ("symbol", "bar_date"))
    click.echo(f"Wrote {len(panel):,} rows × {n_features} feature cols → {out_path}")


if __name__ == "__main__":
    main()
