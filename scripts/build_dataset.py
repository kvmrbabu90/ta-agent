"""CLI: assemble the master training dataset (features + labels) for a universe.

    python -m scripts.build_dataset --universe SP500 --start 2014-01-01 \\
        --end 2024-12-31 --horizon 5 --output data/processed/training_sp500.parquet
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from packages.common.config import PROCESSED_DIR
from packages.common.logging import log
from packages.labels.dataset import build_training_dataset


@click.command()
@click.option("--universe", default="SP500", show_default=True)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--horizon", default=5, show_default=True, type=int)
@click.option("--n-classes", default=5, show_default=True, type=int)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
def main(
    universe: str,
    start: datetime,
    end: datetime,
    horizon: int,
    n_classes: int,
    output: Path | None,
) -> None:
    out = (
        output
        or PROCESSED_DIR
        / f"training_{universe.lower()}_{start.date()}_{end.date()}.parquet"
    )
    df = build_training_dataset(
        universe,
        start.date(),
        end.date(),
        horizon_days=horizon,
        n_classes=n_classes,
        output_path=out,
    )
    if df.empty:
        log.warning("dataset is empty")
        return
    n_features = sum(
        1
        for c in df.columns
        if c not in (
            "symbol", "bar_date", f"fwd_return_{horizon}d",
            f"fwd_quintile_{horizon}d", "in_universe",
        )
    )
    click.echo(
        f"Wrote {len(df):,} rows x ({n_features} features + 2 labels + in_universe) -> {out}"
    )


if __name__ == "__main__":
    main()
