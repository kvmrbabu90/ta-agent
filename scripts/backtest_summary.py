"""Summarize historical predictions in the predictions log.

    python -m scripts.backtest_summary --universe SP500 --start 2024-01-01

Outputs a markdown report (overall stats + monthly breakdown) to stdout.
This reads ``predictions_log`` only — it does NOT re-run the model.
"""

from __future__ import annotations

from datetime import date

import click
import numpy as np
import pandas as pd

from packages.inference.db import get_sqlite_conn
from packages.modeling.evaluate import (
    decile_spread,
    information_coefficient,
    rank_information_coefficient,
)


def _load_settled(universe: str, start: date, end: date) -> pd.DataFrame:
    with get_sqlite_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT as_of, symbol, predicted_return, realized_return,
                   predicted_quintile, realized_quintile,
                   top_quintile_proba
            FROM predictions_log
            WHERE universe = ?
              AND realized_return IS NOT NULL
              AND as_of BETWEEN ? AND ?
            ORDER BY as_of, symbol
            """,
            conn,
            params=[universe, start.isoformat(), end.isoformat()],
        )
    if not df.empty:
        df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    return df


def _per_date_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, group in df.groupby("as_of"):
        if len(group) < 5:
            continue
        ic = information_coefficient(
            group["realized_return"].to_numpy(), group["predicted_return"].to_numpy()
        )
        ric = rank_information_coefficient(
            group["realized_return"].to_numpy(), group["predicted_return"].to_numpy()
        )
        if np.isnan(ic):
            continue
        rows.append({"as_of": d, "ic": ic, "rank_ic": ric, "n": len(group)})
    return pd.DataFrame(rows)


def _summary_block(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"### {label}\n\n_No settled predictions in this window._\n"
    daily = _per_date_stats(df)
    if daily.empty:
        return f"### {label}\n\n_Not enough cross-sectional rows per date for IC._\n"

    mean_ic = float(daily["ic"].mean())
    std_ic = float(daily["ic"].std(ddof=1)) if len(daily) > 1 else float("nan")
    n_dates = len(daily)
    t_stat = mean_ic / (std_ic / np.sqrt(n_dates)) if std_ic and n_dates > 1 else float("nan")
    mean_rank_ic = float(daily["rank_ic"].mean())
    hit = float(
        (np.sign(df["realized_return"]) == np.sign(df["predicted_return"])).mean()
    )
    spread = decile_spread(
        df["realized_return"].to_numpy(), df["predicted_return"].to_numpy()
    )

    return (
        f"### {label}\n\n"
        f"| metric | value |\n"
        f"| --- | --- |\n"
        f"| n dates | {n_dates} |\n"
        f"| n predictions | {len(df):,} |\n"
        f"| mean daily IC | {mean_ic:.4f} |\n"
        f"| std daily IC | {std_ic:.4f} |\n"
        f"| IC t-stat | {t_stat:.2f} |\n"
        f"| mean daily rank IC | {mean_rank_ic:.4f} |\n"
        f"| hit rate | {hit:.2%} |\n"
        f"| decile spread (5d) | {spread:.4f} |\n"
    )


def _monthly_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["as_of"]).dt.to_period("M").astype(str)
    parts = []
    for ym, group in df.groupby("year_month"):
        daily = _per_date_stats(group)
        if daily.empty:
            continue
        mean_ic = float(daily["ic"].mean())
        hit = float(
            (np.sign(group["realized_return"]) == np.sign(group["predicted_return"])).mean()
        )
        spread = decile_spread(
            group["realized_return"].to_numpy(), group["predicted_return"].to_numpy()
        )
        parts.append({"month": ym, "n_dates": len(daily), "n": len(group),
                      "mean_ic": mean_ic, "hit_rate": hit, "decile_spread": spread})
    if not parts:
        return "_No months had enough cross-sectional coverage to compute IC._"
    monthly = pd.DataFrame(parts)
    lines = ["| month | n dates | n preds | mean IC | hit rate | decile spread |",
             "| --- | --- | --- | --- | --- | --- |"]
    for _, r in monthly.iterrows():
        lines.append(
            f"| {r['month']} | {r['n_dates']} | {int(r['n']):,} | {r['mean_ic']:.4f} | "
            f"{r['hit_rate']:.2%} | {r['decile_spread']:.4f} |"
        )
    return "\n".join(lines)


@click.command()
@click.option("--universe", required=True)
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Default: today.",
)
def main(universe: str, start, end) -> None:
    end_date = end.date() if end else date.today()
    start_date = start.date()
    df = _load_settled(universe, start_date, end_date)

    click.echo(f"# Backtest summary: {universe} [{start_date} .. {end_date}]\n")
    click.echo(_summary_block(df, "Overall"))
    click.echo("\n### By month\n")
    click.echo(_monthly_block(df))
    click.echo()


if __name__ == "__main__":
    main()
