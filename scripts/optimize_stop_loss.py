"""Grid-search the stop-loss parameters by Sharpe ratio.

Usage:

    python -m scripts.optimize_stop_loss
    python -m scripts.optimize_stop_loss --grid wide
    python -m scripts.optimize_stop_loss --csv data/processed/stop_grid.csv

For each (support_lookback_days N, stop_buffer_pct buf) pair, runs the
overlapping-portfolios long-only backtest end-to-end and computes the
annualized Sharpe ratio of the 5 PM equity curve. Prints a heatmap and
the best cell.

Caveats:
  - Requires meaningful coverage in predictions_log; with the current
    sparse log (a handful of dates) the optimizer is uninformative.
  - Uses Sharpe as the single objective per the original spec. Sortino,
    final equity, and trade count are reported alongside for context.
  - All runs write to disjoint run_ids (`opt_NXX_bufYYYY`) so they don't
    clobber the live `default` run. Cleanup is automatic at next backtest.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from packages.common.config import settings
from packages.common.logging import log
from packages.paper_trading import StrategyConfig, backtest

_GRIDS = {
    "default": {
        "N": [5, 10, 15, 20, 30, 50],
        "buf": [0.003, 0.005, 0.006, 0.008, 0.010, 0.015],
    },
    "wide": {
        "N": [3, 5, 7, 10, 15, 20, 30, 50, 75],
        "buf": [0.002, 0.003, 0.005, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030],
    },
    "tight": {
        "N": [10, 15, 20, 25, 30],
        "buf": [0.004, 0.005, 0.006, 0.007, 0.008],
    },
    # Zoom into the edge of `wide` where Sharpe is highest (small N, tight buf).
    "edge": {
        "N": [1, 2, 3, 4, 5, 7],
        "buf": [0.000, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005],
    },
}

TRADING_DAYS_PER_YEAR = 252


def _paper_db_path() -> str:
    return str(Path(settings.predictions_sqlite_path).parent / "paper.sqlite")


def _equity_curve(run_id: str, *, paper_db_path: str | None = None) -> pd.Series:
    """Return the 5 PM close_5pm_ct equity series indexed by trade_date."""
    conn = sqlite3.connect(paper_db_path or _paper_db_path())
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, equity FROM paper_equity "
            "WHERE run_id = ? AND snapshot_kind = 'close_5pm_ct' "
            "ORDER BY trade_date",
            conn, params=[run_id],
        )
    finally:
        conn.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.set_index("trade_date")["equity"]


def _annualized_sharpe(equity: pd.Series) -> float | None:
    """Sharpe of daily returns × sqrt(252). Returns None if too few points."""
    if len(equity) < 3:
        return None
    rets = equity.pct_change().dropna()
    if rets.std(ddof=1) == 0 or pd.isna(rets.std(ddof=1)):
        return None
    return float(rets.mean() / rets.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _annualized_sortino(equity: pd.Series) -> float | None:
    if len(equity) < 3:
        return None
    rets = equity.pct_change().dropna()
    downside = rets[rets < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return None
    return float(rets.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _trade_count(run_id: str, *, paper_db_path: str | None = None) -> int:
    conn = sqlite3.connect(paper_db_path or _paper_db_path())
    try:
        return int(conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE run_id = ?", (run_id,),
        ).fetchone()[0])
    finally:
        conn.close()


def run_grid(
    grid: dict[str, list],
    *,
    predictions_path: str | None = None,
    paper_db_path: str | None = None,
) -> pd.DataFrame:
    """Sweep the (N, buf) grid against a given predictions DB.

    Default behavior reads from `settings.predictions_sqlite_path` and
    writes paper state to the default paper.sqlite. Pass overrides to
    aim at the walk-forward DB so we can re-optimize on honest predictions.
    """
    rows = []
    n_total = len(grid["N"]) * len(grid["buf"])
    i = 0
    for N in grid["N"]:
        for buf in grid["buf"]:
            i += 1
            run_id = f"opt_N{N:03d}_buf{int(round(buf * 10000)):04d}"
            log.info(f"[{i}/{n_total}] running {run_id}")
            cfg = StrategyConfig(
                run_id=run_id,
                support_lookback_days=N,
                stop_buffer_pct=buf,
                predictions_sqlite_path=predictions_path,
                paper_db_path=paper_db_path,
                notes=f"stop-loss grid search N={N} buf={buf}",
            )
            summary = backtest(cfg)
            equity = _equity_curve(run_id, paper_db_path=paper_db_path)
            sharpe = _annualized_sharpe(equity)
            sortino = _annualized_sortino(equity)
            n_trades = _trade_count(run_id, paper_db_path=paper_db_path)
            rows.append({
                "N": N,
                "buf_pct": buf,
                "sharpe": sharpe,
                "sortino": sortino,
                "final_equity": summary.get("final_equity"),
                "realized_pnl": summary.get("final_realized_pnl"),
                "n_trade_days": summary.get("n_trade_days"),
                "n_trades": n_trades,
            })
    return pd.DataFrame(rows)


def _print_heatmap(df: pd.DataFrame, column: str) -> None:
    """Pivot to N x buf grid and print as a fixed-width table."""
    if df.empty:
        print("(no results)")
        return
    pivot = df.pivot(index="N", columns="buf_pct", values=column)
    print(f"\n=== {column} heatmap (rows=lookback_N, cols=stop_buffer_pct) ===")
    print(pivot.round(3).to_string(na_rep="  n/a"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--grid", choices=list(_GRIDS), default="default",
        help="Which parameter grid to sweep",
    )
    p.add_argument(
        "--csv", type=Path, default=None,
        help="If provided, write results to this CSV path",
    )
    p.add_argument(
        "--predictions-path", default=None,
        help="Path to a predictions SQLite (default: settings.predictions_sqlite_path). "
        "Use this to point at the walk-forward DB to optimize on honest predictions.",
    )
    p.add_argument(
        "--paper-db-path", default=None,
        help="Path to write paper backtest state to (default: data/processed/paper.sqlite). "
        "Use a separate file when optimizing against the WF DB so you don't pollute live state.",
    )
    args = p.parse_args()
    grid = _GRIDS[args.grid]
    log.info(
        f"stop-loss grid: N={grid['N']} buf={grid['buf']}  "
        f"predictions={args.predictions_path or 'default'}  "
        f"paper_db={args.paper_db_path or 'default'}"
    )
    df = run_grid(
        grid,
        predictions_path=args.predictions_path,
        paper_db_path=args.paper_db_path,
    )
    if df.empty:
        log.error("no results")
        return 1

    _print_heatmap(df, "sharpe")
    _print_heatmap(df, "sortino")
    _print_heatmap(df, "final_equity")
    _print_heatmap(df, "n_trades")

    # Best by Sharpe (primary objective).
    best_sharpe = df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
    if not best_sharpe.empty:
        b = best_sharpe.iloc[0]
        print(
            f"\n>>> best by Sharpe: N={int(b['N'])} buf_pct={b['buf_pct']:.4f}  "
            f"sharpe={b['sharpe']:.3f}  sortino={b['sortino']:.3f}  "
            f"final=${b['final_equity']:.2f}  trades={int(b['n_trades'])}"
        )
    else:
        print("\n>>> no cell produced a finite Sharpe (likely too few trade days)")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        log.info(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
