"""Monthly performance breakdown for the strict walk-forward runs.

For each universe with in-progress strict-WF predictions:
  1. Load the latest paper-equity replay (computed by the API at the last
     /performance/strict-wf/{universe} request, so this script reads from
     analysis_live.sqlite without re-running the engine).
  2. Aggregate equity to monthly returns + cumulative + benchmark.
  3. Print a table per universe matching the format the user liked
     previously (header + per-month row + cumulative footer).

This script is read-only — it does not touch DuckDB. Safe to run while
the WFs are training. If analysis_live.sqlite is missing for a universe
(API hasn't been hit since the last retrain), the script tells you and
moves on.
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
from pathlib import Path

import duckdb
import pandas as pd

from packages.common.config import settings

UNIVERSES = [
    {
        "name": "SP500",
        "paper": "data/processed/walkforward_10yr_strict/analysis_live.sqlite",
        "run_id": "strict_wf_live_sp500",
        "benchmark_sym": "SPY",
        "benchmark_label": "SPY",
        "currency": "USD",
    },
    {
        "name": "NIFTY100",
        "paper": "data/processed/wf_nifty100_strict/analysis_live.sqlite",
        "run_id": "strict_wf_live_nifty100",
        "benchmark_sym": "NIFTYBEES",
        "benchmark_label": "NIFTYBEES",
        "currency": "INR",
    },
]


def _load_monthly_equity(paper_path: str, run_id: str) -> pd.DataFrame | None:
    if not Path(paper_path).exists():
        return None
    con = sqlite3.connect(paper_path)
    # IMPORTANT: paper_equity has two snapshots per day (open_8am_ct,
    # close_5pm_ct). Use close_5pm_ct only so first/last of each month
    # are on the same daily clock — mixing the two gives spurious month
    # returns (open of first day vs close of last day) and inflates cum.
    df = pd.read_sql_query(
        "SELECT trade_date, equity FROM paper_equity "
        "WHERE run_id=? AND snapshot_kind='close_5pm_ct' ORDER BY trade_date",
        con,
        params=[run_id],
    )
    con.close()
    if df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def _benchmark_monthly(
    duck: duckdb.DuckDBPyConnection, symbol: str, start, end
) -> pd.DataFrame:
    rows = duck.execute(
        "SELECT bar_date, close FROM ohlcv_daily WHERE symbol = ? "
        "AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
        [symbol, start, end],
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["year", "month", "bench_ret"])
    df = pd.DataFrame(rows, columns=["bar_date", "close"])
    df["bar_date"] = pd.to_datetime(df["bar_date"])
    df["ym"] = df["bar_date"].dt.to_period("M")
    grouped = df.groupby("ym").agg(first=("close", "first"), last=("close", "last"))
    grouped["bench_ret"] = (grouped["last"] / grouped["first"] - 1) * 100
    grouped = grouped.reset_index()
    grouped["year"] = grouped["ym"].dt.year
    grouped["month"] = grouped["ym"].dt.month
    return grouped[["year", "month", "bench_ret"]]


def _per_month_strategy(df_eq: pd.DataFrame) -> pd.DataFrame:
    """Monthly returns using LAST-of-prior-month → LAST-of-this-month.

    Subtle: using (month_last / month_first) per month skips any equity
    change between month boundaries — the strict-WF replay re-balances
    each retrain (independent monthly window), so the cost of rotating
    positions falls between (last close of month N) and (first close of
    month N+1). Compounding (last/first) per month therefore over-states
    cumulative return. Anchoring each month to the prior month's CLOSE
    keeps cum equal to end-to-end equity ratio.
    """
    df_eq = df_eq.sort_values("trade_date").reset_index(drop=True)
    df_eq["ym"] = df_eq["trade_date"].dt.to_period("M")
    # Per-month last close, then per-month return = this_last / prior_last.
    last_per_month = df_eq.groupby("ym")["equity"].last()
    n_days = df_eq.groupby("ym")["equity"].size()
    starting = df_eq["equity"].iloc[0]
    # Anchor first month to the engine's starting close; later months
    # anchor to the previous month's close. This makes the per-month
    # returns compound exactly to the end-to-end equity ratio.
    prev = last_per_month.shift(1)
    prev.iloc[0] = starting
    ret = (last_per_month / prev - 1) * 100
    g = pd.DataFrame({"first": prev.values, "last": last_per_month.values,
                       "n_days": n_days.values, "ret_pct": ret.values},
                      index=last_per_month.index)
    g.index.name = "ym"
    g = g.reset_index()
    # Sharpe per month from intra-month daily returns
    sharpes = []
    max_dds = []
    for ym, sub in df_eq.groupby("ym"):
        eqs = sub["equity"].to_numpy()
        rets = [eqs[i] / eqs[i - 1] - 1 for i in range(1, len(eqs)) if eqs[i - 1] > 0]
        if len(rets) > 1:
            m = sum(rets) / len(rets)
            s = (sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
            sharpes.append(m / s * math.sqrt(252) if s > 0 else None)
        else:
            sharpes.append(None)
        peak = eqs[0]
        dd = 0.0
        for e in eqs:
            peak = max(peak, e)
            dd = max(dd, (peak - e) / peak)
        max_dds.append(dd * 100)
    g["sharpe"] = sharpes
    g["max_dd_pct"] = max_dds
    return g.reset_index()


def _print_universe_report(uni: dict, duck: duckdb.DuckDBPyConnection) -> None:
    print()
    print(f"=== {uni['name']} strict walk-forward — monthly ===")
    print(f"benchmark: {uni['benchmark_label']}  currency: {uni['currency']}")

    df_eq = _load_monthly_equity(uni["paper"], uni["run_id"])
    if df_eq is None:
        print("  (no replay available yet — hit /performance/strict-wf/"
              f"{uni['name']} first or wait for next retrain)")
        return

    strat = _per_month_strategy(df_eq)
    start_d = df_eq["trade_date"].min().date()
    end_d = df_eq["trade_date"].max().date()
    bench = _benchmark_monthly(duck, uni["benchmark_sym"], start_d, end_d)
    strat["year"] = strat["ym"].dt.year
    strat["month"] = strat["ym"].dt.month
    merged = strat.merge(bench, on=["year", "month"], how="left")

    print()
    print(f"{'Month':<10}{'Strategy':>11}{'Bench':>11}{'Excess':>11}"
          f"{'Sharpe':>9}{'MaxDD':>9}{'Days':>6}")
    print("-" * 67)
    strat_cum = 1.0
    bench_cum = 1.0
    for _, r in merged.iterrows():
        ym = f"{int(r['year'])}-{int(r['month']):02d}"
        s = r["ret_pct"]
        b = r.get("bench_ret")
        b_str = f"{b:+.2f}%" if pd.notna(b) else "—"
        excess_str = f"{s - b:+.2f}%" if pd.notna(b) else "—"
        sharpe = r["sharpe"]
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None and not pd.isna(sharpe) else "—"
        print(f"{ym:<10}{s:>+10.2f}% {b_str:>10} {excess_str:>10}"
              f"{sharpe_str:>9}{r['max_dd_pct']:>+8.1f}%{int(r['n_days']):>6}")
        strat_cum *= (1 + s / 100)
        if pd.notna(b):
            bench_cum *= (1 + b / 100)

    print("-" * 67)
    sn = (strat_cum - 1) * 100
    bn = (bench_cum - 1) * 100
    print(f"{'Cum':<10}{sn:>+10.2f}% {bn:>+10.2f}% "
          f"{(sn - bn):>+10.2f}%")
    print()


def main() -> int:
    duck = duckdb.connect(settings.duckdb_path, read_only=True)
    try:
        for uni in UNIVERSES:
            _print_universe_report(uni, duck)
    finally:
        duck.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
