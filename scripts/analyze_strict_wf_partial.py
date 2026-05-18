"""Year-by-year quick-look at an in-progress strict walk-forward.

Run after every N retrains complete to see how the strategy is doing
year-over-year on the survivorship-bias-corrected, no-look-ahead WF.

    python -m scripts.analyze_strict_wf_partial --universe SP500
    python -m scripts.analyze_strict_wf_partial --universe NIFTY100

Reads predictions.sqlite from the strict-WF output dir, replays the
paper engine over only the dates that exist so far, and prints
per-year strategy return vs benchmark (SPY/NIFTYBEES) buy-and-hold,
with both pre- and post-tax (STCG/LTCG) figures.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from packages.common.logging import log
from packages.paper_trading import StrategyConfig, backtest


_UNIVERSE_CFG = {
    "SP500": {
        "wf_dir": "data/processed/walkforward_10yr_strict",
        "benchmark": "SPY",
        "stcg": 0.25,
        "ltcg": 0.15,
        "currency": "USD",
        "commission": "ibkr_lite",
    },
    "NIFTY100": {
        "wf_dir": "data/processed/wf_nifty100_strict",
        "benchmark": "NIFTYBEES",
        "stcg": 0.20,
        "ltcg": 0.125,
        "currency": "INR",
        "commission": "india_zerodha",
    },
}


def replay_engine(universe: str, wf_dir: str, commission: str) -> str | None:
    """Run the paper engine on whatever predictions are present. Returns
    the paper-db path or None if predictions are empty."""
    preds = f"{wf_dir}/predictions.sqlite"
    if not os.path.exists(preds):
        return None
    c = sqlite3.connect(preds)
    n = c.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0]
    c.close()
    if n == 0:
        return None
    paper_db = f"{wf_dir}/analysis.sqlite"
    Path(paper_db).unlink(missing_ok=True)
    cfg = StrategyConfig(
        run_id=f"strict_wf_{universe.lower()}",
        universe=universe,
        predictions_sqlite_path=preds,
        paper_db_path=paper_db,
        commission_model=commission,
    )
    backtest(cfg)
    return paper_db


def per_year_strategy_returns(paper_db: str, run_id: str) -> pd.DataFrame:
    c = sqlite3.connect(paper_db)
    df = pd.read_sql_query(
        "SELECT trade_date, equity FROM paper_equity "
        "WHERE run_id=? AND snapshot_kind='close_5pm_ct' ORDER BY trade_date",
        c, params=[run_id],
    )
    c.close()
    if df.empty:
        return pd.DataFrame(columns=["year", "return", "equity_end"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    rows = []
    for year, g in df.groupby("year"):
        eqs = g["equity"].to_numpy()
        rets = [eqs[i] / eqs[i - 1] - 1 for i in range(1, len(eqs)) if eqs[i - 1] > 0]
        m = sum(rets) / max(len(rets), 1)
        s = (sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)) ** 0.5 if len(rets) > 1 else 0
        sharpe = m / s * math.sqrt(252) if s > 0 else float("nan")
        peak = eqs[0]
        dd = 0.0
        for e in eqs:
            peak = max(peak, e)
            dd = max(dd, (peak - e) / peak)
        rows.append({
            "year": int(year),
            "return": float(eqs[-1] / eqs[0] - 1),
            "equity_end": float(eqs[-1]),
            "sharpe": sharpe,
            "max_dd_pct": dd * 100,
            "n_days": int(len(g)),
        })
    return pd.DataFrame(rows)


def benchmark_per_year(symbol: str) -> dict[int, float]:
    duck = duckdb.connect("data/processed/market.duckdb", read_only=True)
    rows = duck.execute(
        "SELECT bar_date, close FROM ohlcv_daily WHERE symbol = ? ORDER BY bar_date",
        [symbol],
    ).fetchall()
    duck.close()
    df = pd.DataFrame(rows, columns=["bar_date", "close"])
    df["year"] = pd.to_datetime(df["bar_date"]).dt.year
    out: dict[int, float] = {}
    for year, g in df.groupby("year"):
        closes = g["close"].to_numpy()
        if len(closes) >= 2:
            out[int(year)] = float(closes[-1] / closes[0] - 1)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["SP500", "NIFTY100"], required=True)
    args = p.parse_args()
    cfg = _UNIVERSE_CFG[args.universe]

    log.info(f"Replaying engine on partial WF for {args.universe}...")
    paper_db = replay_engine(args.universe, cfg["wf_dir"], cfg["commission"])
    if paper_db is None:
        print("No predictions yet.")
        return 0

    run_id = f"strict_wf_{args.universe.lower()}"
    yearly = per_year_strategy_returns(paper_db, run_id)
    bench = benchmark_per_year(cfg["benchmark"])

    print()
    print("=" * 80)
    print(f"{args.universe} strict-WF — per-year results")
    print(f"(pre-tax strategy vs pre-tax {cfg['benchmark']} B&H)")
    print("=" * 80)
    print(f"{'year':>6} {'strat ret':>11} {'sharpe':>8} {'maxDD':>8} {'bench ret':>11} {'strat - bench':>14}")
    print("-" * 80)
    strat_cum = 1.0
    bench_cum = 1.0
    for _, r in yearly.iterrows():
        y = r["year"]
        s = r["return"]
        b = bench.get(y, float("nan"))
        strat_cum *= (1 + s)
        bench_cum *= (1 + b) if not math.isnan(b) else 1
        excess = s - b if not math.isnan(b) else float("nan")
        print(
            f"{y:>6} {s * 100:>+10.2f}% {r['sharpe']:>+7.2f} {r['max_dd_pct']:>6.1f}% "
            f"{b * 100:>+10.2f}% {excess * 100:>+13.2f}%"
        )
    print("-" * 80)
    n_years = max(len(yearly), 1)
    print(f"  cumulative strategy: {(strat_cum - 1) * 100:+.1f}%  "
          f"({strat_cum:.2f}x)   ann {(strat_cum ** (1/n_years) - 1) * 100:+.1f}%")
    print(f"  cumulative bench:    {(bench_cum - 1) * 100:+.1f}%  "
          f"({bench_cum:.2f}x)   ann {(bench_cum ** (1/n_years) - 1) * 100:+.1f}%")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
