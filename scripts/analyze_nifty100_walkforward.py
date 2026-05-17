"""Phase A: analyze the NIFTY100 walk-forward backtest.

Mirrors `analyze_10yr_walkforward.py` but for the India universe:
  - Runs the paper engine with 4 strategy configs on the WF predictions
  - Per-year Sharpe / return / drawdown breakdown
  - NIFTY50 benchmark (USD/INR-naive — both denominated in INR)
  - Failure-year identification

CRITICAL CAVEATS (Phase A scope):
  - Cost model is IBKR Lite (US fees). Indian retail equity costs are
    different: STT 0.1% on sells, brokerage 0% for delivery on Zerodha,
    plus exchange + GST. Effective drag ≈ 10-15bps per round-trip vs
    ~5bps in the US model. Treat absolute returns as ~5-10% optimistic
    annualized; relative comparisons (strategy vs benchmark) are robust.

  - Engine timing labels (8:35 CT / 5pm CT) are descriptive only; the
    engine operates on daily open/close bars. India market hours
    (9:15 AM - 3:30 PM IST) don't change the math.

  - HAL collision fixed via universe-aware OHLCV fetch (see engine.py
    `_load_ohlcv_for_symbols(universe=...)`).
"""

from __future__ import annotations

import math
import sqlite3
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from packages.common.config import settings
from packages.common.logging import log
from packages.paper_trading import StrategyConfig, backtest

import argparse

DEFAULT_WF_PREDS = "data/processed/india_phase_a/walkforward/predictions.sqlite"
DEFAULT_wf_paper = "data/processed/india_phase_a/analysis.sqlite"
UNIVERSE = "NIFTY100"
BENCHMARK_SYMBOL = "NIFTY 50"  # macro_daily series name? check
BENCHMARK_EXCH = "NSE"


def metrics(paper_db: str, run_id: str) -> dict | None:
    c = sqlite3.connect(paper_db)
    rows = c.execute(
        "SELECT trade_date, equity FROM paper_equity "
        "WHERE run_id=? AND snapshot_kind='close_5pm_ct' ORDER BY trade_date",
        (run_id,),
    ).fetchall()
    n_trades = c.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    c.close()
    if len(rows) < 3:
        return None
    df = pd.DataFrame(rows, columns=["trade_date", "equity"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    eqs = df["equity"].to_numpy()
    rets = [(eqs[i] / eqs[i - 1] - 1) for i in range(1, len(eqs)) if eqs[i - 1] > 0]
    m = sum(rets) / len(rets)
    s = (sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
    sharpe = m / s * math.sqrt(252) if s > 0 else None
    peak = eqs[0]
    dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        dd = max(dd, (peak - e) / peak)
    return {
        "df": df,
        "sharpe": sharpe,
        "final": eqs[-1],
        "max_dd_pct": dd * 100,
        "n_trades": n_trades,
    }


def per_year_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    rows = []
    for year, g in df.groupby("year"):
        g = g.reset_index(drop=True)
        if len(g) < 5:
            continue
        eqs = g["equity"].to_numpy()
        rets = [(eqs[i] / eqs[i - 1] - 1) for i in range(1, len(eqs)) if eqs[i - 1] > 0]
        if not rets:
            continue
        m = sum(rets) / len(rets)
        s = (sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5 if len(rets) > 1 else 0
        sharpe = m / s * math.sqrt(252) if s > 0 else None
        total_ret = (eqs[-1] / eqs[0]) - 1
        peak = eqs[0]
        dd = 0.0
        for e in eqs:
            peak = max(peak, e)
            dd = max(dd, (peak - e) / peak)
        rows.append({
            "year": int(year),
            "sharpe": sharpe,
            "return_pct": total_ret * 100,
            "max_dd_pct": dd * 100,
            "n_days": len(g),
        })
    return pd.DataFrame(rows)


def benchmark_per_year(start: date, end: date) -> pd.DataFrame:
    """Build a NIFTY100 equal-weight 'index' from the OHLCV closes we have.

    We don't have an actual NIFTY100 index series in DuckDB. Building an
    equal-weight proxy from the 100 NSE constituents is closer to a fair
    comparison anyway since the strategy itself is cross-sectional within
    the same 100 names. Returns the per-year stats for this synthetic.
    """
    duck = duckdb.connect(settings.duckdb_path, read_only=True)
    try:
        rows = duck.execute(
            """
            SELECT o.bar_date, AVG(o.close / first_close.fc) AS index_value
            FROM ohlcv_daily o
            JOIN (SELECT DISTINCT symbol, exchange FROM index_membership
                  WHERE universe = ?) m USING (symbol, exchange)
            JOIN (
                SELECT symbol, exchange, MIN_BY(close, bar_date) AS fc
                FROM ohlcv_daily
                WHERE bar_date >= ? AND bar_date <= ?
                GROUP BY symbol, exchange
            ) first_close USING (symbol, exchange)
            WHERE o.bar_date BETWEEN ? AND ?
            GROUP BY o.bar_date
            ORDER BY o.bar_date
            """,
            [UNIVERSE, start, end, start, end],
        ).fetchall()
    finally:
        duck.close()
    df = pd.DataFrame(rows, columns=["trade_date", "equity"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return per_year_metrics(df)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", default=DEFAULT_WF_PREDS,
                   help="Walk-forward predictions sqlite path")
    p.add_argument("--paper-db", default=DEFAULT_wf_paper,
                   help="Paper backtest sqlite output path")
    p.add_argument("--commission-model", default="ibkr_lite",
                   choices=["none", "ibkr_lite", "india_zerodha"],
                   help="Cost model. Use india_zerodha for realistic India retail equity costs.")
    args = p.parse_args()
    wf_preds = args.preds
    wf_paper = args.paper_db
    commission_model = args.commission_model

    Path(wf_paper).unlink(missing_ok=True)
    Path(wf_paper).parent.mkdir(parents=True, exist_ok=True)

    log.info(f"analyzing WF preds={wf_preds} costs={commission_model}")
    configs = [
        (False, False, 1, "Pure ML (no FVG, no regime)"),
        (False, True,  1, "+ regime v1"),
        (False, True,  2, "+ regime v2"),
        (True,  True,  2, "+ FVG + regime v2"),
    ]
    results: dict[str, dict] = {}
    for fvg, regime, rver, label in configs:
        rid = f"nifty_{int(fvg)}_{int(regime)}_{rver}"
        cfg = StrategyConfig(
            run_id=rid,
            universe=UNIVERSE,
            fvg_filter_enabled=fvg,
            regime_gate_enabled=regime,
            regime_version=rver,
            predictions_sqlite_path=wf_preds,
            paper_db_path=wf_paper,
            commission_model=commission_model,
        )
        backtest(cfg)
        m = metrics(wf_paper, rid)
        if m:
            results[label] = m
            log.info(
                f"  {label}: Sharpe {m['sharpe']:.3f}  Final ₹{m['final']:.0f}  "
                f"MaxDD {m['max_dd_pct']:.1f}%  Trades {m['n_trades']}"
            )

    # Overall summary
    print()
    print("=" * 100)
    print(f"OVERALL {UNIVERSE} WALK-FORWARD RESULTS")
    print("=" * 100)
    print(f"{'config':35s} {'Sharpe':>7} {'Final':>10} {'Return':>9} {'MaxDD':>7} {'#trades':>8}")
    print("-" * 100)
    for label, m in results.items():
        ret_pct = (m["final"] / 1000.0 - 1) * 100
        print(
            f"{label:35s} {m['sharpe']:>7.3f} ₹{m['final']:>8.0f} {ret_pct:>+7.1f}%  "
            f"{m['max_dd_pct']:>6.1f}% {m['n_trades']:>8}"
        )

    if not results:
        print("No results — likely no predictions in WF preds db.")
        return 1

    # Benchmark
    first_label = next(iter(results))
    bench = benchmark_per_year(
        results[first_label]["df"]["trade_date"].iloc[0],
        results[first_label]["df"]["trade_date"].iloc[-1],
    )

    print()
    print("=" * 100)
    print(f"PER-YEAR SHARPE — strategy configs vs equal-weight NIFTY100 benchmark")
    print("=" * 100)
    per_yr_per_cfg = {
        label: per_year_metrics(m["df"]).set_index("year")
        for label, m in results.items()
    }
    years = sorted(set().union(*(df.index for df in per_yr_per_cfg.values())))
    bench_by_year = bench.set_index("year")
    headers = list(per_yr_per_cfg.keys())
    print(f"{'year':>6} " + " ".join(f"{h[:14]:>14}" for h in headers) + f" {'BENCH':>10}")
    for y in years:
        cells = []
        for h in headers:
            df = per_yr_per_cfg[h]
            if y in df.index and df.loc[y, "sharpe"] is not None:
                cells.append(f"{df.loc[y, 'sharpe']:>+14.2f}")
            else:
                cells.append(f"{'-':>14}")
        b = (
            f"{bench_by_year.loc[y, 'sharpe']:>+10.2f}"
            if y in bench_by_year.index and bench_by_year.loc[y, 'sharpe'] is not None
            else f"{'-':>10}"
        )
        print(f"{y:>6} " + " ".join(cells) + f" {b}")

    print()
    print("=" * 100)
    print(f"PER-YEAR RETURN % — strategy vs benchmark")
    print("=" * 100)
    print(f"{'year':>6} " + " ".join(f"{h[:14]:>14}" for h in headers) + f" {'BENCH':>10}")
    for y in years:
        cells = []
        for h in headers:
            df = per_yr_per_cfg[h]
            if y in df.index:
                cells.append(f"{df.loc[y, 'return_pct']:>+13.2f}%")
            else:
                cells.append(f"{'-':>14}")
        b = (
            f"{bench_by_year.loc[y, 'return_pct']:>+9.2f}%"
            if y in bench_by_year.index else f"{'-':>10}"
        )
        print(f"{y:>6} " + " ".join(cells) + f" {b}")

    print()
    print("=" * 100)
    print(f"PER-YEAR MAX DRAWDOWN % — strategy vs benchmark")
    print("=" * 100)
    print(f"{'year':>6} " + " ".join(f"{h[:14]:>14}" for h in headers) + f" {'BENCH':>10}")
    for y in years:
        cells = []
        for h in headers:
            df = per_yr_per_cfg[h]
            if y in df.index:
                cells.append(f"{df.loc[y, 'max_dd_pct']:>13.2f}%")
            else:
                cells.append(f"{'-':>14}")
        b = (
            f"{bench_by_year.loc[y, 'max_dd_pct']:>9.2f}%"
            if y in bench_by_year.index else f"{'-':>10}"
        )
        print(f"{y:>6} " + " ".join(cells) + f" {b}")

    # Phase A decision gate
    print()
    print("=" * 100)
    print("PHASE A DECISION GATE")
    print("=" * 100)
    base = per_yr_per_cfg["Pure ML (no FVG, no regime)"]
    base_overall = results["Pure ML (no FVG, no regime)"]
    bench_overall_final = (
        bench["return_pct"].apply(lambda r: 1 + r / 100).prod() * 1000.0
    )
    strat_final = base_overall["final"]
    n_years = len(base.index)
    years_beating = sum(
        1 for y in base.index
        if y in bench_by_year.index and base.loc[y, "return_pct"] > bench_by_year.loc[y, "return_pct"]
    )

    print(f"  Strategy overall Sharpe:         {base_overall['sharpe']:.3f}")
    print(f"  Strategy final equity:           ₹{strat_final:,.0f}")
    print(f"  Benchmark final equity:          ₹{bench_overall_final:,.0f}")
    print(f"  Years beating benchmark:         {years_beating}/{n_years}")
    print()
    sharpe_ok = base_overall["sharpe"] > 1.0
    beat_ok = years_beating / max(n_years, 1) >= 0.6
    overall_ok = sharpe_ok and beat_ok
    print(f"  Gate: Sharpe > 1.0      ?  {'PASS' if sharpe_ok else 'FAIL'}")
    print(f"  Gate: beats >=60% years ?  {'PASS' if beat_ok else 'FAIL'}")
    print()
    print(f"  RECOMMENDATION: {'PROCEED TO PHASE B' if overall_ok else 'SHELF Phase B'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
