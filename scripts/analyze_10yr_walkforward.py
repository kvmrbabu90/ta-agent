"""Comprehensive analysis of the 10-year walk-forward backtest.

Answers two key questions:
  1. Across all 10 years, in which regimes does the strategy actually
     UNDERPERFORM? (i.e. low Sharpe, deep drawdown, negative return)
  2. Does regime detector v2 actually help in those specific regimes?

Produces per-year metrics for 4 configs:
  - pure ML (no FVG, no regime gate)
  - + regime v1
  - + regime v2
  - + FVG filter

And SPY benchmark per year for context.
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

WF_PREDS = "data/processed/walkforward_10yr/predictions.sqlite"
WF_PAPER = "data/processed/walkforward_10yr/analysis.sqlite"


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
    """Slice equity curve by calendar year, compute Sharpe + return + DD per year."""
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


def spy_per_year(start: date, end: date) -> pd.DataFrame:
    duck = duckdb.connect(settings.duckdb_path, read_only=True)
    try:
        rows = duck.execute(
            "SELECT bar_date, close FROM ohlcv_daily WHERE symbol = 'SPY' "
            "AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
            [start, end],
        ).fetchall()
    finally:
        duck.close()
    df = pd.DataFrame(rows, columns=["trade_date", "equity"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return per_year_metrics(df)


def main() -> int:
    Path(WF_PAPER).unlink(missing_ok=True)

    log.info("running 4 strategy configs on 10-year WF predictions...")
    configs = [
        (False, False, 1, "Pure ML (no FVG, no regime)"),
        (False, True,  1, "+ regime v1"),
        (False, True,  2, "+ regime v2"),
        (True,  True,  2, "+ FVG + regime v2"),
    ]
    results: dict[str, dict] = {}
    for fvg, regime, rver, label in configs:
        rid = f"a_{int(fvg)}_{int(regime)}_{rver}"
        cfg = StrategyConfig(
            run_id=rid,
            fvg_filter_enabled=fvg,
            regime_gate_enabled=regime,
            regime_version=rver,
            predictions_sqlite_path=WF_PREDS,
            paper_db_path=WF_PAPER,
        )
        backtest(cfg)
        m = metrics(WF_PAPER, rid)
        if m:
            results[label] = m
            log.info(
                f"  {label}: Sharpe {m['sharpe']:.3f}  Final ${m['final']:.0f}  "
                f"MaxDD {m['max_dd_pct']:.1f}%  Trades {m['n_trades']}"
            )

    # Overall summary
    print()
    print("=" * 100)
    print("OVERALL 10-YEAR RESULTS")
    print("=" * 100)
    print(f"{'config':35s} {'Sharpe':>7} {'Final':>10} {'Return':>9} {'MaxDD':>7} {'#trades':>8}")
    print("-" * 100)
    for label, m in results.items():
        ret_pct = (m["final"] / 1000.0 - 1) * 100
        print(
            f"{label:35s} {m['sharpe']:>7.3f} ${m['final']:>8.0f} {ret_pct:>+7.1f}%  "
            f"{m['max_dd_pct']:>6.1f}% {m['n_trades']:>8}"
        )

    # Per-year breakdown for each config
    first_label = next(iter(results))
    spy_yr = spy_per_year(
        results[first_label]["df"]["trade_date"].iloc[0],
        results[first_label]["df"]["trade_date"].iloc[-1],
    )

    print()
    print("=" * 100)
    print("PER-YEAR SHARPE — pure ML vs +v1 vs +v2 vs +FVG+v2 vs SPY benchmark")
    print("=" * 100)
    per_yr_per_cfg = {
        label: per_year_metrics(m["df"]).set_index("year")
        for label, m in results.items()
    }
    years = sorted(set().union(*(df.index for df in per_yr_per_cfg.values())))
    spy_by_year = spy_yr.set_index("year")
    headers = list(per_yr_per_cfg.keys())
    print(f"{'year':>6} " + " ".join(f"{h[:14]:>14}" for h in headers) + f" {'SPY':>10}")
    for y in years:
        cells = []
        for h in headers:
            df = per_yr_per_cfg[h]
            if y in df.index:
                cells.append(f"{df.loc[y, 'sharpe']:>+14.2f}")
            else:
                cells.append(f"{'-':>14}")
        spy_sharpe = (
            f"{spy_by_year.loc[y, 'sharpe']:>+10.2f}"
            if y in spy_by_year.index else f"{'-':>10}"
        )
        print(f"{y:>6} " + " ".join(cells) + f" {spy_sharpe}")

    print()
    print("=" * 100)
    print("PER-YEAR RETURN % — pure ML vs +v1 vs +v2 vs +FVG+v2 vs SPY benchmark")
    print("=" * 100)
    print(f"{'year':>6} " + " ".join(f"{h[:14]:>14}" for h in headers) + f" {'SPY':>10}")
    for y in years:
        cells = []
        for h in headers:
            df = per_yr_per_cfg[h]
            if y in df.index:
                cells.append(f"{df.loc[y, 'return_pct']:>+13.2f}%")
            else:
                cells.append(f"{'-':>14}")
        spy_ret = (
            f"{spy_by_year.loc[y, 'return_pct']:>+9.2f}%"
            if y in spy_by_year.index else f"{'-':>10}"
        )
        print(f"{y:>6} " + " ".join(cells) + f" {spy_ret}")

    print()
    print("=" * 100)
    print("PER-YEAR MAX DRAWDOWN % — pure ML vs +v1 vs +v2 vs +FVG+v2 vs SPY")
    print("=" * 100)
    print(f"{'year':>6} " + " ".join(f"{h[:14]:>14}" for h in headers) + f" {'SPY':>10}")
    for y in years:
        cells = []
        for h in headers:
            df = per_yr_per_cfg[h]
            if y in df.index:
                cells.append(f"{df.loc[y, 'max_dd_pct']:>13.2f}%")
            else:
                cells.append(f"{'-':>14}")
        spy_dd = (
            f"{spy_by_year.loc[y, 'max_dd_pct']:>9.2f}%"
            if y in spy_by_year.index else f"{'-':>10}"
        )
        print(f"{y:>6} " + " ".join(cells) + f" {spy_dd}")

    # Identify FAILURE years: pure-ML Sharpe < 0.5 or DD > 15%
    base = per_yr_per_cfg["Pure ML (no FVG, no regime)"]
    print()
    print("=" * 100)
    print("FAILURE YEARS (pure-ML Sharpe < 0.5 OR MaxDD > 15%)")
    print("=" * 100)
    failure_years = []
    for y in base.index:
        s = base.loc[y, "sharpe"]
        d = base.loc[y, "max_dd_pct"]
        if (s is not None and s < 0.5) or d > 15:
            failure_years.append(y)
            print(f"  {y}: Sharpe={s:.2f}, MaxDD={d:.1f}%, Return={base.loc[y,'return_pct']:+.1f}%")
    if not failure_years:
        print("  None — pure ML works in every year.")
    else:
        print()
        print(f"In FAILURE years, did v2 help? Compare base vs +v2:")
        v2 = per_yr_per_cfg["+ regime v2"]
        for y in failure_years:
            if y in v2.index:
                bs, bd, br = base.loc[y, "sharpe"], base.loc[y, "max_dd_pct"], base.loc[y, "return_pct"]
                vs, vd, vr = v2.loc[y, "sharpe"], v2.loc[y, "max_dd_pct"], v2.loc[y, "return_pct"]
                ds = (vs or 0) - (bs or 0)
                dd_diff = vd - bd
                dr = vr - br
                helped = "+v2 helped" if (ds > 0 or dd_diff < 0) else "+v2 hurt"
                print(
                    f"  {y}: Sharpe {bs:.2f}→{vs:.2f} ({ds:+.2f})  "
                    f"DD {bd:.1f}→{vd:.1f}% ({dd_diff:+.1f})  "
                    f"Return {br:+.1f}→{vr:+.1f}% ({dr:+.1f})  ← {helped}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
