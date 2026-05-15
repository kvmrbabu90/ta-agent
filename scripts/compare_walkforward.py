"""Compare honest walk-forward predictions against the look-ahead-biased
predictions, and report Sharpe / final-equity / drawdown / regime impact.

Reads:
  data/processed/predictions.sqlite          — biased (today's model used everywhere)
  data/processed/walkforward/predictions.sqlite — honest (model retrained monthly)

Runs the paper engine over a shared date range with both prediction sources,
both regime-gate ON and OFF, and writes a markdown report to:
  data/processed/walkforward/comparison_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from packages.common.config import settings
from packages.common.logging import log
from packages.paper_trading import StrategyConfig, backtest

_WF_DIR = Path("data/processed/walkforward")


@dataclass
class RunResult:
    label: str
    run_id: str
    sharpe: float | None
    sortino: float | None
    final_equity: float | None
    max_dd_pct: float | None
    hit_rate_5d: float | None
    n_trade_days: int
    n_trades: int


def _equity_metrics(paper_db: str, run_id: str) -> RunResult:
    conn = sqlite3.connect(paper_db)
    try:
        rows = conn.execute(
            "SELECT trade_date, equity FROM paper_equity "
            "WHERE run_id = ? AND snapshot_kind = 'close_5pm_ct' "
            "ORDER BY trade_date",
            (run_id,),
        ).fetchall()
        n_trades = int(
            conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        )
    finally:
        conn.close()
    if len(rows) < 3:
        return RunResult(label="", run_id=run_id, sharpe=None, sortino=None,
                         final_equity=None, max_dd_pct=None, hit_rate_5d=None,
                         n_trade_days=len(rows), n_trades=n_trades)
    eqs = [r[1] for r in rows]
    rets = [(eqs[i] / eqs[i - 1] - 1) for i in range(1, len(eqs)) if eqs[i - 1] > 0]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    downside = [r for r in rets if r < 0]
    if len(downside) >= 2:
        d_mean = sum(downside) / len(downside)
        d_var = sum((r - d_mean) ** 2 for r in downside) / (len(downside) - 1)
        d_std = math.sqrt(d_var) if d_var > 0 else 0.0
    else:
        d_std = 0.0
    sharpe = mean / std * math.sqrt(252) if std > 0 else None
    sortino = mean / d_std * math.sqrt(252) if d_std > 0 else None
    peak = eqs[0]
    max_dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)
    return RunResult(
        label="", run_id=run_id, sharpe=sharpe, sortino=sortino,
        final_equity=eqs[-1], max_dd_pct=max_dd * 100,
        hit_rate_5d=None, n_trade_days=len(eqs), n_trades=n_trades,
    )


def _hit_rate_from_predictions(preds_path: str, universe: str) -> float | None:
    """5-day directional hit rate across the top-N picks per date."""
    if not Path(preds_path).exists():
        return None
    conn = sqlite3.connect(preds_path)
    try:
        df = pd.read_sql_query(
            "SELECT predicted_return, realized_return FROM predictions_log "
            "WHERE universe = ? AND realized_return IS NOT NULL",
            conn, params=[universe],
        )
    finally:
        conn.close()
    if df.empty:
        return None
    same = (df["predicted_return"] * df["realized_return"]) > 0
    return float(same.mean())


def _shared_date_range(biased_path: str, wf_path: str, universe: str) -> tuple[date, date] | None:
    """Pick the overlap of dates where BOTH have predictions."""
    def _dates(p):
        if not Path(p).exists():
            return set()
        conn = sqlite3.connect(p)
        try:
            rows = conn.execute(
                "SELECT DISTINCT as_of FROM predictions_log WHERE universe = ?",
                (universe,),
            ).fetchall()
        finally:
            conn.close()
        return {date.fromisoformat(r[0]) for r in rows}
    biased = _dates(biased_path)
    wf = _dates(wf_path)
    common = biased & wf
    if not common:
        return None
    return min(common), max(common)


def _run_one(
    *,
    label: str,
    run_id: str,
    preds_path: str,
    paper_db: str,
    start: date,
    end: date,
    regime_on: bool,
) -> RunResult:
    cfg = StrategyConfig(
        run_id=run_id,
        start_date=start,
        end_date=end,
        predictions_sqlite_path=preds_path,
        paper_db_path=paper_db,
        regime_gate_enabled=regime_on,
        notes=label,
    )
    backtest(cfg)
    res = _equity_metrics(paper_db, run_id)
    res.label = label
    return res


def _fmt_pct(x: float | None) -> str:
    return f"{x:+.2f}%" if x is not None else "—"


def _fmt_money(x: float | None) -> str:
    return f"${x:,.2f}" if x is not None else "—"


def _fmt_num(x: float | None, d: int = 2) -> str:
    return f"{x:.{d}f}" if x is not None else "—"


def _md_table(results: list[RunResult]) -> str:
    lines = [
        "| Run | Sharpe | Sortino | Final equity | Max DD | Trade days | Trades |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ret_pct = (r.final_equity / 1000.0 - 1.0) * 100 if r.final_equity else None
        final_str = (
            f"{_fmt_money(r.final_equity)} ({_fmt_pct(ret_pct)})"
            if r.final_equity else "—"
        )
        lines.append(
            f"| {r.label} | {_fmt_num(r.sharpe)} | {_fmt_num(r.sortino)} | "
            f"{final_str} | {_fmt_num(r.max_dd_pct, 1)}% | {r.n_trade_days} | {r.n_trades} |"
        )
    return "\n".join(lines)


def _build_report(
    results: list[RunResult],
    *,
    biased_path: str,
    wf_path: str,
    universe: str,
    start: date,
    end: date,
    wf_summary: dict,
) -> str:
    biased_hit = _hit_rate_from_predictions(biased_path, universe)
    wf_hit = _hit_rate_from_predictions(wf_path, universe)
    return f"""# Walk-forward backtest comparison

**Universe:** {universe}
**Window:** {start} → {end} ({(end-start).days} calendar days)
**Walk-forward retrains:** {len(wf_summary.get('retrains', []))} monthly
**Walk-forward predictions:** {wf_summary.get('total_predictions', 0):,}
**Walk-forward elapsed:** {wf_summary.get('elapsed_s', 0)/60:.1f} min

## TL;DR

The walk-forward backtest retrains the model at the start of each month
using only data available before that date, then predicts forward. The
look-ahead-biased baseline reuses today's model on every historical date,
which inflates results because the model "saw" what eventually worked.

## Per-prediction directional hit rate

(Pure model quality signal — does the predicted return have the right sign?)

| Source | Hit rate |
|---|---|
| Look-ahead biased | {_fmt_pct((biased_hit or 0) * 100)} |
| Walk-forward      | {_fmt_pct((wf_hit or 0) * 100)} |

## Strategy backtest (overlapping portfolios + stop-loss + IBKR Lite)

{_md_table(results)}

## How to read this

- **Honest vs Biased Sharpe ratio** is the headline number. If the
  walk-forward Sharpe is meaningfully positive (say > 1.0), the strategy
  has real edge. If it collapses to ~0 or negative, the entire +145%
  number from the look-ahead backtest was an artifact of leakage.
- **Regime ON vs OFF**: gate impact in the current window. Mostly
  irrelevant during 2024-2026 (we showed only ~4% of days were in
  trend regime). Real test is years like 2022 (trend bear) or
  2017 (trend bull). Keep it ON as cheap insurance.
- **Final equity**: % return on $1,000 starting cash. Honest value is
  the upper bound on what live paper would deliver in the same period.

## What to do next

1. If walk-forward Sharpe ≥ 1.0 and final equity is positive: the
   strategy passes the integrity test. Paper-trade for 4-8 weeks at
   $1000 then scale up if live results track honest backtest within
   ±50%.
2. If walk-forward Sharpe is between 0 and 1.0: marginal edge. Worth
   continuing to paper-trade but don't deploy real capital.
3. If walk-forward Sharpe ≤ 0 or final equity < $1000: model has no
   real out-of-sample edge. Most of the +145% was leakage. Need to
   refactor (different features, different target horizon, different
   universe) before this can be trusted.

## Audit detail

- Look-ahead predictions DB: `{biased_path}`
- Walk-forward predictions DB: `{wf_path}`
- Per-retrain breakdown: `data/processed/walkforward/report.json`
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="SP500")
    p.add_argument("--biased-path", default=settings.predictions_sqlite_path)
    p.add_argument("--wf-path", default=str(_WF_DIR / "predictions.sqlite"))
    p.add_argument("--paper-db", default=str(_WF_DIR / "comparison_paper.sqlite"))
    p.add_argument("--out", default=str(_WF_DIR / "comparison_report.md"))
    args = p.parse_args()

    rng = _shared_date_range(args.biased_path, args.wf_path, args.universe)
    if rng is None:
        log.error("no overlapping dates between biased and walk-forward predictions")
        return 1
    start, end = rng
    log.info(f"comparing on shared range {start} .. {end}")

    # Settle walk-forward predictions so the hit-rate column is meaningful.
    # Without this, all WF rows are unsettled (realized_return IS NULL) and
    # the hit-rate computation returns 0%.
    try:
        from packages.inference.tracking import settle_predictions
        n_settled = settle_predictions(
            as_of=date.today(), horizon_days=5, sqlite_path=args.wf_path,
        )
        log.info(f"settled {n_settled} walk-forward predictions before comparison")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"settle of walk-forward DB failed: {exc!r} (hit-rate will be 0)")

    # Wipe any prior comparison runs.
    if Path(args.paper_db).exists():
        Path(args.paper_db).unlink()

    results = [
        _run_one(label="Biased  |regime OFF", run_id="biased_regime_off",
                 preds_path=args.biased_path, paper_db=args.paper_db,
                 start=start, end=end, regime_on=False),
        _run_one(label="Biased  |regime ON",  run_id="biased_regime_on",
                 preds_path=args.biased_path, paper_db=args.paper_db,
                 start=start, end=end, regime_on=True),
        _run_one(label="WF (honest) |regime OFF", run_id="wf_regime_off",
                 preds_path=args.wf_path, paper_db=args.paper_db,
                 start=start, end=end, regime_on=False),
        _run_one(label="WF (honest) |regime ON",  run_id="wf_regime_on",
                 preds_path=args.wf_path, paper_db=args.paper_db,
                 start=start, end=end, regime_on=True),
    ]
    for r in results:
        log.info(
            f"  {r.label}: Sharpe={r.sharpe} final={r.final_equity} "
            f"dd={r.max_dd_pct} trades={r.n_trades}"
        )

    wf_summary = {}
    # report.json lives next to predictions.sqlite when wf was run via
    # walkforward_backtest.py with a custom out-dir, so look there first.
    wf_report = Path(args.wf_path).parent / "report.json"
    if not wf_report.exists():
        wf_report = _WF_DIR / "report.json"
    if wf_report.exists():
        wf_summary = json.loads(wf_report.read_text())

    md = _build_report(
        results, biased_path=args.biased_path, wf_path=args.wf_path,
        universe=args.universe, start=start, end=end, wf_summary=wf_summary,
    )
    # utf-8 explicit because Windows defaults to cp1252 which chokes on
    # the arrow / em-dash chars we use in the report headings.
    Path(args.out).write_text(md, encoding="utf-8")
    log.info(f"wrote {args.out}")
    # Echo to stdout encoded as utf-8 for visibility, falling back if
    # stdout's encoding (cp1252 on Windows) chokes on arrows/em-dashes.
    try:
        sys.stdout.buffer.write(md.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
    except Exception:  # noqa: BLE001
        sys.stdout.write(md.encode("ascii", "replace").decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
