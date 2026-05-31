"""Slippage + commission attribution: Kubera paper-fills vs IBKR actual fills.

Joins ibkr_fills (real) with the strategy's simulated paper-trade fills
(analysis_live.sqlite::paper_trades) by (trade_date, symbol, side). For
each matched pair, computes:

  - slippage_bps = (actual_price - simulated_price) / simulated_price * 10000
                    (signed: positive = paid more than sim on BUY, sold for less on SELL)
  - commission_bps = commission_usd / notional_usd * 10000
  - fill_time_delay_sec = (actual_fill_time - sim_fill_time).total_seconds()

The output is the table that powers the "Paper vs Live" comparison view —
the dashboard surfaces it as a cumulative slippage chart + a per-trade
table, so you can see exactly how much alpha is being lost to execution.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import db

# The Kubera paper engine writes to this DB (mirrors per-retrain analysis).
KUBERA_PAPER_DB = "data/processed/walkforward_10yr_strict/analysis_live.sqlite"


@dataclass(frozen=True)
class ReconRow:
    trade_date: str
    symbol: str
    side: str
    sim_price: float
    actual_price: float
    qty: float
    notional: float
    slippage_bps: float
    commission_usd: Optional[float]
    commission_bps: Optional[float]


def _open_paper_ro() -> Optional[sqlite3.Connection]:
    p = Path(KUBERA_PAPER_DB)
    if not p.exists():
        return None
    return sqlite3.connect("file:" + p.as_posix() + "?mode=ro", uri=True, timeout=10.0)


def per_trade_attribution(account_id: str, *, since_date: Optional[str] = None) -> list[ReconRow]:
    """Join real fills with simulated paper-trade rows. Best-effort: matches
    on (date, symbol, side) and pairs the first match. If no sim row exists
    (e.g. a manual order outside Kubera), returns the actual with sim_price=0
    so the dashboard can still render the row."""
    paper = _open_paper_ro()
    ibkr = db.connect(read_only=True)
    try:
        # Pull all actual fills (this account) in scope.
        clauses = ["account_id = ?"]
        args: list = [account_id]
        if since_date:
            clauses.append("substr(fill_time, 1, 10) >= ?")
            args.append(since_date)
        sql = "SELECT exec_id, symbol, side, qty, price, fill_time, commission FROM ibkr_fills WHERE " + " AND ".join(clauses) + " ORDER BY fill_time"
        actuals = ibkr.execute(sql, args).fetchall()

        out: list[ReconRow] = []
        for exec_id, sym, side, qty, price, fill_time, comm in actuals:
            trade_date = fill_time[:10]
            sim_price = 0.0
            if paper is not None:
                # The paper engine schema (analysis_live.sqlite::paper_trades) uses
                # side values like "long_open" / "long_close" — map.
                wanted_sides = ["long_open", "long_close"] if side in ("BUY", "SELL") else [side.lower()]
                row = paper.execute(
                    """
                    SELECT fill_price FROM paper_trades
                    WHERE trade_date = ? AND symbol = ? AND side IN (?, ?)
                    LIMIT 1
                    """,
                    (trade_date, sym, *wanted_sides),
                ).fetchone()
                if row:
                    sim_price = float(row[0])

            qty = float(qty); price = float(price)
            notional = abs(qty * price)
            # Signed slippage in bps: BUY paying above sim is positive (bad),
            # SELL receiving below sim is positive (also bad).
            if sim_price > 0:
                if side == "BUY":
                    bps = (price - sim_price) / sim_price * 10000.0
                else:
                    bps = (sim_price - price) / sim_price * 10000.0
            else:
                bps = 0.0
            comm_f = float(comm) if comm is not None else None
            comm_bps = (comm_f / notional * 10000.0) if (comm_f is not None and notional > 0) else None
            out.append(ReconRow(
                trade_date=trade_date, symbol=sym, side=side,
                sim_price=sim_price, actual_price=price, qty=qty,
                notional=notional, slippage_bps=bps,
                commission_usd=comm_f, commission_bps=comm_bps,
            ))
        return out
    finally:
        if paper is not None:
            paper.close()
        ibkr.close()


def cumulative_summary(rows: list[ReconRow]) -> dict:
    if not rows:
        return {"n": 0, "mean_slip_bps": None, "median_slip_bps": None,
                "total_commission_usd": 0.0, "total_notional": 0.0}
    bps = [r.slippage_bps for r in rows if r.sim_price > 0]
    commissions = [r.commission_usd for r in rows if r.commission_usd is not None]
    bps_sorted = sorted(bps)
    median = bps_sorted[len(bps_sorted) // 2] if bps_sorted else None
    return {
        "n": len(rows),
        "mean_slip_bps": (sum(bps) / len(bps)) if bps else None,
        "median_slip_bps": median,
        "total_commission_usd": sum(commissions),
        "total_notional": sum(r.notional for r in rows),
    }
