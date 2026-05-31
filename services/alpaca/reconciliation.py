"""Slippage attribution: Kubera paper fills vs Alpaca actual fills.

Joins alpaca_fills (real) with the strategy's simulated paper-trade fills
(analysis_live.sqlite::paper_trades) by (trade_date, symbol, side). For
each matched pair, computes:

  - slippage_bps = signed: positive = paid more on BUY / received less on SELL
  - commission_bps = commission_usd / notional_usd * 10000 (~0 for stocks)
  - fill_time_delay_sec is implicit in the timestamps and surfaced by the API

Notes vs IBKR:
  - Alpaca side values are 'buy' / 'sell' (lowercase).
  - Commission on Alpaca stocks is zero today; the column is kept for
    parity with the IBKR schema and future fee surfacing.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import db

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


def per_trade_attribution(account_number: str, *, since_date: Optional[str] = None) -> list[ReconRow]:
    """Join real Alpaca fills with simulated paper-trade rows."""
    paper = _open_paper_ro()
    alp = db.connect(read_only=True)
    try:
        clauses = ["account_number = ?"]
        args: list = [account_number]
        if since_date:
            clauses.append("substr(fill_time, 1, 10) >= ?")
            args.append(since_date)
        sql = (
            "SELECT exec_id, symbol, side, qty, price, fill_time, commission "
            "FROM alpaca_fills WHERE " + " AND ".join(clauses) +
            " ORDER BY fill_time"
        )
        actuals = alp.execute(sql, args).fetchall()

        out: list[ReconRow] = []
        for exec_id, sym, side, qty, price, fill_time, comm in actuals:
            trade_date = fill_time[:10]
            sim_price = 0.0
            side_norm = (side or "").lower()
            if paper is not None:
                wanted_sides = (
                    ["long_open", "long_close"]
                    if side_norm in ("buy", "sell")
                    else [side_norm]
                )
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
            if sim_price > 0:
                if side_norm == "buy":
                    bps = (price - sim_price) / sim_price * 10000.0
                else:
                    bps = (sim_price - price) / sim_price * 10000.0
            else:
                bps = 0.0
            comm_f = float(comm) if comm is not None else None
            comm_bps = (comm_f / notional * 10000.0) if (comm_f is not None and notional > 0) else None
            out.append(ReconRow(
                trade_date=trade_date, symbol=sym, side=side_norm,
                sim_price=sim_price, actual_price=price, qty=qty,
                notional=notional, slippage_bps=bps,
                commission_usd=comm_f, commission_bps=comm_bps,
            ))
        return out
    finally:
        if paper is not None:
            paper.close()
        alp.close()


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
