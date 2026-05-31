"""Live IBKR dashboard endpoints — reads from ibkr_live.sqlite written by
the background sync loop. The API does not touch IB directly here; that
keeps requests fast and the dashboard usable even if Gateway is briefly
disconnected.

Endpoints under /live-ibkr:
  GET  /status                  — latest known mode/account/connection state
  GET  /positions               — most-recent snapshot of holdings
  GET  /orders/today            — all orders placed today
  GET  /equity                  — NAV timeseries
  GET  /reconciliation          — per-trade slippage table + summary
  GET  /signals/pending          — pending-approval signals
  POST /signals/approve         — approve a batch and submit to IB
  POST /signals/reject          — reject a batch
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.ibkr import db as ibkr_db
from services.ibkr import reconciliation as recon
from services.ibkr import orders as orders_module
from services.ibkr.orders import RiskConfig

router = APIRouter(prefix="/live-ibkr", tags=["live-ibkr"])


class StatusResponse(BaseModel):
    connected: bool
    mode: Optional[str] = None         # 'paper' | 'live'
    surface: Optional[str] = None      # 'gateway' | 'tws'
    account_id: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    last_seen_at: Optional[str] = None
    reason: Optional[str] = None       # populated when connected=False


class PositionRow(BaseModel):
    symbol: str
    qty: float
    avg_cost: float
    mkt_price: Optional[float] = None
    mkt_value: Optional[float] = None
    unreal_pnl: Optional[float] = None


class OrderRow(BaseModel):
    order_id: int
    symbol: str
    side: str
    qty: float
    order_type: str
    status: str
    filled_qty: float
    avg_fill_price: Optional[float] = None
    placed_at: str


class EquityPoint(BaseModel):
    snapshot_at: str
    nav: float
    cash: float
    long_mv: float


class ReconciliationRow(BaseModel):
    trade_date: str
    symbol: str
    side: str
    sim_price: float
    actual_price: float
    qty: float
    notional: float
    slippage_bps: float
    commission_usd: Optional[float] = None
    commission_bps: Optional[float] = None


class ReconciliationResponse(BaseModel):
    rows: list[ReconciliationRow]
    n: int
    mean_slip_bps: Optional[float] = None
    median_slip_bps: Optional[float] = None
    total_commission_usd: float = 0.0
    total_notional: float = 0.0


class PendingSignal(BaseModel):
    id: int
    signal_date: str
    intended_action: str
    symbol: str
    qty: float
    target_price: Optional[float] = None
    status: str


class ApproveRequest(BaseModel):
    signal_ids: list[int]
    approved_by: str = "dashboard"
    # Live-mode safety: dashboard must pass the connected account id back to
    # confirm the user intends to act on it.
    confirm_live_account_id: Optional[str] = None


class ApproveResult(BaseModel):
    signal_id: int
    status: str
    reason: Optional[str] = None
    ibkr_order_id: Optional[int] = None


def _latest_account() -> Optional[tuple]:
    """Returns (account_id, mode, surface, host, port, last_seen_at) or None."""
    con = ibkr_db.connect(read_only=True)
    try:
        row = con.execute(
            "SELECT account_id, mode, surface, host, port, last_seen_at "
            "FROM ibkr_account_meta ORDER BY last_seen_at DESC LIMIT 1"
        ).fetchone()
        return row
    finally:
        con.close()


def _is_fresh(last_seen_at: str, *, max_age_sec: int = 120) -> bool:
    try:
        t = dt.datetime.fromisoformat(last_seen_at.rstrip("Z"))
    except ValueError:
        return False
    return (dt.datetime.utcnow() - t).total_seconds() < max_age_sec


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    row = _latest_account()
    if row is None:
        return StatusResponse(connected=False, reason="no IBKR session ever recorded")
    account_id, mode, surface, host, port, last_seen = row
    fresh = _is_fresh(last_seen)
    return StatusResponse(
        connected=fresh,
        mode=mode,
        surface=surface,
        account_id=account_id,
        host=host,
        port=port,
        last_seen_at=last_seen,
        reason=None if fresh else "sync stale (no update in 2+ min)",
    )


@router.get("/positions", response_model=list[PositionRow])
def get_positions() -> list[PositionRow]:
    row = _latest_account()
    if row is None:
        return []
    account_id = row[0]
    con = ibkr_db.connect(read_only=True)
    try:
        latest_snapshot = con.execute(
            "SELECT MAX(snapshot_at) FROM ibkr_positions WHERE account_id = ?",
            (account_id,),
        ).fetchone()[0]
        if not latest_snapshot:
            return []
        rows = con.execute(
            """SELECT symbol, qty, avg_cost, mkt_price, mkt_value, unreal_pnl
               FROM ibkr_positions
               WHERE account_id = ? AND snapshot_at = ?
               ORDER BY mkt_value DESC NULLS LAST""",
            (account_id, latest_snapshot),
        ).fetchall()
    finally:
        con.close()
    return [PositionRow(symbol=r[0], qty=r[1], avg_cost=r[2], mkt_price=r[3],
                          mkt_value=r[4], unreal_pnl=r[5]) for r in rows]


@router.get("/orders/today", response_model=list[OrderRow])
def get_orders_today() -> list[OrderRow]:
    row = _latest_account()
    if row is None:
        return []
    account_id = row[0]
    today = dt.date.today().isoformat()
    con = ibkr_db.connect(read_only=True)
    try:
        rows = con.execute(
            """SELECT order_id, symbol, side, qty, order_type, status, filled_qty,
                      avg_fill_price, placed_at
               FROM ibkr_orders
               WHERE account_id = ? AND substr(placed_at, 1, 10) = ?
               ORDER BY placed_at""",
            (account_id, today),
        ).fetchall()
    finally:
        con.close()
    return [OrderRow(order_id=r[0], symbol=r[1], side=r[2], qty=r[3], order_type=r[4],
                       status=r[5], filled_qty=r[6], avg_fill_price=r[7],
                       placed_at=r[8]) for r in rows]


@router.get("/equity", response_model=list[EquityPoint])
def get_equity() -> list[EquityPoint]:
    row = _latest_account()
    if row is None:
        return []
    account_id = row[0]
    con = ibkr_db.connect(read_only=True)
    try:
        rows = con.execute(
            "SELECT snapshot_at, nav, cash, long_mv FROM ibkr_equity "
            "WHERE account_id = ? ORDER BY snapshot_at",
            (account_id,),
        ).fetchall()
    finally:
        con.close()
    return [EquityPoint(snapshot_at=r[0], nav=r[1], cash=r[2], long_mv=r[3]) for r in rows]


@router.get("/reconciliation", response_model=ReconciliationResponse)
def get_reconciliation(since_date: Optional[str] = None) -> ReconciliationResponse:
    row = _latest_account()
    if row is None:
        return ReconciliationResponse(rows=[], n=0)
    account_id = row[0]
    recon_rows = recon.per_trade_attribution(account_id, since_date=since_date)
    summary = recon.cumulative_summary(recon_rows)
    return ReconciliationResponse(
        rows=[ReconciliationRow(**r.__dict__) for r in recon_rows],
        **summary,
    )


@router.get("/signals/pending", response_model=list[PendingSignal])
def get_pending_signals() -> list[PendingSignal]:
    pending = orders_module.list_pending()
    return [PendingSignal(id=p.id, signal_date=p.signal_date,
                              intended_action=p.intended_action, symbol=p.symbol,
                              qty=p.qty, target_price=p.target_price, status=p.status)
            for p in pending]


@router.post("/signals/approve", response_model=list[ApproveResult])
def approve_signals(req: ApproveRequest) -> list[ApproveResult]:
    if not req.signal_ids:
        raise HTTPException(status_code=400, detail="signal_ids must not be empty")
    row = _latest_account()
    if row is None:
        raise HTTPException(status_code=503, detail="no IBKR session recorded")

    cfg = RiskConfig(confirm_live_account_id=req.confirm_live_account_id)
    try:
        result = orders_module.approve_and_submit(
            req.signal_ids, approved_by=req.approved_by, cfg=cfg,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [ApproveResult(signal_id=sid, **{k: v for k, v in r.items()})
            for sid, r in result.items()]


class RejectRequest(BaseModel):
    signal_ids: list[int]
    reason: str = "manual"


@router.post("/signals/reject")
def reject_signals(req: RejectRequest) -> dict:
    if not req.signal_ids:
        raise HTTPException(status_code=400, detail="signal_ids must not be empty")
    n = orders_module.reject(req.signal_ids, reason=req.reason)
    return {"rejected": n}
