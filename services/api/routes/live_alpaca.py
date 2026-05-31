"""Live Alpaca dashboard endpoints — reads from alpaca_live.sqlite written
by the background sync loop.

Endpoints under /live-alpaca:
  GET  /status                  — latest known mode/account/connection state
  GET  /positions               — most-recent snapshot of holdings
  GET  /orders/today            — all orders submitted today
  GET  /equity                  — NAV timeseries
  GET  /reconciliation          — per-trade slippage table + summary
  GET  /signals/pending          — pending-approval signals
  POST /signals/approve         — approve a batch and submit to Alpaca
  POST /signals/reject          — reject a batch
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.alpaca import db as alp_db
from services.alpaca import reconciliation as recon
from services.alpaca import orders as orders_module
from services.alpaca import engine_state
from services.alpaca.orders import RiskConfig

router = APIRouter(prefix="/live-alpaca", tags=["live-alpaca"])


class StatusResponse(BaseModel):
    connected: bool
    mode: Optional[str] = None          # 'paper' | 'live'
    account_number: Optional[str] = None
    account_id: Optional[str] = None
    status: Optional[str] = None        # Alpaca account status (ACTIVE, etc.)
    currency: Optional[str] = None
    last_seen_at: Optional[str] = None
    reason: Optional[str] = None


class PositionRow(BaseModel):
    symbol: str
    qty: float
    avg_entry_price: float
    mkt_price: Optional[float] = None
    mkt_value: Optional[float] = None
    unreal_pnl: Optional[float] = None
    unreal_pnl_pct: Optional[float] = None
    side: str


class OrderRow(BaseModel):
    order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    status: str
    filled_qty: float
    filled_avg_price: Optional[float] = None
    submitted_at: str


class EquityPoint(BaseModel):
    snapshot_at: str
    nav: float
    cash: float
    long_mv: float
    equity: float
    buying_power: float


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
    # Live-mode safety: dashboard must echo the connected account number back.
    confirm_live_account_number: Optional[str] = None


class ApproveResult(BaseModel):
    signal_id: int
    status: str
    reason: Optional[str] = None
    alpaca_order_id: Optional[str] = None


def _latest_account() -> Optional[tuple]:
    """(account_number, account_id, mode, status, currency, last_seen_at) or None."""
    con = alp_db.connect(read_only=True)
    try:
        row = con.execute(
            "SELECT account_number, account_id, mode, status, currency, last_seen_at "
            "FROM alpaca_account_meta ORDER BY last_seen_at DESC LIMIT 1"
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
        return StatusResponse(connected=False, reason="no Alpaca session ever recorded")
    account_number, account_id, mode, status, currency, last_seen = row
    fresh = _is_fresh(last_seen)
    return StatusResponse(
        connected=fresh,
        mode=mode,
        account_number=account_number,
        account_id=account_id,
        status=status,
        currency=currency,
        last_seen_at=last_seen,
        reason=None if fresh else "sync stale (no update in 2+ min)",
    )


@router.get("/positions", response_model=list[PositionRow])
def get_positions() -> list[PositionRow]:
    row = _latest_account()
    if row is None:
        return []
    account_number = row[0]
    con = alp_db.connect(read_only=True)
    try:
        latest_snapshot = con.execute(
            "SELECT MAX(snapshot_at) FROM alpaca_positions WHERE account_number = ?",
            (account_number,),
        ).fetchone()[0]
        if not latest_snapshot:
            return []
        rows = con.execute(
            """SELECT symbol, qty, avg_entry_price, mkt_price, mkt_value,
                      unreal_pnl, unreal_pnl_pct, side
               FROM alpaca_positions
               WHERE account_number = ? AND snapshot_at = ?
               ORDER BY mkt_value DESC NULLS LAST""",
            (account_number, latest_snapshot),
        ).fetchall()
    finally:
        con.close()
    return [PositionRow(symbol=r[0], qty=r[1], avg_entry_price=r[2], mkt_price=r[3],
                         mkt_value=r[4], unreal_pnl=r[5], unreal_pnl_pct=r[6],
                         side=r[7]) for r in rows]


@router.get("/orders/today", response_model=list[OrderRow])
def get_orders_today() -> list[OrderRow]:
    row = _latest_account()
    if row is None:
        return []
    account_number = row[0]
    today = dt.date.today().isoformat()
    con = alp_db.connect(read_only=True)
    try:
        rows = con.execute(
            """SELECT order_id, symbol, side, qty, order_type, status, filled_qty,
                      filled_avg_price, submitted_at
               FROM alpaca_orders
               WHERE account_number = ? AND substr(submitted_at, 1, 10) = ?
               ORDER BY submitted_at""",
            (account_number, today),
        ).fetchall()
    finally:
        con.close()
    return [OrderRow(order_id=r[0], symbol=r[1], side=r[2], qty=r[3], order_type=r[4],
                      status=r[5], filled_qty=r[6], filled_avg_price=r[7],
                      submitted_at=r[8]) for r in rows]


@router.get("/equity", response_model=list[EquityPoint])
def get_equity() -> list[EquityPoint]:
    row = _latest_account()
    if row is None:
        return []
    account_number = row[0]
    con = alp_db.connect(read_only=True)
    try:
        rows = con.execute(
            "SELECT snapshot_at, nav, cash, long_mv, equity, buying_power "
            "FROM alpaca_equity WHERE account_number = ? ORDER BY snapshot_at",
            (account_number,),
        ).fetchall()
    finally:
        con.close()
    return [EquityPoint(snapshot_at=r[0], nav=r[1], cash=r[2], long_mv=r[3],
                         equity=r[4], buying_power=r[5]) for r in rows]


@router.get("/reconciliation", response_model=ReconciliationResponse)
def get_reconciliation(since_date: Optional[str] = None) -> ReconciliationResponse:
    row = _latest_account()
    if row is None:
        return ReconciliationResponse(rows=[], n=0)
    account_number = row[0]
    recon_rows = recon.per_trade_attribution(account_number, since_date=since_date)
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
        raise HTTPException(status_code=503, detail="no Alpaca session recorded")

    cfg = RiskConfig(confirm_live_account_number=req.confirm_live_account_number)
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


# ----------------------------------------------------------------------
# Kubera engine — start/stop/status
# ----------------------------------------------------------------------

class EngineStatusResponse(BaseModel):
    status: str                          # 'stopped' | 'running' | 'error'
    pid: Optional[int] = None
    sync_pid: Optional[int] = None
    engine_alive: bool = False
    sync_alive: bool = False
    started_at: Optional[str] = None
    last_run_at: Optional[str] = None
    last_run_date: Optional[str] = None
    last_run_status: Optional[str] = None
    last_error: Optional[str] = None
    heartbeat_at: Optional[str] = None
    stopped_at: Optional[str] = None


def _to_response(s: engine_state.EngineStatus) -> EngineStatusResponse:
    return EngineStatusResponse(
        status=s.status, pid=s.pid, sync_pid=s.sync_pid,
        engine_alive=s.engine_alive, sync_alive=s.sync_alive,
        started_at=s.started_at, last_run_at=s.last_run_at,
        last_run_date=s.last_run_date, last_run_status=s.last_run_status,
        last_error=s.last_error, heartbeat_at=s.heartbeat_at,
        stopped_at=s.stopped_at,
    )


@router.get("/engine/status", response_model=EngineStatusResponse)
def engine_status_endpoint() -> EngineStatusResponse:
    return _to_response(engine_state.status())


@router.post("/engine/start", response_model=EngineStatusResponse)
def engine_start_endpoint() -> EngineStatusResponse:
    try:
        s = engine_state.start()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _to_response(s)


@router.post("/engine/stop", response_model=EngineStatusResponse)
def engine_stop_endpoint() -> EngineStatusResponse:
    return _to_response(engine_state.stop())
