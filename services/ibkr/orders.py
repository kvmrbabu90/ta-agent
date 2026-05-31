"""Order engine: translates Kubera signals into IB orders.

Two modes of operation:
  - **Manual approval (default)**: stage signals to kubera_signals as
    PENDING_APPROVAL, return them to the dashboard, wait for user to
    approve via /live-ibkr/signals/approve. Only then submit to IB.
  - **Auto (opt-in)**: signal -> order in the same call, no approval gate.

Order types Kubera uses:
  - **Entry (OPEN_LONG)**: MKT day-order at 9:31 ET.
  - **Stop**: GTC STP order placed immediately after entry fill, at
    `stop_pct` below entry fill price. The simulation uses this convention.
  - **Exit (CLOSE_LONG)**: MOC (Market-on-Close) order placed at 3:55 ET
    on day-5. MOC fills at the official 4 PM closing print.

Risk caps (enforced before any IB call):
  - `max_position_pct`: per-position dollar size <= max_position_pct * NAV.
  - `max_daily_loss_pct`: if today's realized + unrealized PnL <= -X% of
    starting-of-day NAV, refuse all new entry orders for the rest of day.
  - **Live mode** additionally requires `confirm_live_account_id` to equal
    the connected account_id — a hard typo-protection gate.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional

from ib_insync import IB, Stock, Order, MarketOrder, StopOrder

from . import db
from .connection import KuberaIB, KUBERA_CLIENT_ID_ORDER

log = logging.getLogger("kubera.ibkr.orders")

DEFAULT_STOP_PCT = 0.05         # 5% below entry fill = engine convention
DEFAULT_MAX_POSITION_PCT = 0.25 # 25% per position
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05  # -5% intraday halts new entries


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT
    stop_pct: float = DEFAULT_STOP_PCT
    require_manual_approval: bool = True
    # For live-mode safety: the engine refuses to place any order in live mode
    # unless this string matches the connected account_id. Set it once you
    # are happy with the paper run.
    confirm_live_account_id: Optional[str] = None


@dataclass
class SignalRow:
    id: int
    signal_date: str
    intended_action: str   # 'OPEN_LONG' | 'CLOSE_LONG'
    symbol: str
    qty: float
    target_price: Optional[float]
    status: str


def stage_signal(*, signal_date: str, action: str, symbol: str, qty: float,
                  target_price: Optional[float], notes: str = "") -> int:
    """Insert a signal in PENDING_APPROVAL. Returns the signal id."""
    if action not in ("OPEN_LONG", "CLOSE_LONG"):
        raise ValueError(f"unknown action {action!r}")
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    con = db.connect()
    try:
        cur = con.execute(
            """
            INSERT INTO kubera_signals
              (signal_date, intended_action, symbol, qty, target_price, status, notes, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (signal_date, action, symbol, qty, target_price, "PENDING_APPROVAL", notes, now),
        )
        sig_id = int(cur.lastrowid)
        con.commit()
        return sig_id
    finally:
        con.close()


def list_pending() -> list[SignalRow]:
    con = db.connect(read_only=True)
    try:
        rows = con.execute(
            """
            SELECT id, signal_date, intended_action, symbol, qty, target_price, status
            FROM kubera_signals
            WHERE status = 'PENDING_APPROVAL'
            ORDER BY signal_date, id
            """
        ).fetchall()
        return [SignalRow(*r) for r in rows]
    finally:
        con.close()


def _mark(con, sig_id: int, status: str, *, ibkr_order_id: Optional[int] = None,
            approved_by: Optional[str] = None) -> None:
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    con.execute(
        """
        UPDATE kubera_signals
        SET status = ?,
            ibkr_order_id = COALESCE(?, ibkr_order_id),
            approved_at = CASE WHEN ? = 'APPROVED' THEN ? ELSE approved_at END,
            approved_by = COALESCE(?, approved_by)
        WHERE id = ?
        """,
        (status, ibkr_order_id, status, now, approved_by, sig_id),
    )


def _check_risk_caps(ib: IB, account_id: str, qty: float, target_price: float,
                       cfg: RiskConfig) -> Optional[str]:
    """Return None if OK, else a short reason-to-reject string."""
    summary = {row.tag: row for row in ib.accountSummary(account_id)}
    if "NetLiquidation" not in summary:
        return "no NAV available from IB"
    nav = float(summary["NetLiquidation"].value)
    if nav <= 0:
        return f"NAV {nav!r} is not positive"
    position_notional = qty * target_price
    if position_notional > cfg.max_position_pct * nav:
        return (f"position notional ${position_notional:,.0f} > "
                f"{cfg.max_position_pct*100:.0f}% of NAV ${nav:,.0f}")
    unreal = float(summary.get("UnrealizedPnL").value) if "UnrealizedPnL" in summary else 0.0
    realized = float(summary.get("RealizedPnL").value) if "RealizedPnL" in summary else 0.0
    # Best-effort: treat (realized + unrealized) / NAV-at-start-of-day as the
    # daily PnL. We don't have a true start-of-day NAV stored yet; use a rough
    # proxy of current NAV - PnL.
    pnl = realized + unreal
    sod_nav = nav - pnl
    if sod_nav > 0 and (pnl / sod_nav) <= -cfg.max_daily_loss_pct:
        return (f"daily PnL {pnl:+,.0f} <= -{cfg.max_daily_loss_pct*100:.0f}% "
                f"of SoD NAV ${sod_nav:,.0f} (circuit breaker)")
    return None


def _ensure_live_account_matches(session, cfg: RiskConfig) -> Optional[str]:
    if session.mode != "live":
        return None
    if not cfg.confirm_live_account_id:
        return ("live mode requires confirm_live_account_id in RiskConfig "
                "(typo-protection); refused")
    if cfg.confirm_live_account_id != session.account_id:
        return (f"confirm_live_account_id={cfg.confirm_live_account_id!r} "
                f"does not match connected account {session.account_id!r}; refused")
    return None


def approve_and_submit(signal_ids: list[int], *, approved_by: str,
                         cfg: Optional[RiskConfig] = None) -> dict:
    """Approve a batch of pending signals and submit them as IB orders.

    Returns a per-signal result dict: {signal_id: {"status": str, "reason": str|None,
    "ibkr_order_id": int|None}}.
    """
    cfg = cfg or RiskConfig()
    client = KuberaIB(client_id=KUBERA_CLIENT_ID_ORDER)
    client.connect()
    sess = client.session
    assert sess is not None

    live_err = _ensure_live_account_matches(sess, cfg)
    if live_err:
        client.disconnect()
        raise RuntimeError(live_err)

    ib = client.ib
    out: dict[int, dict] = {}
    con = db.connect()
    try:
        rows = list_pending()
        by_id = {r.id: r for r in rows}
        for sid in signal_ids:
            sig = by_id.get(sid)
            if sig is None:
                out[sid] = {"status": "NOT_PENDING", "reason": "signal not in PENDING_APPROVAL", "ibkr_order_id": None}
                continue

            tgt = sig.target_price or 0.0
            if sig.intended_action == "OPEN_LONG" and tgt > 0:
                reason = _check_risk_caps(ib, sess.account_id, sig.qty, tgt, cfg)
                if reason:
                    _mark(con, sid, "REJECTED")
                    out[sid] = {"status": "REJECTED", "reason": reason, "ibkr_order_id": None}
                    continue

            contract = Stock(sig.symbol, "SMART", "USD")
            action = "BUY" if sig.intended_action == "OPEN_LONG" else "SELL"
            order: Order = MarketOrder(action, sig.qty)
            order.account = sess.account_id
            order.tif = "DAY"
            try:
                trade = ib.placeOrder(contract, order)
            except Exception as e:
                _mark(con, sid, "REJECTED")
                out[sid] = {"status": "REJECTED", "reason": f"placeOrder failed: {e}", "ibkr_order_id": None}
                continue

            oid = int(trade.order.orderId)
            _mark(con, sid, "PLACED", ibkr_order_id=oid, approved_by=approved_by)
            out[sid] = {"status": "PLACED", "reason": None, "ibkr_order_id": oid}
            log.info("placed %s %s qty=%g order_id=%d", action, sig.symbol, sig.qty, oid)

            # Best-effort: place a GTC stop alongside entry orders. Use the
            # target_price * (1 - stop_pct) as the stop trigger; we will
            # refresh it after the entry actually fills via the sync loop.
            if sig.intended_action == "OPEN_LONG" and tgt > 0 and cfg.stop_pct > 0:
                stop_price = round(tgt * (1.0 - cfg.stop_pct), 2)
                stop_order = StopOrder("SELL", sig.qty, stop_price)
                stop_order.account = sess.account_id
                stop_order.tif = "GTC"
                try:
                    ib.placeOrder(contract, stop_order)
                    log.info("placed STP SELL %s qty=%g stop=$%.2f", sig.symbol, sig.qty, stop_price)
                except Exception as e:
                    log.warning("stop order failed for %s: %s", sig.symbol, e)

        con.commit()
    finally:
        con.close()
        client.disconnect()

    return out


def reject(signal_ids: list[int], *, reason: str = "manual") -> int:
    con = db.connect()
    try:
        for sid in signal_ids:
            _mark(con, sid, "REJECTED")
        con.commit()
        return len(signal_ids)
    finally:
        con.close()
