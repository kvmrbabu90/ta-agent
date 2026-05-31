"""Order engine: translates Kubera signals into Alpaca orders.

Two modes of operation:
  - **Manual approval (default)**: stage signals to kubera_alpaca_signals
    as PENDING_APPROVAL, return them to the dashboard, wait for user to
    approve via /live-alpaca/signals/approve. Only then submit to Alpaca.
  - **Auto (opt-in)**: caller skips the approval gate.

Order types Kubera uses on Alpaca:
  - **Entry (OPEN_LONG)**: MarketOrder (TIF=day) submitted at 9:31 ET.
  - **Stop**: GTC StopOrder placed immediately after entry fill at
    target_price * (1 - stop_pct). Refreshed by the sync loop once the
    actual fill price is known.
  - **Exit (CLOSE_LONG)**: MarketOrder with TIF=cls (Market-on-Close) at
    3:55 ET on day-5. Fills at the official 4 PM closing print.

Risk caps (enforced before any Alpaca call):
  - `max_position_pct`: per-position dollar size <= max_position_pct * NAV.
  - `max_daily_loss_pct`: if today's equity has dropped X% vs last_equity,
    refuse all new entries for the rest of the day.
  - **Live mode** additionally requires `confirm_live_account_number` to
    equal the connected account_number — typo-protection.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional

from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest

from . import db
from .connection import KuberaAlpaca, Mode

log = logging.getLogger("kubera.alpaca.orders")

DEFAULT_STOP_PCT = 0.05
DEFAULT_MAX_POSITION_PCT = 0.25
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT
    stop_pct: float = DEFAULT_STOP_PCT
    require_manual_approval: bool = True
    # Live-mode typo gate: must match the connected account_number.
    confirm_live_account_number: Optional[str] = None


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
            INSERT INTO kubera_alpaca_signals
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
            FROM kubera_alpaca_signals
            WHERE status = 'PENDING_APPROVAL'
            ORDER BY signal_date, id
            """
        ).fetchall()
        return [SignalRow(*r) for r in rows]
    finally:
        con.close()


def _mark(con, sig_id: int, status: str, *, alpaca_order_id: Optional[str] = None,
            approved_by: Optional[str] = None) -> None:
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    con.execute(
        """
        UPDATE kubera_alpaca_signals
        SET status = ?,
            alpaca_order_id = COALESCE(?, alpaca_order_id),
            approved_at = CASE WHEN ? = 'APPROVED' THEN ? ELSE approved_at END,
            approved_by = COALESCE(?, approved_by)
        WHERE id = ?
        """,
        (status, alpaca_order_id, status, now, approved_by, sig_id),
    )


def _check_risk_caps(client, qty: float, target_price: float,
                       cfg: RiskConfig) -> Optional[str]:
    """Return None if OK, else a short reason-to-reject string."""
    acct = client.get_account()
    nav = float(acct.portfolio_value or 0.0)
    if nav <= 0:
        return f"NAV {nav!r} is not positive"
    position_notional = qty * target_price
    if position_notional > cfg.max_position_pct * nav:
        return (f"position notional ${position_notional:,.0f} > "
                f"{cfg.max_position_pct*100:.0f}% of NAV ${nav:,.0f}")
    last_equity = float(acct.last_equity or 0.0)
    equity = float(acct.equity or nav)
    if last_equity > 0:
        day_pnl_pct = (equity - last_equity) / last_equity
        if day_pnl_pct <= -cfg.max_daily_loss_pct:
            return (f"day PnL {day_pnl_pct*100:+.2f}% <= -{cfg.max_daily_loss_pct*100:.0f}% "
                    f"of last_equity (circuit breaker)")
    if bool(acct.trading_blocked):
        return "account trading_blocked = true (Alpaca-side)"
    return None


def _ensure_live_account_matches(session, cfg: RiskConfig) -> Optional[str]:
    if session.mode != "live":
        return None
    if not cfg.confirm_live_account_number:
        return ("live mode requires confirm_live_account_number in RiskConfig "
                "(typo-protection); refused")
    if cfg.confirm_live_account_number != session.account_number:
        return (f"confirm_live_account_number={cfg.confirm_live_account_number!r} "
                f"does not match connected account {session.account_number!r}; refused")
    return None


def approve_and_submit(signal_ids: list[int], *, approved_by: str,
                         cfg: Optional[RiskConfig] = None,
                         mode: Optional[Mode] = None) -> dict:
    """Approve a batch of pending signals and submit them as Alpaca orders.

    Returns a per-signal result dict: {signal_id: {"status": str, "reason": str|None,
    "alpaca_order_id": str|None}}.
    """
    cfg = cfg or RiskConfig()
    wrapper = KuberaAlpaca(mode=mode)
    wrapper.connect()
    sess = wrapper.session
    assert sess is not None

    live_err = _ensure_live_account_matches(sess, cfg)
    if live_err:
        wrapper.disconnect()
        raise RuntimeError(live_err)

    client = wrapper.client
    out: dict[int, dict] = {}
    con = db.connect()
    try:
        rows = list_pending()
        by_id = {r.id: r for r in rows}
        for sid in signal_ids:
            sig = by_id.get(sid)
            if sig is None:
                out[sid] = {"status": "NOT_PENDING", "reason": "signal not in PENDING_APPROVAL", "alpaca_order_id": None}
                continue

            tgt = sig.target_price or 0.0
            if sig.intended_action == "OPEN_LONG" and tgt > 0:
                reason = _check_risk_caps(client, sig.qty, tgt, cfg)
                if reason:
                    _mark(con, sid, "REJECTED")
                    out[sid] = {"status": "REJECTED", "reason": reason, "alpaca_order_id": None}
                    continue

            side = OrderSide.BUY if sig.intended_action == "OPEN_LONG" else OrderSide.SELL

            # For OPEN_LONG with a stop_pct, submit as an OTO (One-Triggers-Other)
            # bracket so the entry + protective stop submit atomically. Alpaca
            # rejects "place entry, then place opposite stop" as a wash-trade
            # detection (code 40310000), so this is the only correct shape.
            # CLOSE_LONG (or OPEN_LONG without a stop) goes as a plain MarketOrder.
            use_bracket = (
                sig.intended_action == "OPEN_LONG"
                and tgt > 0
                and cfg.stop_pct > 0
            )
            if use_bracket:
                stop_price = round(tgt * (1.0 - cfg.stop_pct), 2)
                req = MarketOrderRequest(
                    symbol=sig.symbol,
                    qty=sig.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.OTO,
                    stop_loss=StopLossRequest(stop_price=stop_price),
                )
            else:
                stop_price = None
                req = MarketOrderRequest(
                    symbol=sig.symbol,
                    qty=sig.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            try:
                order = client.submit_order(order_data=req)
            except Exception as e:
                _mark(con, sid, "REJECTED")
                out[sid] = {"status": "REJECTED", "reason": f"submit_order failed: {e}", "alpaca_order_id": None}
                continue

            oid = str(order.id)
            _mark(con, sid, "PLACED", alpaca_order_id=oid, approved_by=approved_by)
            out[sid] = {"status": "PLACED", "reason": None, "alpaca_order_id": oid}
            if use_bracket:
                log.info("placed %s %s qty=%g order_id=%s OTO stop=$%.2f",
                         side.value, sig.symbol, sig.qty, oid, stop_price)
            else:
                log.info("placed %s %s qty=%g order_id=%s",
                         side.value, sig.symbol, sig.qty, oid)

        con.commit()
    finally:
        con.close()
        wrapper.disconnect()

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
