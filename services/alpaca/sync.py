"""Background sync: poll Alpaca state and persist to alpaca_live.sqlite.

Run as:  python -m services.alpaca.sync [--interval 30] [--mode paper|live]

The sync loop is the source of truth for the dashboard — the API reads
from the SQLite tables, never from Alpaca directly. Same design as the
IBKR sync (see services/ibkr/sync.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import sys
import time
from typing import Optional

from alpaca.trading.enums import OrderStatus, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from .connection import KuberaAlpaca, Mode
from . import db

log = logging.getLogger("kubera.alpaca.sync")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_stop = False


def _handle_signal(_signum, _frame):
    global _stop
    _stop = True
    log.info("stop signal received; exiting after current tick")


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ts_to_iso(ts) -> str:
    """Coerce an Alpaca datetime (already TZ-aware UTC) to ISO-Z string."""
    if ts is None:
        return _now_iso()
    if isinstance(ts, str):
        return ts
    try:
        return ts.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception:
        return str(ts)


def _sync_account_summary(client, con, account_number: str, now: str) -> None:
    acct = client.get_account()
    nav = float(acct.portfolio_value or 0.0)
    cash = float(acct.cash or 0.0)
    long_mv = float(acct.long_market_value or 0.0)
    equity = float(acct.equity or 0.0)
    bp = float(acct.buying_power or 0.0)
    ccy = str(acct.currency or "USD")
    db.insert_equity_snapshot(
        con,
        account_number=account_number, snapshot_at=now,
        nav=nav, cash=cash, long_mv=long_mv, equity=equity,
        buying_power=bp, currency=ccy,
    )


def _sync_positions(client, con, account_number: str, now: str) -> None:
    positions = client.get_all_positions()
    rows: list[dict] = []
    for p in positions:
        qty = float(p.qty or 0.0)
        if qty == 0:
            continue
        rows.append({
            "symbol": p.symbol,
            "qty": qty,
            "avg_entry_price": float(p.avg_entry_price or 0.0),
            "mkt_price": float(p.current_price) if p.current_price is not None else None,
            "mkt_value": float(p.market_value) if p.market_value is not None else None,
            "unreal_pnl": float(p.unrealized_pl) if p.unrealized_pl is not None else None,
            "unreal_pnl_pct": float(p.unrealized_plpc) if p.unrealized_plpc is not None else None,
            "side": str(p.side).lower() if p.side else "long",
        })
    db.replace_positions_snapshot(con, account_number=account_number, snapshot_at=now, rows=rows)


def _sync_orders_and_fills(client, con, account_number: str, now: str,
                              since: Optional[dt.datetime] = None) -> None:
    """Pull today's orders + any fills attached to them.

    Alpaca's REST `get_orders` returns Order objects with `filled_qty` and
    `filled_avg_price`. Per-execution fill detail comes either from
    trade_updates streaming (richer; preferred) or from order.legs for
    bracket orders. The sync loop here uses REST as a pragmatic baseline
    so the dashboard works even when the WS daemon isn't running.
    """
    after = since or (dt.datetime.utcnow() - dt.timedelta(days=7))
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=after,
        limit=500,
        direction="desc",
    )
    orders = client.get_orders(filter=req)
    for o in orders:
        oid = str(o.id)
        status = str(o.status.value if hasattr(o.status, "value") else o.status)
        filled_qty = float(o.filled_qty or 0.0)
        filled_avg = float(o.filled_avg_price) if o.filled_avg_price is not None else None
        db.upsert_order(
            con,
            account_number=account_number,
            order_id=oid,
            client_order_id=str(o.client_order_id) if o.client_order_id else None,
            symbol=str(o.symbol),
            side=str(o.side.value if hasattr(o.side, "value") else o.side),
            qty=float(o.qty or 0.0),
            order_type=str(o.order_type.value if hasattr(o.order_type, "value") else (o.order_type or o.type)),
            limit_price=float(o.limit_price) if o.limit_price is not None else None,
            stop_price=float(o.stop_price) if o.stop_price is not None else None,
            tif=str(o.time_in_force.value if hasattr(o.time_in_force, "value") else o.time_in_force),
            status=status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg,
            submitted_at=_ts_to_iso(o.submitted_at or o.created_at),
            last_update_at=_ts_to_iso(o.updated_at or now),
            kubera_signal_id=None,
        )
        # Synthesize a single fill record when the order is fully or partially
        # filled. exec_id is deterministic from order_id so re-runs are idempotent.
        # For per-execution detail, run the trade_updates stream daemon.
        if filled_qty > 0 and filled_avg is not None:
            exec_id = f"rest-{oid}"
            db.insert_fill(
                con,
                account_number=account_number,
                exec_id=exec_id,
                order_id=oid,
                symbol=str(o.symbol),
                side=str(o.side.value if hasattr(o.side, "value") else o.side),
                qty=filled_qty,
                price=filled_avg,
                fill_time=_ts_to_iso(o.filled_at or o.updated_at or now),
                commission=0.0,   # Alpaca stock trades are commission-free today
                realized_pnl=None,
            )


def tick(client_wrapper: KuberaAlpaca, con) -> None:
    """One sync iteration. Caller commits."""
    sess = client_wrapper.session
    assert sess is not None
    client = client_wrapper.client
    now = _now_iso()
    db.upsert_account_meta(
        con,
        account_number=sess.account_number, account_id=sess.account_id,
        mode=sess.mode, status=sess.status, currency=sess.currency, now_iso=now,
    )
    _sync_account_summary(client, con, sess.account_number, now)
    _sync_positions(client, con, sess.account_number, now)
    _sync_orders_and_fills(client, con, sess.account_number, now)


def run(interval: float = 30.0, mode: Optional[Mode] = None) -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    wrapper = KuberaAlpaca(mode=mode)
    while not _stop:
        try:
            if not wrapper.is_connected():
                wrapper.connect()
            con = db.connect()
            try:
                tick(wrapper, con)
                con.commit()
            finally:
                con.close()
        except Exception as e:
            log.warning("tick failed: %s; will retry in %.0fs", e, interval)
            try:
                wrapper.disconnect()
            except Exception:
                pass
        slept = 0.0
        while slept < interval and not _stop:
            time.sleep(1.0)
            slept += 1.0
    wrapper.disconnect()
    log.info("sync stopped cleanly")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=30.0, help="seconds between polls")
    p.add_argument("--mode", choices=["paper", "live"], default=None,
                   help="override ALPACA_MODE env var")
    args = p.parse_args()
    sys.exit(run(args.interval, mode=args.mode))
