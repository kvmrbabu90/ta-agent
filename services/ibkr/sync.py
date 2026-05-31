"""Background sync: poll IBKR state and persist to ibkr_live.sqlite.

Run as: python -m services.ibkr.sync [--interval 30]

The sync loop is the source of truth for the dashboard — the API reads from
the SQLite tables, never from the live IB session. This means: the API stays
fast (no IB round-trip per request), the dashboard works even if IB is
briefly disconnected, and the historical equity timeseries is preserved.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import sys
import time
from typing import Optional

from ib_insync import Position, Trade

from .connection import KuberaIB, KUBERA_CLIENT_ID_SYNC
from . import db

log = logging.getLogger("kubera.ibkr.sync")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_stop = False


def _handle_signal(_signum, _frame):
    global _stop
    _stop = True
    log.info("stop signal received; exiting after current tick")


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ibkr_status_to_str(s) -> str:
    """Coerce ib_insync's OrderStatus to a clean string."""
    return str(s) if s is not None else ""


def _sync_account_summary(ib, con, account_id: str, now: str) -> None:
    summary = {row.tag: row for row in ib.accountSummary(account_id)}
    nav = float(summary.get("NetLiquidation").value) if "NetLiquidation" in summary else 0.0
    cash = float(summary.get("TotalCashValue").value) if "TotalCashValue" in summary else 0.0
    gross = float(summary.get("GrossPositionValue").value) if "GrossPositionValue" in summary else 0.0
    realized = float(summary.get("RealizedPnL").value) if "RealizedPnL" in summary else 0.0
    unreal = float(summary.get("UnrealizedPnL").value) if "UnrealizedPnL" in summary else 0.0
    ccy = summary.get("NetLiquidation").currency if "NetLiquidation" in summary else "USD"
    db.insert_equity_snapshot(
        con,
        account_id=account_id, snapshot_at=now,
        nav=nav, cash=cash, long_mv=gross,
        realized=realized, unreal=unreal, currency=ccy,
    )


def _sync_positions(ib, con, account_id: str, now: str) -> None:
    positions: list[Position] = ib.positions(account=account_id)
    rows = []
    for p in positions:
        sym = p.contract.symbol
        qty = float(p.position)
        if qty == 0:
            continue
        rows.append({
            "symbol": sym,
            "qty": qty,
            "avg_cost": float(p.avgCost or 0.0),
            "mkt_price": None,    # filled below if available
            "mkt_value": None,
            "unreal_pnl": None,
        })
    # Try to enrich with market price + unrealized PnL via portfolio() which
    # carries marketPrice/marketValue/unrealizedPNL on each PortfolioItem.
    portfolio_by_symbol = {p.contract.symbol: p for p in ib.portfolio(account=account_id)}
    for r in rows:
        pi = portfolio_by_symbol.get(r["symbol"])
        if pi is not None:
            r["mkt_price"] = float(pi.marketPrice or 0.0) or None
            r["mkt_value"] = float(pi.marketValue or 0.0) or None
            r["unreal_pnl"] = float(pi.unrealizedPNL or 0.0)
    db.replace_positions_snapshot(con, account_id=account_id, snapshot_at=now, rows=rows)


def _sync_orders_and_fills(ib, con, account_id: str, now: str) -> None:
    trades: list[Trade] = ib.trades()
    for t in trades:
        # ib_insync attaches the account ID on the Order; filter to ours.
        order_account = t.order.account or account_id
        if order_account != account_id:
            continue
        st = _ibkr_status_to_str(t.orderStatus.status)
        placed_at = (
            t.log[0].time.isoformat(timespec="seconds") + "Z"
            if t.log else now
        )
        db.upsert_order(
            con,
            account_id=account_id,
            order_id=int(t.order.orderId),
            perm_id=int(t.order.permId) if t.order.permId else None,
            symbol=t.contract.symbol,
            side=t.order.action,
            qty=float(t.order.totalQuantity),
            order_type=t.order.orderType,
            limit_price=float(t.order.lmtPrice) if t.order.lmtPrice else None,
            stop_price=float(t.order.auxPrice) if t.order.auxPrice else None,
            tif=t.order.tif or "DAY",
            status=st,
            filled_qty=float(t.orderStatus.filled or 0),
            avg_fill_price=float(t.orderStatus.avgFillPrice) if t.orderStatus.avgFillPrice else None,
            placed_at=placed_at,
            last_update_at=now,
            kubera_signal_id=None,
        )
        for f in t.fills:
            db.insert_fill(
                con,
                account_id=account_id,
                exec_id=f.execution.execId,
                order_id=int(t.order.orderId),
                perm_id=int(t.order.permId) if t.order.permId else None,
                symbol=t.contract.symbol,
                side=f.execution.side,
                qty=float(f.execution.shares),
                price=float(f.execution.price),
                fill_time=f.time.isoformat(timespec="seconds") + "Z",
                commission=float(f.commissionReport.commission) if f.commissionReport else None,
                realized_pnl=float(f.commissionReport.realizedPNL) if f.commissionReport else None,
            )


def tick(client: KuberaIB, con) -> None:
    """One sync iteration. Caller commits."""
    ib = client.ib
    sess = client.session
    assert sess is not None
    now = _now_iso()
    db.upsert_account_meta(
        con,
        account_id=sess.account_id, mode=sess.mode, surface=sess.surface,
        host=sess.host, port=sess.port,
        server_version=sess.server_version, now_iso=now,
    )
    _sync_account_summary(ib, con, sess.account_id, now)
    _sync_positions(ib, con, sess.account_id, now)
    _sync_orders_and_fills(ib, con, sess.account_id, now)


def run(interval: float = 30.0) -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    client = KuberaIB(client_id=KUBERA_CLIENT_ID_SYNC)
    while not _stop:
        try:
            if not client.is_connected():
                client.connect()
            con = db.connect()
            try:
                tick(client, con)
                con.commit()
            finally:
                con.close()
        except Exception as e:
            log.warning("tick failed: %s; will retry in %.0fs", e, interval)
            try:
                client.disconnect()
            except Exception:
                pass
        # Sleep in 1-second chunks so SIGINT is responsive.
        slept = 0.0
        while slept < interval and not _stop:
            time.sleep(1.0)
            slept += 1.0
    client.disconnect()
    log.info("sync stopped cleanly")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=30.0, help="seconds between polls")
    args = p.parse_args()
    sys.exit(run(args.interval))
