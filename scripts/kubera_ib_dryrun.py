"""Sunday-night dry-run: validate the IBKR integration end-to-end before
Monday open.

Usage (after IB Gateway is running and you've logged in):

    .venv\\Scripts\\python.exe -m scripts.kubera_ib_dryrun

Runs 5 checks:
  1. Connect — finds the running Gateway/TWS port, identifies paper vs live
  2. Account summary — pulls NAV/cash/currency
  3. Place a small test order — 1 share of SPY at a far-from-market limit
     so it sits in the book
  4. Cancel the test order — verifies the order plumbing in both directions
  5. Pull positions + fills tables — confirms the read side works

Prints PASS / FAIL per step. Exits non-zero if any step failed.

NOTE: this is a SAFETY check. In live mode, the script REFUSES to place
the test order (even at $0.01) unless --i-know-what-im-doing is passed.
Paper mode places + cancels the test order automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from ib_insync import LimitOrder, Stock

from services.ibkr.connection import KuberaIB, KUBERA_CLIENT_ID_CLI


def step(label: str, ok: bool, detail: str = "") -> None:
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {label}" + (f" — {detail}" if detail else ""))


def main(allow_live: bool = False) -> int:
    print("=" * 60)
    print("Kubera <-> IBKR dry-run")
    print("=" * 60)

    client = KuberaIB(client_id=KUBERA_CLIENT_ID_CLI)

    # ---------- 1. Connect ----------
    try:
        sess = client.connect()
        step("1. Connect", True,
             f"{sess.mode.upper()} {sess.surface} on {sess.host}:{sess.port}, account {sess.account_id}, server v{sess.server_version}")
    except Exception as e:
        step("1. Connect", False, str(e))
        print()
        print("Make sure IB Gateway (or TWS) is running and logged in.")
        return 1

    if sess.mode == "live" and not allow_live:
        print()
        print(f"  Connected to LIVE account {sess.account_id}.")
        print("  Refusing to place a test order without --i-know-what-im-doing.")
        print("  Run with that flag if you really want to exercise live order plumbing")
        print("  (the order is far OTM so it will NOT fill, but it WILL appear in your account log).")
        client.disconnect()
        return 0

    ib = client.ib

    # ---------- 2. Account summary ----------
    try:
        rows = ib.accountSummary(sess.account_id)
        summary = {r.tag: r for r in rows}
        nav = float(summary["NetLiquidation"].value) if "NetLiquidation" in summary else None
        cash = float(summary["TotalCashValue"].value) if "TotalCashValue" in summary else None
        ccy = summary["NetLiquidation"].currency if "NetLiquidation" in summary else "?"
        step("2. Account summary", True, f"NAV={nav!r} {ccy}, cash={cash!r}")
    except Exception as e:
        step("2. Account summary", False, str(e))
        client.disconnect()
        return 1

    # ---------- 3. Place a far-OTM test order ----------
    contract = Stock("SPY", "SMART", "USD")
    test_qty = 1
    test_limit = 1.00  # absurdly far below market — guarantees no fill
    test_order = LimitOrder("BUY", test_qty, test_limit)
    test_order.account = sess.account_id
    test_order.tif = "DAY"
    trade = None
    try:
        ib.qualifyContracts(contract)
        trade = ib.placeOrder(contract, test_order)
        # Wait briefly for the order to acknowledge.
        for _ in range(20):
            ib.sleep(0.25)
            if trade.orderStatus.status in ("PreSubmitted", "Submitted", "PendingSubmit"):
                break
        step("3. Place test order", True,
             f"order_id={trade.order.orderId}, status={trade.orderStatus.status}, perm_id={trade.order.permId}")
    except Exception as e:
        step("3. Place test order", False, str(e))
        client.disconnect()
        return 1

    # ---------- 4. Cancel the test order ----------
    try:
        ib.cancelOrder(trade.order)
        for _ in range(20):
            ib.sleep(0.25)
            if trade.orderStatus.status == "Cancelled":
                break
        ok = trade.orderStatus.status == "Cancelled"
        step("4. Cancel test order", ok, f"final status={trade.orderStatus.status}")
        if not ok:
            client.disconnect()
            return 1
    except Exception as e:
        step("4. Cancel test order", False, str(e))
        client.disconnect()
        return 1

    # ---------- 5. Pull positions + fills (verifies read plumbing) ----------
    try:
        positions = ib.positions(account=sess.account_id)
        fills = ib.fills()
        trades = ib.trades()
        step("5. Read positions/fills/trades", True,
             f"positions={len(positions)}, fills={len(fills)}, trades={len(trades)}")
    except Exception as e:
        step("5. Read positions/fills/trades", False, str(e))
        client.disconnect()
        return 1

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED — IBKR plumbing ready for Monday.")
    print("=" * 60)
    client.disconnect()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--i-know-what-im-doing", dest="allow_live", action="store_true",
                   help="allow the dry-run to place a test order against a LIVE account "
                        "(order is far OTM so it cannot fill, but will appear in IBKR logs)")
    args = p.parse_args()
    sys.exit(main(allow_live=args.allow_live))
