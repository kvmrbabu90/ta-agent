"""Sunday-night dry-run: validate the Alpaca integration end-to-end before
Monday open.

Usage:
    .venv\\Scripts\\python.exe -m scripts.kubera_alpaca_dryrun
    .venv\\Scripts\\python.exe -m scripts.kubera_alpaca_dryrun --mode paper
    .venv\\Scripts\\python.exe -m scripts.kubera_alpaca_dryrun --mode live --i-know-what-im-doing

Runs 5 checks:
  1. Connect — uses paper or live keys based on --mode (default: paper)
  2. Account summary — pulls NAV / cash / buying power / status
  3. Place a small test order — 1 share of SPY at a far-from-market limit
     so it sits in the book
  4. Cancel the test order — verifies the order plumbing both directions
  5. Pull positions + orders — confirms the read side works

Prints PASS / FAIL per step. Exits non-zero if any step failed.

NOTE: in live mode the script REFUSES to place the test order (even at
$0.01) unless --i-know-what-im-doing is passed. Paper mode places +
cancels the test order automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest

from services.alpaca.connection import KuberaAlpaca, Mode


def step(label: str, ok: bool, detail: str = "") -> None:
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {label}" + (f" — {detail}" if detail else ""))


def main(mode: Optional[Mode] = None, allow_live: bool = False) -> int:
    print("=" * 60)
    print(f"Kubera <-> Alpaca dry-run ({'auto' if mode is None else mode.upper()})")
    print("=" * 60)

    wrapper = KuberaAlpaca(mode=mode)

    # ---------- 1. Connect ----------
    try:
        sess = wrapper.connect()
        step(
            "1. Connect", True,
            f"{sess.mode.upper()} account={sess.account_number} status={sess.status} {sess.currency}",
        )
    except Exception as e:
        step("1. Connect", False, str(e))
        print()
        print("Make sure the right ALPACA_*_KEY / ALPACA_*_SECRET env vars are set,")
        print("and that ALPACA_MODE matches the keys you provided.")
        return 1

    if sess.mode == "live" and not allow_live:
        print()
        print(f"  Connected to LIVE account {sess.account_number}.")
        print("  Refusing to place a test order without --i-know-what-im-doing.")
        print("  Run with that flag if you want to exercise live order plumbing")
        print("  (the order is far OTM so it will NOT fill, but it WILL appear in your account log).")
        wrapper.disconnect()
        return 0

    client = wrapper.client

    # ---------- 2. Account summary ----------
    try:
        acct = client.get_account()
        nav = float(acct.portfolio_value or 0.0)
        cash = float(acct.cash or 0.0)
        bp = float(acct.buying_power or 0.0)
        step("2. Account summary", True,
             f"NAV=${nav:,.2f} cash=${cash:,.2f} BP=${bp:,.2f} blocked={bool(acct.trading_blocked)}")
    except Exception as e:
        step("2. Account summary", False, str(e))
        wrapper.disconnect()
        return 1

    # ---------- 3. Place a far-OTM limit test order ----------
    test_qty = 1
    test_limit = 1.00  # absurdly far below market — guarantees no fill
    req = LimitOrderRequest(
        symbol="SPY",
        qty=test_qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=test_limit,
    )
    placed_order = None
    try:
        placed_order = client.submit_order(order_data=req)
        # Wait briefly for status to settle.
        for _ in range(20):
            time.sleep(0.25)
            o = client.get_order_by_id(placed_order.id)
            status = str(o.status.value if hasattr(o.status, "value") else o.status)
            if status in ("new", "accepted", "pending_new", "partially_filled"):
                placed_order = o
                break
            placed_order = o
        status = str(placed_order.status.value if hasattr(placed_order.status, "value") else placed_order.status)
        step("3. Place test order", True,
             f"order_id={placed_order.id} status={status} client_id={placed_order.client_order_id}")
    except Exception as e:
        step("3. Place test order", False, str(e))
        wrapper.disconnect()
        return 1

    # ---------- 4. Cancel the test order ----------
    try:
        client.cancel_order_by_id(placed_order.id)
        final_status = "unknown"
        for _ in range(20):
            time.sleep(0.25)
            o = client.get_order_by_id(placed_order.id)
            final_status = str(o.status.value if hasattr(o.status, "value") else o.status)
            if final_status in ("canceled", "cancelled", "filled", "expired", "rejected"):
                break
        ok = final_status in ("canceled", "cancelled")
        step("4. Cancel test order", ok, f"final status={final_status}")
        if not ok:
            wrapper.disconnect()
            return 1
    except Exception as e:
        step("4. Cancel test order", False, str(e))
        wrapper.disconnect()
        return 1

    # ---------- 5. Pull positions + orders (verifies read plumbing) ----------
    try:
        positions = client.get_all_positions()
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50, direction="desc")
        orders = client.get_orders(filter=req)
        step("5. Read positions/orders", True,
             f"positions={len(positions)} recent_orders={len(orders)}")
    except Exception as e:
        step("5. Read positions/orders", False, str(e))
        wrapper.disconnect()
        return 1

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED — Alpaca plumbing ready for Monday.")
    print("=" * 60)
    wrapper.disconnect()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["paper", "live"], default=None,
                   help="override ALPACA_MODE env var (default: paper)")
    p.add_argument("--i-know-what-im-doing", dest="allow_live", action="store_true",
                   help="allow the dry-run to place a test order against a LIVE account "
                        "(order is far OTM so it cannot fill, but will appear in Alpaca logs)")
    args = p.parse_args()
    sys.exit(main(mode=args.mode, allow_live=args.allow_live))
