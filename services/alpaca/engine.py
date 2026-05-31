"""Kubera engine daemon — auto-trades the strategy on Alpaca.

Runs as a long-lived detached process. Every minute, checks the Alpaca
clock. When the market opens (or first run of the trading day after open),
executes one Kubera rotation:

  1. CLOSE_LONG every lot that was opened exactly 5 trading days ago
  2. OPEN_LONG the latest top-5 predictions, sized at NAV / 25 each
     (steady-state 25-lot book)

All orders go through services.alpaca.orders.approve_and_submit so the
audit trail in kubera_alpaca_signals is preserved (approved_by =
'kubera-engine'). Entries are submitted as OTO brackets so the
protective stop attaches atomically.

Daemon control: services.alpaca.engine_state.{start,stop,status}.
Dashboard: POST /live-alpaca/engine/start, POST /live-alpaca/engine/stop,
GET /live-alpaca/engine/status.

This process is launched DETACHED. It survives:
  - The API server restarting
  - The terminal that started it closing
  - The Claude session ending

It does NOT survive system reboot — user must click "Start Kubera" again
after a restart.
"""
from __future__ import annotations

import datetime as dt
import logging
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

# Kubera strategy constants — mirror the WF backtest exactly.
TOP_N_PICKS = 5             # daily top-N picks
HOLDING_DAYS = 5            # trading days per lot
STEADY_LOTS = TOP_N_PICKS * HOLDING_DAYS  # 25 lots at steady state
TICK_INTERVAL_SEC = 60      # check clock every minute

# Predictions DB (written by the daily inference pipeline, separate from WF results).
PREDICTIONS_DB = "data/processed/predictions.sqlite"

log = logging.getLogger("kubera.alpaca.engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_stop = False


def _handle_signal(_signum, _frame):
    global _stop
    _stop = True
    log.info("stop signal received; engine will exit after current tick")


# ----------------------------------------------------------------------
# Predictions + market data helpers
# ----------------------------------------------------------------------

def _top_picks(as_of: str, n: int = TOP_N_PICKS) -> list[tuple[str, float]]:
    """Top-N symbols by predicted_return for the given as_of date.

    Returns [(symbol, predicted_return), ...] sorted descending.
    Falls back to the most recent as_of in the DB if `as_of` itself has
    no rows (e.g. weekend, holiday — we want Friday's picks for Monday).
    """
    p = Path(PREDICTIONS_DB)
    if not p.exists():
        log.warning("predictions DB %s missing; no picks today", p)
        return []
    con = sqlite3.connect("file:" + p.as_posix() + "?mode=ro", uri=True, timeout=10.0)
    try:
        # Use the latest as_of <= today (so on Monday we get Friday's picks).
        latest = con.execute(
            "SELECT MAX(as_of) FROM predictions_log WHERE as_of <= ?",
            (as_of,),
        ).fetchone()[0]
        if not latest:
            log.warning("no predictions <= %s; engine cannot pick", as_of)
            return []
        rows = con.execute(
            """
            SELECT symbol, predicted_return
            FROM predictions_log
            WHERE as_of = ?
            ORDER BY predicted_return DESC, symbol ASC
            LIMIT ?
            """,
            (latest, n),
        ).fetchall()
        log.info("picks for %s (as_of=%s): %s",
                 as_of, latest, [(s, round(r, 5)) for s, r in rows])
        return [(str(s), float(r)) for s, r in rows]
    finally:
        con.close()


def _latest_price(client, symbol: str) -> Optional[float]:
    """Use Alpaca's last-trade endpoint via the market-data feed when available;
    fall back to the most-recent close on the trading-API position quote.

    Implementation note: alpaca-py exposes market data via a separate client
    (StockHistoricalDataClient). To keep this engine dependency-light, we
    use a tiny REST call here through the trading client's underlying session.
    """
    try:
        # The trading-API account also exposes `last_quote` via the underlying
        # REST session. Cleanest path: just submit a tiny notional check by
        # asking for the asset and using its `current_price` if open, otherwise
        # use a placeholder. We avoid hitting market-data infra to keep this
        # daemon self-contained.
        #
        # In practice the daily_run waits until market is open (Alpaca clock
        # says is_open=True), so latest_trade is always available. We use the
        # alpaca-py market-data client only if user has data subscription.
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from packages.common.config import settings
        mode = (settings.alpaca_mode or "paper").strip().lower()
        if mode == "paper":
            key, sec = settings.alpaca_paper_key, settings.alpaca_paper_secret
        else:
            key, sec = settings.alpaca_live_key, settings.alpaca_live_secret
        mdc = StockHistoricalDataClient(api_key=key, secret_key=sec)
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = mdc.get_stock_latest_quote(req)
        q = quotes.get(symbol)
        if q is None:
            return None
        # Use midpoint of bid/ask
        bid = float(q.bid_price or 0.0)
        ask = float(q.ask_price or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if ask > 0:
            return ask
        if bid > 0:
            return bid
        return None
    except Exception as e:
        log.warning("price lookup for %s failed: %s", symbol, e)
        return None


# ----------------------------------------------------------------------
# Trading-day arithmetic
# ----------------------------------------------------------------------

def _trading_day_n_back(client, today_iso: str, n: int) -> Optional[str]:
    """Return the date (YYYY-MM-DD) of `n` trading days before today.

    Uses Alpaca's calendar so it respects exchange holidays.
    Returns None if calendar unavailable.
    """
    from alpaca.trading.requests import GetCalendarRequest
    try:
        end = dt.date.fromisoformat(today_iso)
        start = end - dt.timedelta(days=n * 3 + 14)  # safe window
        cal = client.get_calendar(GetCalendarRequest(start=start, end=end))
        sessions = [c.date.isoformat() if hasattr(c.date, "isoformat") else str(c.date)[:10]
                    for c in cal]
        # Drop today if it's the last session; we want n back from today.
        sessions = [s for s in sessions if s < today_iso]
        if len(sessions) < n:
            return None
        return sessions[-n]
    except Exception as e:
        log.warning("calendar lookup failed: %s", e)
        return None


# ----------------------------------------------------------------------
# Engine state DB helpers — record last-run + heartbeat
# ----------------------------------------------------------------------

ENGINE_DB = "data/processed/alpaca_live.sqlite"

ENGINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS alpaca_engine_state (
  id              INTEGER PRIMARY KEY CHECK (id = 1),
  status          TEXT NOT NULL,       -- 'stopped' | 'running' | 'error'
  pid             INTEGER,
  started_at      TEXT,
  last_run_at     TEXT,
  last_run_date   TEXT,                -- ISO date of the most recent daily_run
  last_run_status TEXT,                -- 'ok' | 'no_picks' | 'error'
  last_error      TEXT,
  heartbeat_at    TEXT,
  stopped_at      TEXT
);
INSERT OR IGNORE INTO alpaca_engine_state (id, status) VALUES (1, 'stopped');
"""


def _engine_db() -> sqlite3.Connection:
    con = sqlite3.connect(ENGINE_DB, timeout=30.0)
    con.executescript(ENGINE_SCHEMA)
    con.commit()
    return con


def _heartbeat(con, *, last_run_date: Optional[str] = None,
                 last_run_status: Optional[str] = None,
                 last_error: Optional[str] = None) -> None:
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    set_clauses = ["heartbeat_at = ?"]
    args: list = [now]
    if last_run_date is not None:
        set_clauses += ["last_run_at = ?", "last_run_date = ?"]
        args += [now, last_run_date]
    if last_run_status is not None:
        set_clauses.append("last_run_status = ?")
        args.append(last_run_status)
    if last_error is not None:
        set_clauses.append("last_error = ?")
        args.append(last_error)
    con.execute("UPDATE alpaca_engine_state SET " + ", ".join(set_clauses) + " WHERE id = 1", args)
    con.commit()


# ----------------------------------------------------------------------
# Daily-run logic
# ----------------------------------------------------------------------

def daily_run(wrapper) -> str:
    """One Kubera rotation. Returns a status string ('ok' / 'no_picks' / etc)."""
    from services.alpaca import db as alp_db
    from services.alpaca import orders as orders_module

    client = wrapper.client
    sess = wrapper.session
    assert sess is not None

    today_iso = dt.date.today().isoformat()

    # ------- 1. Determine what to CLOSE -------
    close_signal_date = _trading_day_n_back(client, today_iso, HOLDING_DAYS)
    closes_to_submit: list[tuple[str, float]] = []
    if close_signal_date:
        con = alp_db.connect(read_only=True)
        try:
            rows = con.execute(
                """
                SELECT symbol, qty FROM kubera_alpaca_signals
                WHERE signal_date = ? AND intended_action = 'OPEN_LONG'
                  AND status IN ('PLACED', 'FILLED')
                """,
                (close_signal_date,),
            ).fetchall()
            closes_to_submit = [(r[0], float(r[1])) for r in rows]
        finally:
            con.close()
    log.info("close candidates (opened %s): %s", close_signal_date, closes_to_submit)

    # ------- 2. Determine what to OPEN -------
    picks = _top_picks(today_iso, n=TOP_N_PICKS)
    if not picks and not closes_to_submit:
        log.warning("no picks and nothing to close — skipping run")
        return "no_picks"

    nav = float(client.get_account().portfolio_value or 0.0)
    per_lot = nav / STEADY_LOTS if nav > 0 else 0.0
    log.info("NAV=%.2f per_lot=%.2f", nav, per_lot)

    opens_to_submit: list[tuple[str, int, float]] = []
    for sym, _score in picks:
        price = _latest_price(client, sym)
        if price is None or price <= 0:
            log.warning("skip %s: no quote", sym)
            continue
        qty = max(1, int(per_lot / price))
        opens_to_submit.append((sym, qty, price))
    log.info("open candidates: %s", opens_to_submit)

    # ------- 3. Stage + auto-submit CLOSE_LONG -------
    pending_close_ids: list[int] = []
    for sym, qty in closes_to_submit:
        sid = orders_module.stage_signal(
            signal_date=today_iso, action="CLOSE_LONG",
            symbol=sym, qty=qty, target_price=None,
            notes=f"auto-close: opened {close_signal_date}",
        )
        pending_close_ids.append(sid)

    # ------- 4. Stage + auto-submit OPEN_LONG -------
    pending_open_ids: list[int] = []
    for sym, qty, price in opens_to_submit:
        sid = orders_module.stage_signal(
            signal_date=today_iso, action="OPEN_LONG",
            symbol=sym, qty=qty, target_price=price,
            notes=f"auto-open: top-{TOP_N_PICKS} as_of <= {today_iso}",
        )
        pending_open_ids.append(sid)

    # ------- 5. Submit all (CLOSE first, then OPEN) -------
    if pending_close_ids:
        out = orders_module.approve_and_submit(
            pending_close_ids, approved_by="kubera-engine",
        )
        log.info("close submit: %s", out)
    if pending_open_ids:
        out = orders_module.approve_and_submit(
            pending_open_ids, approved_by="kubera-engine",
        )
        log.info("open submit: %s", out)

    return "ok"


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def run() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    from services.alpaca.connection import KuberaAlpaca
    wrapper = KuberaAlpaca()
    try:
        wrapper.connect()
    except Exception as e:
        log.error("engine cannot connect to Alpaca: %s", e)
        con = _engine_db(); _heartbeat(con, last_error=str(e)); con.close()
        return 1

    log.info("Kubera engine started (mode=%s account=%s)",
             wrapper.session.mode, wrapper.session.account_number)

    last_run_date: Optional[str] = None
    con = _engine_db()
    row = con.execute(
        "SELECT last_run_date FROM alpaca_engine_state WHERE id = 1"
    ).fetchone()
    if row and row[0]:
        last_run_date = row[0]

    while not _stop:
        try:
            clock = wrapper.client.get_clock()
            today_iso = dt.date.today().isoformat()
            if clock.is_open and last_run_date != today_iso:
                log.info("market open — running daily Kubera rotation for %s", today_iso)
                try:
                    status = daily_run(wrapper)
                    last_run_date = today_iso
                    _heartbeat(con, last_run_date=today_iso, last_run_status=status)
                except Exception as e:
                    log.exception("daily_run failed")
                    _heartbeat(con, last_run_status="error", last_error=str(e))
            else:
                _heartbeat(con)
        except Exception as e:
            log.warning("loop tick failed: %s", e)
            _heartbeat(con, last_error=str(e))
        # Sleep in 5-second chunks so SIGTERM is responsive.
        slept = 0.0
        while slept < TICK_INTERVAL_SEC and not _stop:
            time.sleep(5.0)
            slept += 5.0

    con.execute(
        "UPDATE alpaca_engine_state SET status = 'stopped', stopped_at = ? WHERE id = 1",
        (dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",),
    )
    con.commit()
    con.close()
    wrapper.disconnect()
    log.info("Kubera engine stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(run())
