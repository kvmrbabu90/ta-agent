"""Kubera engine daemon — auto-trades the strategy on Alpaca with WF-backtest fidelity.

This is the LIVE counterpart to packages/paper_trading/engine.py. The two
share the same scoring rule, slice sizing, vol-scaling, and stop-loss
geometry — only the execution surface differs (real Alpaca orders here
vs simulated fills there).

Daily routines:

  daily_open_run()    fires once when market opens (~9:30 ET):
    1. CLOSE_LONG every lot opened HOLDING_DAYS (=5) trading days ago.
    2. OPEN_LONG today's top-N picks. Per-stock dollars sized as
         (NAV / HOLDING_DAYS) × normalised_weight
       where weight_i ∝ combined_score_i / ATR(14)_i within the slice.
       This matches StrategyConfig(conviction_weighted=True,
       vol_scaling='inverse').
    3. Wait briefly for entry fills, then submit a GTC SELL STOP for
       each new lot at  min(low[-3:]) × (1 - 0.003)  = the 3-day rolling
       low with a 30bps buffer. Tagged client_order_id 'kubera-stop-{sym}-{date}'
       so we can find and replace it on subsequent close-mark runs.

  daily_close_mark()  fires once after market closes (~16:00 ET):
    For each currently-open lot, recompute stop_level from the new
    rolling-low (3 most recent daily bars including today). Cancel the
    old GTC stop, place a new one at the updated level. If today's close
    has already breached the new stop, submit a MarketOrder SELL with
    TIF=OPG so it fills at tomorrow's open print (closest live analog
    to the simulator's "exit at stop_level today" behaviour).

Process lifecycle: launched detached by services/alpaca/engine_state.py.
Survives API restarts, terminal close, Claude session end. Does NOT
survive system reboot — click Start Kubera again afterward.

Daemon control:  services.alpaca.engine_state.{start,stop,status}.
Dashboard:       POST /live-alpaca/engine/start, POST /engine/stop,
                 GET  /live-alpaca/engine/status.
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

# Strategy constants — mirror packages.paper_trading.StrategyConfig defaults exactly.
TOP_N_PICKS = 5             # n_long
HOLDING_DAYS = 5            # holding_days
STEADY_LOTS = TOP_N_PICKS * HOLDING_DAYS  # 25 lots steady-state book
ATR_LOOKBACK = 14           # atr_lookback_days (Wilder's standard)
SUPPORT_LOOKBACK = 3        # support_lookback_days (Connors-style short MR)
STOP_BUFFER_PCT = 0.003     # stop_buffer_pct (0.3% below rolling low)
TICK_INTERVAL_SEC = 60      # check clock every minute

# Predictions DB (written by the daily inference pipeline).
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
# Lazy market-data client (created once, reused)
# ----------------------------------------------------------------------

_mdc = None


def _market_data_client():
    """Singleton StockHistoricalDataClient, lazy-initialised from settings."""
    global _mdc
    if _mdc is None:
        from alpaca.data.historical import StockHistoricalDataClient
        from packages.common.config import settings
        mode = (settings.alpaca_mode or "paper").strip().lower()
        if mode == "paper":
            key, sec = settings.alpaca_paper_key, settings.alpaca_paper_secret
        else:
            key, sec = settings.alpaca_live_key, settings.alpaca_live_secret
        _mdc = StockHistoricalDataClient(api_key=key, secret_key=sec)
    return _mdc


# ----------------------------------------------------------------------
# Predictions + scoring
# ----------------------------------------------------------------------

def _top_picks(as_of: str, n: int = TOP_N_PICKS) -> list[tuple[str, float]]:
    """Top-N symbols by Kubera *combined score* for the given as_of date.

    score = predicted_return * (1 + (top_quintile_proba - bottom_quintile_proba))

    Falls back to the most recent as_of <= today so weekend / holiday
    runs (and Monday) get Friday's picks.
    """
    p = Path(PREDICTIONS_DB)
    if not p.exists():
        log.warning("predictions DB %s missing; no picks today", p)
        return []
    con = sqlite3.connect("file:" + p.as_posix() + "?mode=ro", uri=True, timeout=10.0)
    try:
        latest = con.execute(
            "SELECT MAX(as_of) FROM predictions_log WHERE as_of <= ?",
            (as_of,),
        ).fetchone()[0]
        if not latest:
            log.warning("no predictions <= %s; engine cannot pick", as_of)
            return []
        rows = con.execute(
            """
            SELECT symbol,
                   predicted_return * (1.0 + (COALESCE(top_quintile_proba, 0.0)
                                              - COALESCE(bottom_quintile_proba, 0.0))) AS combined_score
            FROM predictions_log
            WHERE as_of = ?
            ORDER BY combined_score DESC, symbol ASC
            LIMIT ?
            """,
            (latest, n),
        ).fetchall()
        log.info("picks for %s (as_of=%s, conviction-weighted): %s",
                 as_of, latest, [(s, round(r, 5)) for s, r in rows])
        return [(str(s), float(r)) for s, r in rows]
    finally:
        con.close()


# ----------------------------------------------------------------------
# Market data — daily bars for ATR + rolling-low stop
# ----------------------------------------------------------------------

def _get_daily_bars(symbols: list[str], days_back: int) -> dict:
    """Return {symbol: [(date, open, high, low, close), ...]} sorted ascending.

    Reads from the local DuckDB (data/processed/market.duckdb::ohlcv_daily),
    not from Alpaca's market-data API. Two reasons:
      1. Alpaca's free paper tier rejects SIP bar queries ("subscription does
         not permit querying recent SIP data") — IEX-only fallback covers
         only ~3% of NYSE volume and is too sparse for clean ATR.
      2. The simulator uses this exact same table, so reading from it
         guarantees ATR(14) and 3-day rolling-low match the backtest
         numerically.

    The daily pipeline writes today's bar to ohlcv_daily after market
    close (typically by 17:00 CT). The open run at ~9:30 ET works off
    bars through yesterday — that's correct: ATR + rolling-low for the
    entry stop are SUPPOSED to be computed off pre-open data.
    """
    import duckdb
    from packages.common.config import settings

    if not symbols:
        return {}
    out: dict[str, list[tuple]] = {s: [] for s in symbols}
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    try:
        conn = duckdb.connect(settings.duckdb_path, read_only=True)
    except Exception as e:
        log.warning("duckdb open failed: %s", e)
        return out
    try:
        # De-dup on (symbol, bar_date) — keep most-recent ingestion if there
        # are multiple rows. Same pattern as packages.paper_trading.engine.
        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY symbol, bar_date
                    ORDER BY ingested_at DESC
                ) AS rn
                FROM ohlcv_daily
                WHERE symbol = ANY(?) AND bar_date BETWEEN ? AND ?
            )
            SELECT symbol, bar_date, open, high, low, close
            FROM ranked WHERE rn = 1
            ORDER BY symbol, bar_date
            """,
            [symbols, start, end],
        ).fetchall()
    finally:
        conn.close()
    for sym, d, o, h, l, c in rows:
        out.setdefault(sym, []).append((d, float(o), float(h), float(l), float(c)))
    for sym in out:
        out[sym].sort(key=lambda r: r[0])
    return out


def _atr_wilder(bars: list[tuple], lookback: int = ATR_LOOKBACK) -> Optional[float]:
    """Wilder's ATR over `lookback` bars. Bars are (date, o, h, l, c) ascending.

    TR_t = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR  = Wilder's running average (EMA with alpha = 1/N) over TR series.

    Returns the most recent ATR value, or None if insufficient bars.
    """
    if len(bars) < lookback + 1:
        return None
    trs: list[float] = []
    prev_close = bars[0][4]
    for _, _, h, l, c in bars[1:]:
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if len(trs) < lookback:
        return None
    # Seed with simple average of the first `lookback` TRs, then Wilder-smooth.
    atr = sum(trs[:lookback]) / lookback
    for tr in trs[lookback:]:
        atr = (atr * (lookback - 1) + tr) / lookback
    return atr


def _rolling_low_stop(bars: list[tuple],
                       lookback: int = SUPPORT_LOOKBACK,
                       buffer_pct: float = STOP_BUFFER_PCT) -> Optional[float]:
    """stop = min(low[-lookback:]) × (1 - buffer_pct). Bars ascending."""
    if len(bars) < lookback:
        return None
    lows = [b[3] for b in bars[-lookback:]]
    support = min(lows)
    return round(support * (1.0 - buffer_pct), 2)


# ----------------------------------------------------------------------
# Trading-day arithmetic (Alpaca calendar)
# ----------------------------------------------------------------------

def _trading_day_n_back(client, today_iso: str, n: int) -> Optional[str]:
    from alpaca.trading.requests import GetCalendarRequest
    try:
        end = dt.date.fromisoformat(today_iso)
        start = end - dt.timedelta(days=n * 3 + 14)
        cal = client.get_calendar(GetCalendarRequest(start=start, end=end))
        sessions = [c.date.isoformat() if hasattr(c.date, "isoformat") else str(c.date)[:10]
                    for c in cal]
        sessions = [s for s in sessions if s < today_iso]
        if len(sessions) < n:
            return None
        return sessions[-n]
    except Exception as e:
        log.warning("calendar lookup failed: %s", e)
        return None


# ----------------------------------------------------------------------
# Engine state DB
# ----------------------------------------------------------------------

ENGINE_DB = "data/processed/alpaca_live.sqlite"

ENGINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS alpaca_engine_state (
  id              INTEGER PRIMARY KEY CHECK (id = 1),
  status          TEXT NOT NULL,
  pid             INTEGER,
  sync_pid        INTEGER,
  started_at      TEXT,
  last_run_at     TEXT,
  last_run_date   TEXT,
  last_run_status TEXT,
  last_close_mark_date TEXT,
  last_error      TEXT,
  heartbeat_at    TEXT,
  stopped_at      TEXT
);
INSERT OR IGNORE INTO alpaca_engine_state (id, status) VALUES (1, 'stopped');
"""


def _engine_db() -> sqlite3.Connection:
    con = sqlite3.connect(ENGINE_DB, timeout=30.0)
    con.executescript(ENGINE_SCHEMA)
    # Tolerate older schema (no last_close_mark_date column) on upgrade.
    try:
        con.execute("ALTER TABLE alpaca_engine_state ADD COLUMN last_close_mark_date TEXT")
        con.commit()
    except sqlite3.OperationalError:
        pass
    con.commit()
    return con


def _heartbeat(con, *, last_run_date: Optional[str] = None,
                 last_run_status: Optional[str] = None,
                 last_close_mark_date: Optional[str] = None,
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
    if last_close_mark_date is not None:
        set_clauses.append("last_close_mark_date = ?")
        args.append(last_close_mark_date)
    if last_error is not None:
        set_clauses.append("last_error = ?")
        args.append(last_error)
    con.execute("UPDATE alpaca_engine_state SET " + ", ".join(set_clauses) + " WHERE id = 1", args)
    con.commit()


# ----------------------------------------------------------------------
# Stop-order helpers — find / cancel / place GTC SELL stops tagged per-lot
# ----------------------------------------------------------------------

def _stop_client_id(symbol: str, signal_date: str) -> str:
    return f"kubera-stop-{symbol}-{signal_date}"


def _find_existing_stop(client, symbol: str, signal_date: str) -> Optional[str]:
    """Look up an open SELL stop tagged for this lot. Returns Alpaca order_id."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200)
        target_cid = _stop_client_id(symbol, signal_date)
        for o in client.get_orders(filter=req):
            if str(o.symbol) == symbol and str(o.client_order_id) == target_cid:
                return str(o.id)
    except Exception as e:
        log.warning("stop lookup failed for %s: %s", symbol, e)
    return None


def _place_stop(client, *, symbol: str, qty: float, stop_price: float,
                  signal_date: str, account_number: str) -> Optional[str]:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import StopOrderRequest
    req = StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=stop_price,
        client_order_id=_stop_client_id(symbol, signal_date),
    )
    try:
        order = client.submit_order(order_data=req)
        log.info("STOP placed: %s qty=%g stop=$%.2f order_id=%s",
                 symbol, qty, stop_price, str(order.id)[:8])
        return str(order.id)
    except Exception as e:
        log.warning("stop placement failed for %s: %s", symbol, e)
        return None


# ----------------------------------------------------------------------
# Open-of-day run
# ----------------------------------------------------------------------

def daily_open_run(wrapper) -> str:
    """Open new slice + close 5-day-old slice + place initial stops."""
    from services.alpaca import db as alp_db
    from services.alpaca import orders as orders_module

    client = wrapper.client
    sess = wrapper.session
    today_iso = dt.date.today().isoformat()

    # ------- 1. CLOSE_LONG every lot opened HOLDING_DAYS trading days ago -------
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

    # Before CLOSE_LONG submits — cancel the GTC stops on those lots,
    # otherwise the market SELL will trip wash-trade detection.
    for sym, _qty in closes_to_submit:
        sid = _find_existing_stop(client, sym, close_signal_date)
        if sid:
            try:
                client.cancel_order_by_id(sid)
                log.info("cancelled stop %s for %s before CLOSE_LONG", sid[:8], sym)
            except Exception as e:
                log.warning("stop cancel failed for %s: %s", sym, e)

    # ------- 2. OPEN_LONG today's top-N picks (conviction × inverse-vol sized) -------
    picks = _top_picks(today_iso, n=TOP_N_PICKS)
    if not picks and not closes_to_submit:
        log.warning("no picks and nothing to close — skipping run")
        return "no_picks"

    nav = float(client.get_account().portfolio_value or 0.0)
    slice_budget = (nav / HOLDING_DAYS) if nav > 0 else 0.0
    log.info("NAV=%.2f slice_budget=%.2f", nav, slice_budget)

    # Pull bars: ATR(14) needs 15 bars, rolling-low needs 3; ask 60 cal days
    # ≈ 42 sessions to handle weekends/holidays comfortably.
    symbols = [s for s, _ in picks]
    bars = _get_daily_bars(symbols, days_back=60)

    raw_weights: dict[str, float] = {}
    atrs: dict[str, float] = {}
    last_prices: dict[str, float] = {}
    stop_prices: dict[str, float] = {}
    for sym, score in picks:
        b = bars.get(sym, [])
        if len(b) < ATR_LOOKBACK + 1:
            log.warning("skip %s: only %d bars (need %d for ATR)",
                        sym, len(b), ATR_LOOKBACK + 1)
            continue
        atr = _atr_wilder(b, ATR_LOOKBACK)
        if atr is None or atr <= 0:
            log.warning("skip %s: ATR=%s", sym, atr)
            continue
        last_price = b[-1][4]  # last bar's close as proxy until market opens
        if last_price <= 0:
            log.warning("skip %s: bad last_price=%s", sym, last_price)
            continue
        # Inverse-vol weight: score / ATR. Negative scores get negative
        # weight which we floor at 0 (top picks have positive score by
        # construction, but be robust).
        w = max(0.0, score) / atr
        raw_weights[sym] = w
        atrs[sym] = atr
        last_prices[sym] = last_price
        stop_prices[sym] = _rolling_low_stop(b, SUPPORT_LOOKBACK, STOP_BUFFER_PCT) or 0.0

    total_w = sum(raw_weights.values())
    if total_w <= 0:
        log.warning("all weights zero/negative; nothing to open this slice")
        opens_to_submit: list[tuple[str, int, float, float]] = []
    else:
        opens_to_submit = []
        for sym, w in raw_weights.items():
            allocation = slice_budget * (w / total_w)
            qty = max(1, int(allocation / last_prices[sym]))
            opens_to_submit.append((sym, qty, last_prices[sym], stop_prices[sym]))
        log.info("open plan: %s",
                 [(s, q, round(p, 2), round(stp, 2)) for s, q, p, stp in opens_to_submit])

    # ------- 3. Stage CLOSE_LONG signals -------
    pending_close_ids: list[int] = []
    for sym, qty in closes_to_submit:
        sid = orders_module.stage_signal(
            signal_date=today_iso, action="CLOSE_LONG",
            symbol=sym, qty=qty, target_price=None,
            notes=f"auto-close: opened {close_signal_date}",
        )
        pending_close_ids.append(sid)

    # ------- 4. Stage OPEN_LONG signals -------
    pending_open_ids: list[int] = []
    for sym, qty, price, _stop in opens_to_submit:
        sid = orders_module.stage_signal(
            signal_date=today_iso, action="OPEN_LONG",
            symbol=sym, qty=qty, target_price=price,
            notes=f"auto-open conv×inv-vol: ATR={atrs[sym]:.3f} stop=${_stop:.2f}",
        )
        pending_open_ids.append(sid)

    # ------- 5. Submit. CLOSE first so we free up wash-trade lockouts. -------
    # cfg.stop_pct=0.0 disables the OTO bracket in orders.approve_and_submit
    # — the engine places its own rolling-low SELL stops below, separately.
    from services.alpaca.orders import RiskConfig
    no_bracket = RiskConfig(stop_pct=0.0)

    if pending_close_ids:
        out = orders_module.approve_and_submit(
            pending_close_ids, approved_by="kubera-engine", cfg=no_bracket,
        )
        log.info("close submit: %s", out)

    if pending_open_ids:
        out = orders_module.approve_and_submit(
            pending_open_ids, approved_by="kubera-engine", cfg=no_bracket,
        )
        log.info("open submit: %s", out)

    # ------- 6. Wait briefly for entry fills, then place initial stops -------
    # Market orders on a paper account usually fill within ~1 second; give
    # a few seconds of grace before placing the protective stops.
    if pending_open_ids:
        time.sleep(5)
        for sym, qty, _price, stop_price in opens_to_submit:
            if stop_price > 0:
                _place_stop(client, symbol=sym, qty=qty, stop_price=stop_price,
                            signal_date=today_iso, account_number=sess.account_number)

    return "ok"


# ----------------------------------------------------------------------
# After-close mark — recompute stops for every open lot
# ----------------------------------------------------------------------

def daily_close_mark(wrapper) -> str:
    """Recompute stop_level for every open lot from fresh end-of-day data,
    cancel the old GTC stop, place a new one at the updated level.
    """
    from services.alpaca import db as alp_db
    client = wrapper.client
    today_iso = dt.date.today().isoformat()

    # Find all lots currently open: OPEN_LONG signals with status=PLACED/FILLED
    # and (signal_date) more recent than HOLDING_DAYS trading days ago.
    horizon = _trading_day_n_back(client, today_iso, HOLDING_DAYS)
    con = alp_db.connect(read_only=True)
    try:
        params = []
        sql = (
            "SELECT signal_date, symbol, qty FROM kubera_alpaca_signals "
            "WHERE intended_action='OPEN_LONG' AND status IN ('PLACED','FILLED')"
        )
        if horizon:
            sql += " AND signal_date > ?"
            params.append(horizon)
        rows = con.execute(sql, params).fetchall()
        open_lots = [(r[0], r[1], float(r[2])) for r in rows]
    finally:
        con.close()
    if not open_lots:
        log.info("close-mark: no open lots; nothing to update")
        return "ok"

    # Batch-fetch bars for the unique symbols.
    symbols = sorted({sym for _, sym, _ in open_lots})
    bars = _get_daily_bars(symbols, days_back=30)
    log.info("close-mark: %d open lots across %d symbols", len(open_lots), len(symbols))

    for signal_date, sym, qty in open_lots:
        b = bars.get(sym, [])
        new_stop = _rolling_low_stop(b, SUPPORT_LOOKBACK, STOP_BUFFER_PCT)
        if new_stop is None or new_stop <= 0:
            log.warning("close-mark skip %s: no rolling-low stop", sym)
            continue
        # Cancel any existing stop for this lot.
        old_id = _find_existing_stop(client, sym, signal_date)
        if old_id:
            try:
                client.cancel_order_by_id(old_id)
            except Exception as e:
                log.warning("close-mark cancel failed %s: %s", sym, e)
        # Brief beat for the cancel to settle before re-submit.
        time.sleep(0.5)
        _place_stop(client, symbol=sym, qty=qty, stop_price=new_stop,
                    signal_date=signal_date,
                    account_number=wrapper.session.account_number)
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

    con = _engine_db()
    row = con.execute(
        "SELECT last_run_date, last_close_mark_date FROM alpaca_engine_state WHERE id = 1"
    ).fetchone()
    last_open_date: Optional[str] = row[0] if row and row[0] else None
    last_close_mark_date: Optional[str] = row[1] if row and row[1] else None

    while not _stop:
        try:
            clock = wrapper.client.get_clock()
            today_iso = dt.date.today().isoformat()

            # Open run: when market just opened today and we haven't run yet.
            if clock.is_open and last_open_date != today_iso:
                log.info("market open — running Kubera open rotation for %s", today_iso)
                try:
                    status = daily_open_run(wrapper)
                    last_open_date = today_iso
                    _heartbeat(con, last_run_date=today_iso, last_run_status=status)
                except Exception as e:
                    log.exception("daily_open_run failed")
                    _heartbeat(con, last_run_status="error", last_error=str(e))

            # Close-mark: market just closed today, we already ran open today,
            # and we haven't done close-mark yet today.
            if (not clock.is_open
                and last_open_date == today_iso
                and last_close_mark_date != today_iso):
                log.info("after close — running stop-level refresh for %s", today_iso)
                try:
                    daily_close_mark(wrapper)
                    last_close_mark_date = today_iso
                    _heartbeat(con, last_close_mark_date=today_iso)
                except Exception as e:
                    log.exception("daily_close_mark failed")
                    _heartbeat(con, last_error=str(e))

            _heartbeat(con)
        except Exception as e:
            log.warning("loop tick failed: %s", e)
            _heartbeat(con, last_error=str(e))

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
