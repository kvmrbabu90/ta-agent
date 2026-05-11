"""Paper-trading backtest engine.

Strategy (minimal viable, can be tuned via StrategyConfig):
  - Universe: SP500
  - Every trading day at 8am CT (i.e. before the day's open):
      * Look at the model's predictions for today (logged from daily_predict)
      * Top N by predicted_return_5d  -> long, equal-weight $position_size each
      * Bottom N by predicted_return_5d -> short, equal-weight $position_size each
        (only if predicted < -short_threshold; otherwise skip shorts that day)
      * Close any existing position not in today's new top/bottom set
      * Open new positions at today's OPEN price (8am-CT execution proxy)
  - 5pm CT snapshot = mark-to-market with today's CLOSE price
  - Next day at 8am CT: snapshot uses prior close (no overnight bars), then
    re-trade as above.

This is BACKTEST-ONLY (no live broker calls). The job/scheduler can be
wired in later to run the same `step_one_day()` function daily.

Schema:
    paper_runs       — one row per backtest run (config + start/end + final_equity)
    paper_trades     — every fill: (run_id, trade_date, symbol, side, qty, fill_price, ...)
    paper_positions  — open positions snapshot per (run_id, trade_date, symbol)
    paper_equity     — equity curve: (run_id, trade_date, snapshot_kind, equity, cash,
                                       long_mv, short_mv, realized_pnl, unrealized_pnl)
        snapshot_kind in ('open_8am_ct', 'close_5pm_ct')
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from packages.common.config import settings
from packages.common.logging import log
from packages.ingestion.storage import _connect as duck_connect


_PAPER_DDL = """
CREATE TABLE IF NOT EXISTS paper_runs (
    run_id          TEXT PRIMARY KEY,
    universe        TEXT NOT NULL,
    starting_cash   REAL NOT NULL,
    position_size   REAL NOT NULL,
    n_long          INTEGER NOT NULL,
    n_short         INTEGER NOT NULL,
    short_threshold REAL NOT NULL,
    started_at      TEXT NOT NULL,
    first_trade_date TEXT,
    last_trade_date  TEXT,
    final_equity    REAL,
    final_realized_pnl REAL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS paper_trades (
    run_id          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,   -- 'long_open', 'short_open', 'long_close', 'short_close'
    qty             REAL NOT NULL,   -- positive shares; sign comes from side
    fill_price      REAL NOT NULL,
    cash_delta      REAL NOT NULL,
    realized_pnl    REAL,
    PRIMARY KEY (run_id, trade_date, symbol, side)
);

CREATE TABLE IF NOT EXISTS paper_positions (
    run_id          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,   -- 'long' or 'short'
    qty             REAL NOT NULL,
    entry_price     REAL NOT NULL,
    entry_date      TEXT NOT NULL,
    PRIMARY KEY (run_id, trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS paper_equity (
    run_id          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    snapshot_kind   TEXT NOT NULL,   -- 'open_8am_ct' | 'close_5pm_ct'
    equity          REAL NOT NULL,
    cash            REAL NOT NULL,
    long_mv         REAL NOT NULL,
    short_mv        REAL NOT NULL,
    realized_pnl    REAL NOT NULL,
    unrealized_pnl  REAL NOT NULL,
    PRIMARY KEY (run_id, trade_date, snapshot_kind)
);

CREATE INDEX IF NOT EXISTS paper_equity_run_date
    ON paper_equity (run_id, trade_date);
CREATE INDEX IF NOT EXISTS paper_trades_run_date
    ON paper_trades (run_id, trade_date);
"""


def _paper_db_path() -> str:
    return str(Path(settings.predictions_sqlite_path).parent / "paper.sqlite")


def init_paper_db(path: str | None = None) -> str:
    """Create the paper-trading schema. Returns the SQLite path used."""
    p = path or _paper_db_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    try:
        conn.executescript(_PAPER_DDL)
        conn.commit()
    finally:
        conn.close()
    return p


@dataclass
class StrategyConfig:
    universe: str = "SP500"
    starting_cash: float = 1000.0
    n_long: int = 5
    n_short: int = 5
    position_size: float = 100.0  # dollar size per position
    short_threshold: float = 0.001  # min |predicted_return| for shorts (0.1%)
    short_enabled: bool = True
    # `start_date` defaults to a sensible lookback: covers all logged predictions
    start_date: date | None = None
    end_date: date | None = None
    run_id: str = "default"
    notes: str = ""


DEFAULT_CONFIG = StrategyConfig()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_predictions(
    universe: str, start: date | None, end: date | None
) -> pd.DataFrame:
    conn = sqlite3.connect(settings.predictions_sqlite_path)
    try:
        sql = (
            "SELECT as_of, symbol, predicted_return, top_quintile_proba, "
            "bottom_quintile_proba FROM predictions_log WHERE universe = ?"
        )
        params: list = [universe]
        if start is not None:
            sql += " AND as_of >= ?"
            params.append(start.isoformat())
        if end is not None:
            sql += " AND as_of <= ?"
            params.append(end.isoformat())
        sql += " ORDER BY as_of, predicted_return"
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    if df.empty:
        return df
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    return df


def _load_ohlcv_for_symbols(
    duck: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "bar_date", "open", "close"])
    rows = duck.execute(
        """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY symbol, bar_date
                ORDER BY ingested_at DESC
            ) AS rn
            FROM ohlcv_daily
            WHERE symbol = ANY(?) AND bar_date BETWEEN ? AND ?
        )
        SELECT symbol, bar_date, open, close
        FROM ranked WHERE rn = 1
        ORDER BY symbol, bar_date
        """,
        [symbols, start, end],
    ).df()
    if not rows.empty:
        rows["bar_date"] = pd.to_datetime(rows["bar_date"]).dt.date
    return rows


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def _persist_run_metadata(
    conn: sqlite3.Connection, cfg: StrategyConfig, started_at: str
) -> None:
    conn.execute(
        """
        INSERT INTO paper_runs (
            run_id, universe, starting_cash, position_size,
            n_long, n_short, short_threshold, started_at, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            universe = excluded.universe,
            starting_cash = excluded.starting_cash,
            position_size = excluded.position_size,
            n_long = excluded.n_long,
            n_short = excluded.n_short,
            short_threshold = excluded.short_threshold,
            started_at = excluded.started_at,
            notes = excluded.notes
        """,
        (
            cfg.run_id, cfg.universe, cfg.starting_cash, cfg.position_size,
            cfg.n_long, cfg.n_short, cfg.short_threshold, started_at, cfg.notes,
        ),
    )


def _clear_existing_run(conn: sqlite3.Connection, run_id: str) -> None:
    for tbl in ("paper_trades", "paper_positions", "paper_equity"):
        conn.execute(f"DELETE FROM {tbl} WHERE run_id = ?", (run_id,))


def backtest(cfg: StrategyConfig = DEFAULT_CONFIG) -> dict:
    """Run a paper-trade backtest from cfg.start_date to cfg.end_date.

    Returns a small summary dict; full results land in the paper.sqlite tables.
    """
    paper_db = init_paper_db()
    log.info(f"paper backtest: run_id={cfg.run_id} db={paper_db}")

    predictions = _load_predictions(cfg.universe, cfg.start_date, cfg.end_date)
    if predictions.empty:
        log.warning("paper backtest: no predictions in window; nothing to do")
        return {"run_id": cfg.run_id, "n_trade_days": 0, "final_equity": cfg.starting_cash}

    trade_dates: list[date] = sorted(predictions["as_of"].unique().tolist())
    pred_start = trade_dates[0]
    pred_end = trade_dates[-1]
    log.info(f"paper backtest: {len(trade_dates)} trade days {pred_start} -> {pred_end}")

    # Pull OHLCV for ALL symbols referenced + SPY benchmark; cache once.
    all_syms = sorted(set(predictions["symbol"].unique().tolist()))
    duck = duck_connect()
    try:
        ohlcv = _load_ohlcv_for_symbols(
            duck, all_syms, pred_start, pred_end + pd.Timedelta(days=10).to_pytimedelta()
        )
    finally:
        duck.close()
    if ohlcv.empty:
        log.error("paper backtest: no OHLCV for any prediction symbol; aborting")
        return {"run_id": cfg.run_id, "n_trade_days": 0, "final_equity": cfg.starting_cash}

    # (symbol, date) -> (open, close)
    bars = ohlcv.set_index(["symbol", "bar_date"]).to_dict("index")

    paper_conn = sqlite3.connect(paper_db)
    paper_conn.execute("BEGIN")
    try:
        started_at = pd.Timestamp.utcnow().isoformat()
        _persist_run_metadata(paper_conn, cfg, started_at)
        _clear_existing_run(paper_conn, cfg.run_id)

        cash = float(cfg.starting_cash)
        realized_pnl = 0.0
        # symbol -> {'side': 'long'/'short', 'qty': float, 'entry_price': float, 'entry_date': date}
        positions: dict[str, dict] = {}

        for d in trade_dates:
            day_preds = predictions[predictions["as_of"] == d]
            if day_preds.empty:
                continue

            # ---- 8am CT snapshot (BEFORE trades): mark using PRIOR close,
            #   but for simplicity here we mark using the open on this day
            #   as the proxy for 8am state. (8am is pre-market; the most
            #   recent observable price is yesterday's close, but we don't
            #   have that in the position dict's entry context. We use today's
            #   open as the trade-execution price AND as the 8am mark.)
            #   For positions held overnight, mark using today's open.
            unreal_pnl_8am, long_mv_8am, short_mv_8am = _mark_positions(positions, bars, d, "open")
            equity_8am = cash + long_mv_8am - short_mv_8am
            paper_conn.execute(
                "INSERT OR REPLACE INTO paper_equity (run_id, trade_date, snapshot_kind, "
                "equity, cash, long_mv, short_mv, realized_pnl, unrealized_pnl) "
                "VALUES (?, ?, 'open_8am_ct', ?, ?, ?, ?, ?, ?)",
                (cfg.run_id, d.isoformat(), equity_8am, cash, long_mv_8am, short_mv_8am,
                 realized_pnl, unreal_pnl_8am),
            )

            # ---- Pick targets for this day
            sorted_today = day_preds.sort_values("predicted_return", ascending=False)
            top_longs = sorted_today.head(cfg.n_long)
            bottom_for_shorts = sorted_today.tail(cfg.n_short)
            if cfg.short_enabled:
                bottom_for_shorts = bottom_for_shorts[
                    bottom_for_shorts["predicted_return"] < -cfg.short_threshold
                ]
            else:
                bottom_for_shorts = bottom_for_shorts.iloc[0:0]

            target_longs = set(top_longs["symbol"].tolist())
            target_shorts = set(bottom_for_shorts["symbol"].tolist())
            target_all = target_longs | target_shorts

            # ---- Close stale positions (anything not in today's target set)
            for sym in list(positions.keys()):
                if sym not in target_all:
                    cash, realized_pnl = _close_position(
                        paper_conn, cfg.run_id, d, sym, positions, bars, cash, realized_pnl
                    )

            # ---- Open new positions
            for sym in target_longs:
                if sym in positions and positions[sym]["side"] == "long":
                    continue  # already long, no change
                if sym in positions and positions[sym]["side"] == "short":
                    # was short, now want long: close then re-open
                    cash, realized_pnl = _close_position(
                        paper_conn, cfg.run_id, d, sym, positions, bars, cash, realized_pnl
                    )
                cash, opened = _open_position(
                    paper_conn, cfg.run_id, d, sym, "long", cfg.position_size, bars, cash
                )
                if opened:
                    positions[sym] = opened

            for sym in target_shorts:
                if sym in positions and positions[sym]["side"] == "short":
                    continue
                if sym in positions and positions[sym]["side"] == "long":
                    cash, realized_pnl = _close_position(
                        paper_conn, cfg.run_id, d, sym, positions, bars, cash, realized_pnl
                    )
                cash, opened = _open_position(
                    paper_conn, cfg.run_id, d, sym, "short", cfg.position_size, bars, cash
                )
                if opened:
                    positions[sym] = opened

            # ---- Persist current positions snapshot for the day
            for sym, p in positions.items():
                paper_conn.execute(
                    "INSERT OR REPLACE INTO paper_positions "
                    "(run_id, trade_date, symbol, side, qty, entry_price, entry_date) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (cfg.run_id, d.isoformat(), sym, p["side"], p["qty"],
                     p["entry_price"], p["entry_date"].isoformat()),
                )

            # ---- 5pm CT snapshot: mark using today's CLOSE
            unreal_pnl_5pm, long_mv_5pm, short_mv_5pm = _mark_positions(positions, bars, d, "close")
            equity_5pm = cash + long_mv_5pm - short_mv_5pm
            paper_conn.execute(
                "INSERT OR REPLACE INTO paper_equity (run_id, trade_date, snapshot_kind, "
                "equity, cash, long_mv, short_mv, realized_pnl, unrealized_pnl) "
                "VALUES (?, ?, 'close_5pm_ct', ?, ?, ?, ?, ?, ?)",
                (cfg.run_id, d.isoformat(), equity_5pm, cash, long_mv_5pm, short_mv_5pm,
                 realized_pnl, unreal_pnl_5pm),
            )

        final_unreal, final_long_mv, final_short_mv = _mark_positions(
            positions, bars, trade_dates[-1], "close"
        )
        final_equity = cash + final_long_mv - final_short_mv
        paper_conn.execute(
            "UPDATE paper_runs SET first_trade_date=?, last_trade_date=?, final_equity=?, "
            "final_realized_pnl=? WHERE run_id=?",
            (trade_dates[0].isoformat(), trade_dates[-1].isoformat(),
             final_equity, realized_pnl, cfg.run_id),
        )
        paper_conn.commit()

        return {
            "run_id": cfg.run_id,
            "n_trade_days": len(trade_dates),
            "first_trade_date": trade_dates[0].isoformat(),
            "last_trade_date": trade_dates[-1].isoformat(),
            "starting_cash": cfg.starting_cash,
            "final_equity": final_equity,
            "final_realized_pnl": realized_pnl,
        }
    except Exception:
        paper_conn.rollback()
        raise
    finally:
        paper_conn.close()


def _mark_positions(
    positions: dict, bars: dict, on_date: date, price_field: str
) -> tuple[float, float, float]:
    """Return (unrealized_pnl, long_mv, short_mv) using bars[(sym, date)][price_field]."""
    unreal = long_mv = short_mv = 0.0
    for sym, p in positions.items():
        bar = bars.get((sym, on_date))
        if not bar or pd.isna(bar.get(price_field)):
            # No bar today (holiday, missing data); skip.
            continue
        price = float(bar[price_field])
        qty = p["qty"]
        entry = p["entry_price"]
        if p["side"] == "long":
            long_mv += qty * price
            unreal += qty * (price - entry)
        else:
            short_mv += qty * price
            unreal += qty * (entry - price)
    return unreal, long_mv, short_mv


def _open_position(
    conn: sqlite3.Connection, run_id: str, on_date: date, symbol: str,
    side: str, dollar_size: float, bars: dict, cash: float,
) -> tuple[float, dict | None]:
    bar = bars.get((symbol, on_date))
    if not bar or pd.isna(bar.get("open")):
        return cash, None
    price = float(bar["open"])
    if price <= 0:
        return cash, None
    qty = dollar_size / price
    if side == "long":
        cash_delta = -(qty * price)
    else:
        # Short open: receive proceeds. With equity = cash + long_mv - short_mv,
        # crediting cash here keeps total equity unchanged at trade open
        # (the +cash exactly offsets the new -short_mv leg).
        cash_delta = +(qty * price)
    cash_after = cash + cash_delta
    conn.execute(
        "INSERT OR REPLACE INTO paper_trades (run_id, trade_date, symbol, side, qty, "
        "fill_price, cash_delta, realized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, on_date.isoformat(), symbol, f"{side}_open", qty, price, cash_delta, 0.0),
    )
    return cash_after, {
        "side": side, "qty": qty, "entry_price": price, "entry_date": on_date,
    }


def _close_position(
    conn: sqlite3.Connection, run_id: str, on_date: date, symbol: str,
    positions: dict, bars: dict, cash: float, realized_pnl: float,
) -> tuple[float, float]:
    p = positions.pop(symbol, None)
    if p is None:
        return cash, realized_pnl
    bar = bars.get((symbol, on_date))
    # If today has no bar, close at entry (no PnL) — defensive
    price = float(bar["open"]) if bar and not pd.isna(bar.get("open")) else p["entry_price"]
    qty = p["qty"]
    entry = p["entry_price"]
    if p["side"] == "long":
        gross = qty * price
        cash_delta = +gross  # recover the long cash
        trade_pnl = qty * (price - entry)
    else:
        # Short close: buy back at current price → cash_delta = -(qty*price).
        # Combined with the +entry*qty we credited at open, net cash change
        # is qty*(entry - close) = realized PnL.
        cash_delta = -(qty * price)
        trade_pnl = qty * (entry - price)
    cash_after = cash + cash_delta
    realized_after = realized_pnl + trade_pnl
    side_name = f"{p['side']}_close"
    conn.execute(
        "INSERT OR REPLACE INTO paper_trades (run_id, trade_date, symbol, side, qty, "
        "fill_price, cash_delta, realized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, on_date.isoformat(), symbol, side_name, qty, price, cash_delta, trade_pnl),
    )
    return cash_after, realized_after
