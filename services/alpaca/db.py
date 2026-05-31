"""Schema + write helpers for alpaca_live.sqlite.

Mirrors services/ibkr/db.py so the dashboard can reuse the same patterns
against either broker. Table names use the `alpaca_` prefix to keep
multi-broker setups unambiguous.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

DB_PATH = "data/processed/alpaca_live.sqlite"

# Schema is idempotent. The sync loop runs every 30 s; the same row will
# be seen many times.
SCHEMA = """
CREATE TABLE IF NOT EXISTS alpaca_account_meta (
  account_number TEXT PRIMARY KEY,        -- 'PAxxxxxxx' (paper) or 'xxxxxxx' (live)
  account_id     TEXT NOT NULL,           -- Alpaca's internal UUID
  mode           TEXT NOT NULL,           -- 'paper' | 'live'
  status         TEXT NOT NULL,
  currency       TEXT NOT NULL,
  first_seen_at  TEXT NOT NULL,
  last_seen_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alpaca_equity (
  account_number TEXT NOT NULL,
  snapshot_at    TEXT NOT NULL,
  nav            REAL NOT NULL,           -- portfolio_value
  cash           REAL NOT NULL,
  long_mv        REAL NOT NULL,           -- long_market_value
  equity         REAL NOT NULL,
  buying_power   REAL NOT NULL,
  currency       TEXT NOT NULL,
  PRIMARY KEY (account_number, snapshot_at)
);

CREATE TABLE IF NOT EXISTS alpaca_positions (
  account_number TEXT NOT NULL,
  snapshot_at    TEXT NOT NULL,
  symbol         TEXT NOT NULL,
  qty            REAL NOT NULL,
  avg_entry_price REAL NOT NULL,
  mkt_price      REAL,
  mkt_value      REAL,
  unreal_pnl     REAL,
  unreal_pnl_pct REAL,
  side           TEXT NOT NULL,            -- 'long' | 'short'
  PRIMARY KEY (account_number, snapshot_at, symbol)
);
CREATE INDEX IF NOT EXISTS ix_alpaca_positions_symbol_time
  ON alpaca_positions(symbol, snapshot_at);

CREATE TABLE IF NOT EXISTS alpaca_orders (
  account_number TEXT NOT NULL,
  order_id       TEXT NOT NULL,            -- Alpaca's UUID
  client_order_id TEXT,
  symbol         TEXT NOT NULL,
  side           TEXT NOT NULL,            -- 'buy' | 'sell'
  qty            REAL NOT NULL,
  order_type     TEXT NOT NULL,            -- 'market' | 'limit' | 'stop' | ...
  limit_price    REAL,
  stop_price     REAL,
  tif            TEXT NOT NULL,            -- 'day' | 'gtc' | 'cls' | 'opg' | ...
  status         TEXT NOT NULL,            -- 'new' | 'filled' | 'canceled' | ...
  filled_qty     REAL DEFAULT 0,
  filled_avg_price REAL,
  submitted_at   TEXT NOT NULL,
  last_update_at TEXT NOT NULL,
  kubera_signal_id INTEGER,
  PRIMARY KEY (account_number, order_id)
);
CREATE INDEX IF NOT EXISTS ix_alpaca_orders_status
  ON alpaca_orders(status, submitted_at);

CREATE TABLE IF NOT EXISTS alpaca_fills (
  account_number TEXT NOT NULL,
  exec_id        TEXT PRIMARY KEY,         -- Alpaca trade_updates execution_id
  order_id       TEXT NOT NULL,
  symbol         TEXT NOT NULL,
  side           TEXT NOT NULL,
  qty            REAL NOT NULL,
  price          REAL NOT NULL,
  fill_time      TEXT NOT NULL,
  -- Alpaca generally returns zero commission for stocks today; kept for parity
  -- with the IBKR schema and any future fee surfacing (regulatory, options).
  commission     REAL,
  realized_pnl   REAL
);
CREATE INDEX IF NOT EXISTS ix_alpaca_fills_order
  ON alpaca_fills(account_number, order_id);
CREATE INDEX IF NOT EXISTS ix_alpaca_fills_symbol_time
  ON alpaca_fills(symbol, fill_time);

CREATE TABLE IF NOT EXISTS kubera_alpaca_signals (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_date     TEXT NOT NULL,
  intended_action TEXT NOT NULL,            -- 'OPEN_LONG' | 'CLOSE_LONG'
  symbol          TEXT NOT NULL,
  qty             REAL NOT NULL,
  target_price    REAL,
  status          TEXT NOT NULL,            -- 'PENDING_APPROVAL' | 'APPROVED' | 'PLACED' | 'FILLED' | 'CANCELLED' | 'REJECTED'
  approved_at     TEXT,
  approved_by     TEXT,
  alpaca_order_id TEXT,
  notes           TEXT,
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_alpaca_signals_date_status
  ON kubera_alpaca_signals(signal_date, status);
"""


def db_path() -> Path:
    p = Path(DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(read_only: bool = False) -> sqlite3.Connection:
    """Open the Alpaca-state DB, creating schema if first run."""
    p = db_path()
    if read_only and p.exists():
        uri = "file:" + p.as_posix() + "?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=10.0)
    con = sqlite3.connect(p.as_posix(), timeout=30.0)
    con.executescript(SCHEMA)
    con.commit()
    return con


def upsert_account_meta(con: sqlite3.Connection, *, account_number: str,
                          account_id: str, mode: str, status: str,
                          currency: str, now_iso: str) -> None:
    con.execute(
        """
        INSERT INTO alpaca_account_meta
          (account_number, account_id, mode, status, currency, first_seen_at, last_seen_at)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(account_number) DO UPDATE SET
          account_id = excluded.account_id,
          mode = excluded.mode,
          status = excluded.status,
          currency = excluded.currency,
          last_seen_at = excluded.last_seen_at
        """,
        (account_number, account_id, mode, status, currency, now_iso, now_iso),
    )


def insert_equity_snapshot(con: sqlite3.Connection, *, account_number: str,
                             snapshot_at: str, nav: float, cash: float,
                             long_mv: float, equity: float, buying_power: float,
                             currency: str) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO alpaca_equity
          (account_number, snapshot_at, nav, cash, long_mv, equity, buying_power, currency)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (account_number, snapshot_at, nav, cash, long_mv, equity, buying_power, currency),
    )


def replace_positions_snapshot(con: sqlite3.Connection, *, account_number: str,
                                 snapshot_at: str, rows: Iterable[dict]) -> None:
    payload = [
        (account_number, snapshot_at, r["symbol"], r["qty"], r["avg_entry_price"],
         r.get("mkt_price"), r.get("mkt_value"), r.get("unreal_pnl"),
         r.get("unreal_pnl_pct"), r.get("side", "long"))
        for r in rows
    ]
    if payload:
        con.executemany(
            """
            INSERT OR REPLACE INTO alpaca_positions
              (account_number, snapshot_at, symbol, qty, avg_entry_price, mkt_price,
               mkt_value, unreal_pnl, unreal_pnl_pct, side)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            payload,
        )


def upsert_order(con: sqlite3.Connection, *, account_number: str, order_id: str,
                   client_order_id: Optional[str], symbol: str, side: str,
                   qty: float, order_type: str, limit_price: Optional[float],
                   stop_price: Optional[float], tif: str, status: str,
                   filled_qty: float, filled_avg_price: Optional[float],
                   submitted_at: str, last_update_at: str,
                   kubera_signal_id: Optional[int]) -> None:
    con.execute(
        """
        INSERT INTO alpaca_orders
          (account_number, order_id, client_order_id, symbol, side, qty, order_type,
           limit_price, stop_price, tif, status, filled_qty, filled_avg_price,
           submitted_at, last_update_at, kubera_signal_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(account_number, order_id) DO UPDATE SET
          client_order_id = COALESCE(excluded.client_order_id, alpaca_orders.client_order_id),
          status = excluded.status,
          filled_qty = excluded.filled_qty,
          filled_avg_price = excluded.filled_avg_price,
          last_update_at = excluded.last_update_at,
          kubera_signal_id = COALESCE(excluded.kubera_signal_id, alpaca_orders.kubera_signal_id)
        """,
        (account_number, order_id, client_order_id, symbol, side, qty, order_type,
         limit_price, stop_price, tif, status, filled_qty, filled_avg_price,
         submitted_at, last_update_at, kubera_signal_id),
    )


def insert_fill(con: sqlite3.Connection, *, account_number: str, exec_id: str,
                  order_id: str, symbol: str, side: str, qty: float,
                  price: float, fill_time: str,
                  commission: Optional[float], realized_pnl: Optional[float]) -> None:
    con.execute(
        """
        INSERT OR IGNORE INTO alpaca_fills
          (account_number, exec_id, order_id, symbol, side, qty, price,
           fill_time, commission, realized_pnl)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (account_number, exec_id, order_id, symbol, side, qty, price,
         fill_time, commission, realized_pnl),
    )
