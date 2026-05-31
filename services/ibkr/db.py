"""Schema + write helpers for ibkr_live.sqlite.

Mirrors the existing paper-trading schema (paper_orders / paper_fills /
paper_positions / paper_equity) so the dashboard can use the same
visualisation code paths against either set of tables.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

DB_PATH = "data/processed/ibkr_live.sqlite"

# Schema is idempotent — every CREATE uses IF NOT EXISTS, every INSERT
# pattern uses ON CONFLICT to make repeats safe. The sync loop runs every
# 30 s; the same row will be seen many times.
SCHEMA = """
CREATE TABLE IF NOT EXISTS ibkr_account_meta (
  account_id     TEXT PRIMARY KEY,
  mode           TEXT NOT NULL,           -- 'paper' | 'live'
  surface        TEXT NOT NULL,           -- 'gateway' | 'tws'
  host           TEXT NOT NULL,
  port           INTEGER NOT NULL,
  first_seen_at  TEXT NOT NULL,
  last_seen_at   TEXT NOT NULL,
  server_version INTEGER
);

CREATE TABLE IF NOT EXISTS ibkr_equity (
  account_id   TEXT NOT NULL,
  snapshot_at  TEXT NOT NULL,
  nav          REAL NOT NULL,
  cash         REAL NOT NULL,
  long_mv      REAL NOT NULL,
  realized_pnl REAL NOT NULL,
  unreal_pnl   REAL NOT NULL,
  currency     TEXT NOT NULL,
  PRIMARY KEY (account_id, snapshot_at)
);

CREATE TABLE IF NOT EXISTS ibkr_positions (
  account_id   TEXT NOT NULL,
  snapshot_at  TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  qty          REAL NOT NULL,
  avg_cost     REAL NOT NULL,
  mkt_price    REAL,
  mkt_value    REAL,
  unreal_pnl   REAL,
  PRIMARY KEY (account_id, snapshot_at, symbol)
);
CREATE INDEX IF NOT EXISTS ix_ibkr_positions_symbol_time
  ON ibkr_positions(symbol, snapshot_at);

CREATE TABLE IF NOT EXISTS ibkr_orders (
  account_id   TEXT NOT NULL,
  order_id     INTEGER NOT NULL,
  perm_id      INTEGER,
  symbol       TEXT NOT NULL,
  side         TEXT NOT NULL,     -- 'BUY' | 'SELL'
  qty          REAL NOT NULL,
  order_type   TEXT NOT NULL,     -- 'MKT' | 'LMT' | 'STP' | 'MOC' ...
  limit_price  REAL,
  stop_price   REAL,
  tif          TEXT NOT NULL,     -- 'DAY' | 'GTC' ...
  status       TEXT NOT NULL,     -- 'PendingSubmit' | 'Submitted' | 'Filled' | 'Cancelled' ...
  filled_qty   REAL DEFAULT 0,
  avg_fill_price REAL,
  placed_at    TEXT NOT NULL,
  last_update_at TEXT NOT NULL,
  kubera_signal_id INTEGER,       -- FK to kubera_signals.id when applicable
  PRIMARY KEY (account_id, order_id)
);
CREATE INDEX IF NOT EXISTS ix_ibkr_orders_status
  ON ibkr_orders(status, placed_at);

CREATE TABLE IF NOT EXISTS ibkr_fills (
  account_id   TEXT NOT NULL,
  exec_id      TEXT PRIMARY KEY,  -- IBKR's globally-unique execution id
  order_id     INTEGER NOT NULL,
  perm_id      INTEGER,
  symbol       TEXT NOT NULL,
  side         TEXT NOT NULL,
  qty          REAL NOT NULL,
  price        REAL NOT NULL,
  fill_time    TEXT NOT NULL,
  commission   REAL,
  realized_pnl REAL
);
CREATE INDEX IF NOT EXISTS ix_ibkr_fills_order
  ON ibkr_fills(account_id, order_id);
CREATE INDEX IF NOT EXISTS ix_ibkr_fills_symbol_time
  ON ibkr_fills(symbol, fill_time);

CREATE TABLE IF NOT EXISTS kubera_signals (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_date     TEXT NOT NULL,        -- date the signal was emitted
  intended_action TEXT NOT NULL,         -- 'OPEN_LONG' | 'CLOSE_LONG'
  symbol          TEXT NOT NULL,
  qty             REAL NOT NULL,
  target_price    REAL,                  -- model's expected fill price (for slippage calc)
  status          TEXT NOT NULL,         -- 'PENDING_APPROVAL' | 'APPROVED' | 'PLACED' | 'FILLED' | 'CANCELLED' | 'REJECTED'
  approved_at     TEXT,
  approved_by     TEXT,
  ibkr_order_id   INTEGER,               -- when status >= PLACED
  notes           TEXT,
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_signals_date_status
  ON kubera_signals(signal_date, status);
"""


def db_path() -> Path:
    p = Path(DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(read_only: bool = False) -> sqlite3.Connection:
    """Open the IBKR-state DB, creating schema if first run."""
    p = db_path()
    if read_only and p.exists():
        uri = "file:" + p.as_posix() + "?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=10.0)
    con = sqlite3.connect(p.as_posix(), timeout=30.0)
    con.executescript(SCHEMA)
    con.commit()
    return con


def upsert_account_meta(con: sqlite3.Connection, *, account_id: str, mode: str,
                         surface: str, host: str, port: int, server_version: int,
                         now_iso: str) -> None:
    con.execute(
        """
        INSERT INTO ibkr_account_meta
          (account_id, mode, surface, host, port, first_seen_at, last_seen_at, server_version)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(account_id) DO UPDATE SET
          mode = excluded.mode,
          surface = excluded.surface,
          host = excluded.host,
          port = excluded.port,
          last_seen_at = excluded.last_seen_at,
          server_version = excluded.server_version
        """,
        (account_id, mode, surface, host, port, now_iso, now_iso, server_version),
    )


def insert_equity_snapshot(con: sqlite3.Connection, *, account_id: str,
                             snapshot_at: str, nav: float, cash: float,
                             long_mv: float, realized: float, unreal: float,
                             currency: str) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO ibkr_equity
          (account_id, snapshot_at, nav, cash, long_mv, realized_pnl, unreal_pnl, currency)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (account_id, snapshot_at, nav, cash, long_mv, realized, unreal, currency),
    )


def replace_positions_snapshot(con: sqlite3.Connection, *, account_id: str,
                                 snapshot_at: str, rows: Iterable[dict]) -> None:
    """Each call writes one snapshot timestamp with all current positions."""
    payload = [
        (account_id, snapshot_at, r["symbol"], r["qty"], r["avg_cost"],
         r.get("mkt_price"), r.get("mkt_value"), r.get("unreal_pnl"))
        for r in rows
    ]
    if payload:
        con.executemany(
            """
            INSERT OR REPLACE INTO ibkr_positions
              (account_id, snapshot_at, symbol, qty, avg_cost, mkt_price, mkt_value, unreal_pnl)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            payload,
        )


def upsert_order(con: sqlite3.Connection, *, account_id: str, order_id: int,
                   perm_id: Optional[int], symbol: str, side: str, qty: float,
                   order_type: str, limit_price: Optional[float],
                   stop_price: Optional[float], tif: str, status: str,
                   filled_qty: float, avg_fill_price: Optional[float],
                   placed_at: str, last_update_at: str,
                   kubera_signal_id: Optional[int]) -> None:
    con.execute(
        """
        INSERT INTO ibkr_orders
          (account_id, order_id, perm_id, symbol, side, qty, order_type, limit_price,
           stop_price, tif, status, filled_qty, avg_fill_price, placed_at, last_update_at,
           kubera_signal_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(account_id, order_id) DO UPDATE SET
          perm_id = COALESCE(excluded.perm_id, ibkr_orders.perm_id),
          status = excluded.status,
          filled_qty = excluded.filled_qty,
          avg_fill_price = excluded.avg_fill_price,
          last_update_at = excluded.last_update_at,
          kubera_signal_id = COALESCE(excluded.kubera_signal_id, ibkr_orders.kubera_signal_id)
        """,
        (account_id, order_id, perm_id, symbol, side, qty, order_type, limit_price,
         stop_price, tif, status, filled_qty, avg_fill_price, placed_at, last_update_at,
         kubera_signal_id),
    )


def insert_fill(con: sqlite3.Connection, *, account_id: str, exec_id: str,
                 order_id: int, perm_id: Optional[int], symbol: str, side: str,
                 qty: float, price: float, fill_time: str,
                 commission: Optional[float], realized_pnl: Optional[float]) -> None:
    con.execute(
        """
        INSERT OR IGNORE INTO ibkr_fills
          (account_id, exec_id, order_id, perm_id, symbol, side, qty, price,
           fill_time, commission, realized_pnl)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (account_id, exec_id, order_id, perm_id, symbol, side, qty, price,
         fill_time, commission, realized_pnl),
    )
