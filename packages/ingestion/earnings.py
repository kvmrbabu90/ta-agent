"""Earnings-calendar ingestion.

Stores quarterly earnings announcements (date, fiscal period end, EPS estimate,
EPS actual, surprise) in a separate ``earnings_calendar`` table.

Currently uses yfinance (Ticker.earnings_dates) — the only free source that
returns BOTH the announcement date AND the EPS surprise. Coverage is ~25
quarters per ticker (~6 years), which fills the recent half of our 2014-2026
panel cleanly. Pre-2020 rows will have NaN earnings features — that's fine,
the model masks NaN at training.

A Finnhub fallback is wired in. Finnhub's /calendar/earnings endpoint returns
10+ years of clean announcement dates, but it's premium-tier on Finnhub. The
free tier's /stock/earnings only returns the most recent 4 quarters. If a
paid Finnhub plan is added later, swap the adapter via FINNHUB_API_KEY without
schema migration — same table, same shape.

Schema:
    earnings_calendar (
        symbol           VARCHAR   NOT NULL,
        announce_date    DATE      NOT NULL,  -- the actual reporting day (US/Eastern)
        period_end       DATE,                 -- fiscal quarter end ('period' in Finnhub)
        eps_estimate     DOUBLE,
        eps_actual       DOUBLE,
        surprise_pct     DOUBLE,               -- (actual - estimate) / |estimate| * 100
        source           VARCHAR   NOT NULL,
        ingested_at      TIMESTAMP NOT NULL,
        PRIMARY KEY (symbol, announce_date)
    )
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any

import duckdb
import pandas as pd
import yfinance as yf

from packages.common.logging import log
from packages.ingestion.adapters.yfinance_adapter import to_yfinance_symbol
from packages.ingestion.storage import _connect

_EARNINGS_DDL = """
CREATE TABLE IF NOT EXISTS earnings_calendar (
    symbol         VARCHAR   NOT NULL,
    announce_date  DATE      NOT NULL,
    period_end     DATE,
    eps_estimate   DOUBLE,
    eps_actual     DOUBLE,
    surprise_pct   DOUBLE,
    source         VARCHAR   NOT NULL,
    ingested_at    TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, announce_date)
);
CREATE INDEX IF NOT EXISTS earnings_symbol_date
    ON earnings_calendar (symbol, announce_date);
"""

_SOURCE_YF = "yfinance"
_PER_SYMBOL_SLEEP_SEC: float = 0.5


@contextmanager
def _earnings_conn(path: str | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = _connect(path)
    try:
        conn.execute(_EARNINGS_DDL)
        yield conn
    finally:
        conn.close()


def _ts_to_date(value: Any) -> date | None:
    """Coerce a yfinance timestamp (timezone-aware datetime) to a US/Eastern date.

    yfinance returns earnings_dates indexed by tz-aware Timestamp at the
    announcement time. We collapse to the date of the announcement (in the
    timezone the timestamp came in with — typically America/New_York for
    US-listed names). Bare datetimes fall through as their .date().
    """
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(ts):
        return None
    return ts.date()


def _yf_fetch_earnings(symbol: str, universe: str = "SP500") -> pd.DataFrame:
    """Fetch one symbol's earnings history from yfinance.

    Returns a frame with columns matching earnings_calendar (minus source/ingested_at).
    Empty frame if yfinance returns nothing (delisted, never reported, scrape miss).
    """
    yf_sym = to_yfinance_symbol(symbol, universe)
    try:
        df = yf.Ticker(yf_sym).earnings_dates
    except Exception as exc:  # noqa: BLE001
        log.warning(f"yfinance earnings fetch failed for {symbol}: {exc!r}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    out_rows: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        announce = _ts_to_date(ts)
        if announce is None:
            continue
        eps_est = row.get("EPS Estimate")
        eps_act = row.get("Reported EPS")
        surprise = row.get("Surprise(%)")
        out_rows.append(
            {
                "symbol": symbol,
                "announce_date": announce,
                "period_end": None,  # yfinance doesn't give the fiscal period end here
                "eps_estimate": float(eps_est) if pd.notna(eps_est) else None,
                "eps_actual": float(eps_act) if pd.notna(eps_act) else None,
                "surprise_pct": float(surprise) if pd.notna(surprise) else None,
            }
        )
    return pd.DataFrame(out_rows)


def upsert_earnings(
    df: pd.DataFrame, conn: duckdb.DuckDBPyConnection | None = None
) -> int:
    """Idempotent upsert into earnings_calendar."""
    if df.empty:
        return 0

    own = conn is None
    if own:
        conn = _connect()
        conn.execute(_EARNINGS_DDL)
    try:
        now_utc = datetime.now(UTC)
        df_to_write = df.copy()
        if "source" not in df_to_write.columns:
            df_to_write["source"] = _SOURCE_YF
        if "ingested_at" not in df_to_write.columns:
            df_to_write["ingested_at"] = now_utc

        conn.register("earnings_in", df_to_write)
        try:
            conn.execute(
                """
                INSERT INTO earnings_calendar (
                    symbol, announce_date, period_end,
                    eps_estimate, eps_actual, surprise_pct,
                    source, ingested_at
                )
                SELECT symbol, announce_date, period_end,
                       eps_estimate, eps_actual, surprise_pct,
                       source, ingested_at
                FROM earnings_in
                ON CONFLICT (symbol, announce_date) DO UPDATE SET
                    period_end = EXCLUDED.period_end,
                    eps_estimate = EXCLUDED.eps_estimate,
                    eps_actual = EXCLUDED.eps_actual,
                    surprise_pct = EXCLUDED.surprise_pct,
                    source = EXCLUDED.source,
                    ingested_at = EXCLUDED.ingested_at
                """
            )
        finally:
            conn.unregister("earnings_in")
        return len(df_to_write)
    finally:
        if own:
            conn.close()


def backfill_earnings(
    universe: str = "SP500",
    *,
    symbols: list[str] | None = None,
    duckdb_path: str | None = None,
) -> dict:
    """Fetch earnings for every current member of ``universe`` from yfinance.

    Membership is the *current* snapshot — historical members (delisted) won't
    have current Ticker pages. That's acceptable: the dataset's labels are
    point-in-time and only stocks active during the panel window matter.
    """
    from packages.ingestion.universe.membership import members_on

    if symbols is None:
        members = members_on(universe, datetime.now(UTC).date())
        symbols = members["symbol"].tolist()

    if not symbols:
        log.warning(f"earnings backfill: no symbols for {universe}")
        return {"requested": 0, "succeeded": 0, "failed": 0, "rows_written": 0, "failures": []}

    log.info(f"earnings backfill: {len(symbols)} {universe} symbols (yfinance)")

    succeeded = failed = rows_written = 0
    failures: list[dict] = []

    with _earnings_conn(duckdb_path) as conn:
        for i, sym in enumerate(symbols):
            try:
                if i > 0:
                    time.sleep(_PER_SYMBOL_SLEEP_SEC)
                df = _yf_fetch_earnings(sym, universe=universe)
                if df.empty:
                    failed += 1
                    failures.append({"symbol": sym, "error": "no earnings returned"})
                    continue
                n = upsert_earnings(df, conn=conn)
                rows_written += n
                succeeded += 1
                if (i + 1) % 50 == 0:
                    log.info(f"  ... {i + 1}/{len(symbols)} done ({rows_written} rows)")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append({"symbol": sym, "error": repr(exc)})
                log.error(f"failed {sym}: {exc!r}")

    log.info(
        f"earnings backfill complete: succeeded={succeeded} failed={failed} "
        f"rows_written={rows_written}"
    )
    return {
        "requested": len(symbols),
        "succeeded": succeeded,
        "failed": failed,
        "rows_written": rows_written,
        "failures": failures,
    }


def load_earnings(
    symbols: list[str] | None = None,
    *,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """Return earnings rows as a DataFrame, optionally filtered to ``symbols``."""
    with _earnings_conn(duckdb_path) as conn:
        if symbols is None:
            return conn.execute(
                "SELECT * FROM earnings_calendar ORDER BY symbol, announce_date"
            ).df()
        return conn.execute(
            "SELECT * FROM earnings_calendar WHERE symbol = ANY(?) ORDER BY symbol, announce_date",
            [symbols],
        ).df()


def has_earnings_data(*, duckdb_path: str | None = None) -> bool:
    """Cheap availability probe for the earnings feature group.

    Opens READ-ONLY so it coexists with other processes holding the DB open
    (e.g. a walk-forward backtest). A read-write open raises IOException under
    that contention; the extension resolver then treats the group as
    unavailable and silently drops it, breaking inference with missing
    columns. Returns False if the DB/table can't be read.
    """
    conn = None
    try:
        conn = _connect(duckdb_path, read_only=True)
        row = conn.execute("SELECT COUNT(*) FROM earnings_calendar").fetchone()
        return bool(row and row[0] > 0)
    except duckdb.Error:
        return False
    finally:
        if conn is not None:
            conn.close()
