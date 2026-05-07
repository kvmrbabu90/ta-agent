"""Interactive Brokers adapter for daily OHLCV bars (US universe).

Public API:
    wiki_to_ib_symbol(s)            — symbol normalization
    fetch_daily_bars(symbol, start, end, *, exchange, ib=None)
                                    — single-symbol fetch
    backfill_universe(universe, start, end, symbols=None, *, duckdb_path=None)
                                    — bulk historical backfill
    daily_update(universe, *, duckdb_path=None)
                                    — incremental tail update

IB historical-data limit is roughly 60 requests / 10 minutes. We pace at one
request every 10 seconds to stay well clear; this is conservative on purpose
(IB throttling is opaque and the cost of a rejection is much higher than the
cost of a slower run).

Adjusted prices: IB returns split-and-dividend-adjusted closes when
whatToShow='TRADES' and useRTH=True. We store close_unadj=None here and
cross-check with yfinance later (Phase 3).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from ib_insync import IB, Stock

from packages.common.config import settings
from packages.common.logging import log
from packages.common.schemas import OHLCV_COLUMNS
from packages.ingestion.storage import get_conn, upsert_ohlcv
from packages.ingestion.universe.membership import members_on

if TYPE_CHECKING:
    import duckdb  # noqa: F401

# IB allows ~60 historical-data requests per 10 minutes. Pace at one per
# 10 seconds — well below the cap, deterministic, and trivial to reason about.
_REQUEST_INTERVAL_SEC: float = 10.0
_SOURCE: str = "ib"
_VALID_PRIMARY_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def wiki_to_ib_symbol(symbol: str) -> str:
    """Map Wikipedia-style tickers to the form IB expects.

    Wikipedia uses dotted share-class notation (BRK.B, BF.B); IB uses
    a space (BRK B, BF B). All other symbols pass through unchanged.
    """
    return symbol.replace(".", " ")


def _throttle_sleep(seconds: float) -> None:
    """Indirection layer so tests can monkeypatch the rate limiter to a no-op."""
    if seconds > 0:
        time.sleep(seconds)


class _Pacer:
    """Keeps consecutive calls at least ``interval_sec`` apart."""

    def __init__(self, interval_sec: float = _REQUEST_INTERVAL_SEC) -> None:
        self.interval_sec = interval_sec
        self._last_ts: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_ts is None:
            self._last_ts = now
            return
        remaining = self.interval_sec - (now - self._last_ts)
        if remaining > 0:
            log.debug(f"rate-limit pacing: sleeping {remaining:.1f}s")
            _throttle_sleep(remaining)
        self._last_ts = time.monotonic()


def _create_ib() -> IB:
    """Factory for IB() — isolated so tests can patch it."""
    return IB()


@contextmanager
def _ib_connection() -> Iterator[IB]:
    """Open one IB connection, yield it, and disconnect cleanly on exit."""
    ib = _create_ib()
    log.info(
        f"connecting to IB at {settings.ib_host}:{settings.ib_port} "
        f"(clientId={settings.ib_client_id})"
    )
    ib.connect(
        settings.ib_host,
        settings.ib_port,
        clientId=settings.ib_client_id,
        readonly=True,
    )
    try:
        yield ib
    finally:
        try:
            ib.disconnect()
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            log.warning(f"ib disconnect raised (ignored): {exc!r}")


# ---------------------------------------------------------------------------
# Bar conversion
# ---------------------------------------------------------------------------


def _coerce_bar_date(value: object) -> date:
    """Normalize a BarData date attribute (date or datetime or string) to date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _bars_to_ohlcv(bars: list, *, symbol: str, exchange: str) -> pd.DataFrame:
    """Convert a list of bar objects (BarData / SimpleNamespace / similar) to a
    DataFrame matching OHLCV_COLUMNS. Reads attributes directly rather than
    relying on ``ib_insync.util.df`` so test fixtures stay simple."""
    if not bars:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    n = len(bars)
    now_utc = datetime.now(UTC)
    out = pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "exchange": [exchange] * n,
            "bar_date": [_coerce_bar_date(b.date) for b in bars],
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [int(b.volume) for b in bars],
            "close_unadj": pd.Series([None] * n, dtype=object),
            "source": [_SOURCE] * n,
            "ingested_at": [now_utc] * n,
        }
    )
    return out[OHLCV_COLUMNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_daily_bars(
    symbol: str,
    start: date,
    end: date,
    *,
    exchange: str = "NASDAQ",
    ib: IB | None = None,
) -> pd.DataFrame:
    """Fetch one symbol's daily bars over [start, end] inclusive.

    Returns a DataFrame in canonical OHLCV form (close_unadj=None, source='ib').
    Caller may pass an existing IB connection to avoid reconnect churn; if
    ``ib`` is None, a fresh connection is opened just for this call.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    ib_symbol = wiki_to_ib_symbol(symbol)
    duration_days = (end - start).days + 1
    # IB rejects daily-bar requests > 365 days specified in days — must use
    # years. Round up so we always cover the requested window.
    if duration_days > 365:
        duration_years = (duration_days + 364) // 365
        duration_str = f"{duration_years} Y"
    else:
        duration_str = f"{duration_days} D"
    end_dt_str = f"{end.strftime('%Y%m%d')} 23:59:59"

    primary = exchange if exchange in _VALID_PRIMARY_EXCHANGES else ""
    contract = Stock(ib_symbol, "SMART", "USD", primaryExchange=primary)

    def _request(conn: IB) -> list:
        return conn.reqHistoricalData(
            contract,
            endDateTime=end_dt_str,
            durationStr=duration_str,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            keepUpToDate=False,
        )

    if ib is None:
        with _ib_connection() as conn:
            bars = _request(conn)
    else:
        bars = _request(ib)

    return _bars_to_ohlcv(bars, symbol=symbol, exchange=exchange)


def _membership_for_universe(universe: str, as_of: date | None = None) -> pd.DataFrame:
    if as_of is None:
        as_of = date.today()
    return members_on(universe, as_of)


def _empty_result(requested: int) -> dict:
    return {
        "requested": requested,
        "succeeded": 0,
        "failed": 0,
        "rows_written": 0,
        "failures": [],
    }


def backfill_universe(
    universe: str,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    *,
    duckdb_path: str | None = None,
) -> dict:
    """Backfill daily bars for every current member of ``universe``.

    Failures on individual symbols are captured in the returned dict and do
    not abort the run. Rate-limit pacing is enforced between symbols.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    if symbols is None:
        members = _membership_for_universe(universe)
        symbols_list = members["symbol"].tolist()
        exchange_map = dict(zip(members["symbol"], members["exchange"], strict=True))
    else:
        symbols_list = list(symbols)
        exchange_map = {}

    if not symbols_list:
        log.warning(f"backfill_universe: no symbols for universe={universe}")
        return _empty_result(0)

    log.info(
        f"starting backfill of {len(symbols_list)} symbols in {universe} "
        f"[{start}..{end}]"
    )

    pacer = _Pacer()
    succeeded = failed = rows_written = 0
    failures: list[dict] = []

    with _ib_connection() as ib, get_conn(duckdb_path) as duck:
        for i, sym in enumerate(symbols_list):
            try:
                if i > 0:
                    pacer.wait()
                exchange = exchange_map.get(sym, "NASDAQ")
                df = fetch_daily_bars(sym, start, end, exchange=exchange, ib=ib)
                if df.empty:
                    log.warning(f"no bars returned for {sym}; skipping")
                    failed += 1
                    failures.append({"symbol": sym, "error": "no bars returned"})
                    continue
                n = upsert_ohlcv(df, conn=duck)
                rows_written += n
                succeeded += 1
                log.info(f"completed {sym} ({n} bars)")
            except Exception as exc:  # noqa: BLE001 — keep going on per-symbol errors
                failed += 1
                log.error(f"failed {sym}: {exc!r}")
                failures.append({"symbol": sym, "error": repr(exc)})

    log.info(
        f"backfill complete: requested={len(symbols_list)} succeeded={succeeded} "
        f"failed={failed} rows_written={rows_written}"
    )
    return {
        "requested": len(symbols_list),
        "succeeded": succeeded,
        "failed": failed,
        "rows_written": rows_written,
        "failures": failures,
    }


def _last_bar_date_for_symbol(symbol: str, conn) -> date | None:
    row = conn.execute(
        "SELECT MAX(bar_date) FROM ohlcv_daily WHERE symbol = ? AND source = ?",
        [symbol, _SOURCE],
    ).fetchone()
    return row[0] if row and row[0] else None


def daily_update(universe: str, *, duckdb_path: str | None = None) -> dict:
    """Pull the missing tail of bars for each current member of ``universe``.

    For each symbol, queries DuckDB for the latest stored bar (source='ib')
    and fetches from the day after that through today. Symbols with no
    stored history default to a 60-day lookback.
    """
    members = _membership_for_universe(universe)
    symbols_list = members["symbol"].tolist()
    exchange_map = dict(zip(members["symbol"], members["exchange"], strict=True))

    if not symbols_list:
        log.warning(f"daily_update: no symbols for universe={universe}")
        return _empty_result(0)

    today = date.today()
    pacer = _Pacer()
    succeeded = failed = rows_written = 0
    failures: list[dict] = []

    log.info(
        f"starting daily update for {universe} ({len(symbols_list)} symbols, end={today})"
    )

    with _ib_connection() as ib, get_conn(duckdb_path) as duck:
        for i, sym in enumerate(symbols_list):
            try:
                last = _last_bar_date_for_symbol(sym, duck)
                start = last + timedelta(days=1) if last else today - timedelta(days=60)
                if start > today:
                    log.debug(f"{sym} already up to date (last={last})")
                    succeeded += 1
                    continue
                if i > 0:
                    pacer.wait()
                exchange = exchange_map.get(sym, "NASDAQ")
                df = fetch_daily_bars(sym, start, today, exchange=exchange, ib=ib)
                if df.empty:
                    failed += 1
                    failures.append({"symbol": sym, "error": "no bars returned"})
                    continue
                n = upsert_ohlcv(df, conn=duck)
                rows_written += n
                succeeded += 1
                log.info(f"completed {sym} ({n} bars from {start})")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.error(f"failed {sym}: {exc!r}")
                failures.append({"symbol": sym, "error": repr(exc)})

    log.info(
        f"daily_update complete: requested={len(symbols_list)} succeeded={succeeded} "
        f"failed={failed} rows_written={rows_written}"
    )
    return {
        "requested": len(symbols_list),
        "succeeded": succeeded,
        "failed": failed,
        "rows_written": rows_written,
        "failures": failures,
    }
