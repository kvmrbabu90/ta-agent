"""Kite Connect (Zerodha) adapter for daily OHLCV bars (NIFTY 100 / NSE).

Public API mirrors ``ib_adapter`` so the two are interchangeable per universe:
    fetch_daily_bars(symbol, start, end, *, kite=None, instrument_token=None)
    backfill_universe(universe, start, end, symbols=None, *, duckdb_path=None)
    daily_update(universe, *, duckdb_path=None)
    load_instrument_map(kite, symbols=None) -> dict[str, int]

Kite-specific quirks:
- Historical-data calls take an ``instrument_token`` (int), not a ticker —
  resolved once per backfill via ``kite.instruments(exchange="NSE")``.
- Daily-candle calls are capped at ~2000 calendar days per request, so long
  windows are chunked.
- Rate limit is 3 req/sec for historical endpoints. We pace at 0.4s between
  calls (≈2.5 req/sec) for headroom.
- Access tokens expire daily ~6am IST. On TokenException we log a clear
  re-auth instruction and abort the run — Kite's flow requires manual
  browser interaction; auto-refresh is not possible.
- Prices are split-adjusted but not dividend-adjusted. We store
  ``close_unadj == close`` for now; cross-check against yfinance in Phase 3.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException

from packages.common.config import settings
from packages.common.logging import log
from packages.common.schemas import OHLCV_COLUMNS
from packages.ingestion.storage import get_conn, upsert_ohlcv
from packages.ingestion.universe.membership import members_on

if TYPE_CHECKING:
    import duckdb  # noqa: F401

# 3 req/sec on Kite's historical endpoint. 0.4s ≈ 2.5 req/sec, safely under.
_REQUEST_INTERVAL_SEC: float = 0.4
# Kite caps daily-candle ranges at ~2000 days per call. Chunk to stay within.
_MAX_DAYS_PER_CALL: int = 2000
_SOURCE: str = "kite"
_EXCHANGE: str = "NSE"


# ---------------------------------------------------------------------------
# Pacer (mirrors ib_adapter._Pacer; kept local so each adapter is standalone)
# ---------------------------------------------------------------------------


def _throttle_sleep(seconds: float) -> None:
    """Indirection so tests can monkeypatch the rate limiter to a no-op."""
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
            log.debug(f"rate-limit pacing: sleeping {remaining:.2f}s")
            _throttle_sleep(remaining)
        self._last_ts = time.monotonic()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _create_kite() -> KiteConnect:
    """Factory for KiteConnect — isolated so tests can patch it."""
    return KiteConnect(api_key=settings.kite_api_key)


@contextmanager
def _kite_connection() -> Iterator[KiteConnect]:
    """Yield an authenticated KiteConnect client.

    Aborts immediately if ``KITE_ACCESS_TOKEN`` is not set — the user must
    run ``scripts.kite_login`` first to mint a fresh token (they expire daily).
    """
    if not settings.kite_access_token:
        raise RuntimeError(
            "KITE_ACCESS_TOKEN is empty. Tokens expire daily ~6am IST — "
            "run `python -m scripts.kite_login` and update .env."
        )
    kite = _create_kite()
    kite.set_access_token(settings.kite_access_token)
    try:
        yield kite
    finally:
        # KiteConnect is a thin REST wrapper — nothing to clean up.
        pass


# ---------------------------------------------------------------------------
# Instrument map
# ---------------------------------------------------------------------------


def load_instrument_map(
    kite: KiteConnect, symbols: list[str] | None = None
) -> dict[str, int]:
    """Build {tradingsymbol: instrument_token} for NSE equities.

    Kite returns ~2000 instruments per exchange; filtering to segment='NSE'
    and instrument_type='EQ' drops options/futures/etc. If ``symbols`` is
    provided, the map is restricted to that intersection.
    """
    instruments = kite.instruments(exchange=_EXCHANGE)
    eq_only = (
        i
        for i in instruments
        if i.get("segment") == "NSE" and i.get("instrument_type") == "EQ"
    )
    full = {i["tradingsymbol"]: int(i["instrument_token"]) for i in eq_only}
    if symbols is None:
        return full
    return {s: full[s] for s in symbols if s in full}


# ---------------------------------------------------------------------------
# Bar conversion
# ---------------------------------------------------------------------------


def _coerce_bar_date(value: object) -> date:
    """Normalize Kite's date field (datetime/date/string) to a plain date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _bars_to_ohlcv(bars: list[dict], *, symbol: str) -> pd.DataFrame:
    """Convert Kite historical_data dicts to a DataFrame matching OHLCV_COLUMNS.

    Kite returns prices split-adjusted but not dividend-adjusted. We store
    ``close_unadj = close`` for now (no separate unadjusted series available
    from this source) — yfinance cross-check in Phase 3.
    """
    if not bars:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    n = len(bars)
    now_utc = datetime.now(UTC)
    closes = [float(b["close"]) for b in bars]
    out = pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "exchange": [_EXCHANGE] * n,
            "bar_date": [_coerce_bar_date(b["date"]) for b in bars],
            "open": [float(b["open"]) for b in bars],
            "high": [float(b["high"]) for b in bars],
            "low": [float(b["low"]) for b in bars],
            "close": closes,
            "volume": [int(b["volume"]) for b in bars],
            "close_unadj": closes,
            "source": [_SOURCE] * n,
            "ingested_at": [now_utc] * n,
        }
    )
    return out[OHLCV_COLUMNS]


# ---------------------------------------------------------------------------
# Chunked historical fetch
# ---------------------------------------------------------------------------


def _chunked_date_ranges(
    start: date, end: date, max_days: int = _MAX_DAYS_PER_CALL
) -> Iterator[tuple[date, date]]:
    """Yield contiguous (chunk_start, chunk_end) ranges each <= max_days long."""
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=max_days - 1))
        yield (cur, chunk_end)
        cur = chunk_end + timedelta(days=1)


def _fetch_inner(
    kite: KiteConnect,
    symbol: str,
    start: date,
    end: date,
    instrument_token: int,
    pacer: _Pacer | None = None,
) -> pd.DataFrame:
    """Fetch all chunks for one symbol, paced. Caller supplies the resolved token."""
    pacer = pacer or _Pacer()
    all_bars: list[dict] = []
    for i, (chunk_start, chunk_end) in enumerate(_chunked_date_ranges(start, end)):
        if i > 0:
            pacer.wait()
        chunk = kite.historical_data(instrument_token, chunk_start, chunk_end, "day")
        if chunk:
            all_bars.extend(chunk)
    return _bars_to_ohlcv(all_bars, symbol=symbol)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_daily_bars(
    symbol: str,
    start: date,
    end: date,
    *,
    kite: KiteConnect | None = None,
    instrument_token: int | None = None,
) -> pd.DataFrame:
    """Fetch one NSE symbol's daily bars over [start, end] inclusive.

    Returns a DataFrame in canonical OHLCV form (source='kite', exchange='NSE').
    If ``instrument_token`` is None, the instrument map is fetched to resolve
    the symbol — which costs one extra API call. Bulk callers should resolve
    once via ``load_instrument_map`` and pass the token in.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    own_kite = kite is None
    if own_kite:
        with _kite_connection() as k:
            return _resolve_and_fetch(k, symbol, start, end, instrument_token)
    return _resolve_and_fetch(kite, symbol, start, end, instrument_token)


def _resolve_and_fetch(
    kite: KiteConnect,
    symbol: str,
    start: date,
    end: date,
    instrument_token: int | None,
) -> pd.DataFrame:
    if instrument_token is None:
        m = load_instrument_map(kite, symbols=[symbol])
        if symbol not in m:
            log.warning(f"{symbol} not in NSE instrument map; returning empty")
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        instrument_token = m[symbol]
    return _fetch_inner(kite, symbol, start, end, instrument_token)


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
    """Backfill daily NSE bars for every current member of ``universe``.

    Aborts the whole run on TokenException (re-auth required); per-symbol
    failures are captured and do not abort.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    if symbols is None:
        members = _membership_for_universe(universe)
        symbols_list = members["symbol"].tolist()
    else:
        symbols_list = list(symbols)

    if not symbols_list:
        log.warning(f"backfill_universe: no symbols for universe={universe}")
        return _empty_result(0)

    log.info(
        f"starting Kite backfill of {len(symbols_list)} symbols in {universe} "
        f"[{start}..{end}]"
    )

    pacer = _Pacer()
    succeeded = failed = rows_written = 0
    failures: list[dict] = []
    aborted = False

    with _kite_connection() as kite, get_conn(duckdb_path) as duck:
        try:
            inst_map = load_instrument_map(kite, symbols=symbols_list)
        except TokenException:
            log.error(
                "Kite access token rejected at instrument-map load. "
                "Run `python -m scripts.kite_login` and update .env. Aborting."
            )
            return _empty_result(len(symbols_list))

        for sym in symbols_list:
            try:
                if sym not in inst_map:
                    log.warning(f"{sym} not found in NSE instrument map; skipping")
                    failed += 1
                    failures.append(
                        {"symbol": sym, "error": "not in NSE instrument map"}
                    )
                    continue
                df = _fetch_inner(
                    kite, sym, start, end, inst_map[sym], pacer=pacer
                )
                if df.empty:
                    log.warning(f"no bars returned for {sym}; skipping")
                    failed += 1
                    failures.append({"symbol": sym, "error": "no bars returned"})
                    continue
                n = upsert_ohlcv(df, conn=duck)
                rows_written += n
                succeeded += 1
                log.info(f"completed {sym} ({n} bars)")
            except TokenException:
                log.error(
                    f"Kite token expired mid-run at {sym}. "
                    "Re-run `python -m scripts.kite_login` and retry. Aborting."
                )
                failed += 1
                failures.append({"symbol": sym, "error": "token expired - aborted"})
                aborted = True
                break
            except Exception as exc:  # noqa: BLE001 — keep going on per-symbol errors
                failed += 1
                log.error(f"failed {sym}: {exc!r}")
                failures.append({"symbol": sym, "error": repr(exc)})

    log.info(
        f"Kite backfill complete (aborted={aborted}): requested={len(symbols_list)} "
        f"succeeded={succeeded} failed={failed} rows_written={rows_written}"
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
    """Pull the missing tail of bars for each current member of ``universe``."""
    members = _membership_for_universe(universe)
    symbols_list = members["symbol"].tolist()

    if not symbols_list:
        log.warning(f"daily_update: no symbols for universe={universe}")
        return _empty_result(0)

    today = date.today()
    pacer = _Pacer()
    succeeded = failed = rows_written = 0
    failures: list[dict] = []
    aborted = False

    log.info(
        f"starting Kite daily update for {universe} "
        f"({len(symbols_list)} symbols, end={today})"
    )

    with _kite_connection() as kite, get_conn(duckdb_path) as duck:
        try:
            inst_map = load_instrument_map(kite, symbols=symbols_list)
        except TokenException:
            log.error(
                "Kite access token rejected at instrument-map load. "
                "Run `python -m scripts.kite_login` and update .env. Aborting."
            )
            return _empty_result(len(symbols_list))

        for sym in symbols_list:
            try:
                if sym not in inst_map:
                    log.warning(f"{sym} not in NSE instrument map; skipping")
                    failed += 1
                    failures.append(
                        {"symbol": sym, "error": "not in NSE instrument map"}
                    )
                    continue
                last = _last_bar_date_for_symbol(sym, duck)
                start = (
                    last + timedelta(days=1)
                    if last
                    else today - timedelta(days=60)
                )
                if start > today:
                    log.debug(f"{sym} already up to date (last={last})")
                    succeeded += 1
                    continue
                df = _fetch_inner(
                    kite, sym, start, today, inst_map[sym], pacer=pacer
                )
                if df.empty:
                    failed += 1
                    failures.append({"symbol": sym, "error": "no bars returned"})
                    continue
                n = upsert_ohlcv(df, conn=duck)
                rows_written += n
                succeeded += 1
                log.info(f"completed {sym} ({n} bars from {start})")
            except TokenException:
                log.error(
                    f"Kite token expired mid-run at {sym}. "
                    "Re-run `python -m scripts.kite_login` and retry. Aborting."
                )
                failed += 1
                failures.append({"symbol": sym, "error": "token expired - aborted"})
                aborted = True
                break
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.error(f"failed {sym}: {exc!r}")
                failures.append({"symbol": sym, "error": repr(exc)})

    log.info(
        f"Kite daily_update complete (aborted={aborted}): requested={len(symbols_list)} "
        f"succeeded={succeeded} failed={failed} rows_written={rows_written}"
    )
    return {
        "requested": len(symbols_list),
        "succeeded": succeeded,
        "failed": failed,
        "rows_written": rows_written,
        "failures": failures,
    }
