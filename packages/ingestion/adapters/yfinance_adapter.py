"""yfinance adapter — third data source for sanity-checking and as a
no-credentials fallback when IB or Kite are unavailable.

Treat yfinance as the noisiest of the three: it scrapes Yahoo unofficially,
prices are best-effort, and adjustments occasionally lag corporate actions
by a day or two. We use it for cross-checks (Phase 3 corporate_actions module)
and as a tail-end fallback in the daily ingest pipeline — never as ground truth.

Public API matches ib_adapter / kite_adapter:
    fetch_daily_bars(symbol, start, end, *, universe=None)
    backfill_universe(universe, start, end, symbols=None, *, duckdb_path=None)
    daily_update(universe, *, duckdb_path=None)
    to_yfinance_symbol(symbol, universe)
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

from packages.common.logging import log
from packages.common.schemas import OHLCV_COLUMNS
from packages.ingestion.storage import get_conn, upsert_ohlcv
from packages.ingestion.universe.membership import members_on

if TYPE_CHECKING:
    import duckdb  # noqa: F401

# Conservative pacing — yfinance is unofficial and easy to get rate-limited on.
_PER_SYMBOL_SLEEP_SEC: float = 0.5
_BATCH_SIZE: int = 50
_BATCH_SLEEP_SEC: float = 2.0
_SOURCE: str = "yfinance"


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------


def to_yfinance_symbol(symbol: str, universe: str) -> str:
    """Map our canonical ticker to yfinance's expected form.

    SP500: BRK.B → BRK-B (Yahoo uses dashes for share classes).
    NIFTY100: RELIANCE → RELIANCE.NS (NSE suffix).
    """
    if universe == "NIFTY100":
        return f"{symbol}.NS"
    if universe == "SP500":
        return symbol.replace(".", "-")
    return symbol


def _exchange_for(symbol_row: dict, universe: str | None) -> str:
    """Best-effort exchange tag. Membership row's exchange wins; otherwise
    fall back to NASDAQ for SP500, NSE for NIFTY100, NASDAQ otherwise."""
    if symbol_row and symbol_row.get("exchange"):
        return symbol_row["exchange"]
    if universe == "NIFTY100":
        return "NSE"
    return "NASDAQ"


# ---------------------------------------------------------------------------
# Sleep indirection (so tests can no-op the rate limiter)
# ---------------------------------------------------------------------------


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# Download helper (single seam tests can patch)
# ---------------------------------------------------------------------------


def _download(symbol: str, start: date, end: date, auto_adjust: bool) -> pd.DataFrame:
    """Thin wrapper over yfinance.download. yfinance treats `end` as exclusive,
    so we add a day to make our [start, end] inclusive contract hold."""
    return yf.download(
        symbol,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
        actions=False,
    )


# ---------------------------------------------------------------------------
# Bar conversion
# ---------------------------------------------------------------------------


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex on columns when given a list of tickers and
    sometimes even for a single ticker. Collapse to flat OHLCV column names."""
    if isinstance(df.columns, pd.MultiIndex):
        # Take the first level (price-type), drop the ticker level.
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _to_ohlcv_frame(
    adjusted: pd.DataFrame,
    unadjusted: pd.DataFrame | None,
    *,
    symbol: str,
    exchange: str,
) -> pd.DataFrame:
    """Combine the adjusted-and-unadjusted yfinance frames into our canonical
    schema. ``close_unadj`` comes from the unadjusted frame when available."""
    if adjusted is None or adjusted.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    adj = _flatten_columns(adjusted)
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(adj.columns):
        log.warning(
            f"yfinance returned unexpected columns for {symbol}: {list(adj.columns)}"
        )
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    bar_dates = [d.date() if isinstance(d, datetime) else pd.Timestamp(d).date() for d in adj.index]
    n = len(adj)

    if unadjusted is not None and not unadjusted.empty:
        unadj = _flatten_columns(unadjusted)
        unadj_close_map = {
            (d.date() if isinstance(d, datetime) else pd.Timestamp(d).date()): float(c)
            for d, c in zip(unadj.index, unadj["Close"], strict=False)
        }
        close_unadj = [unadj_close_map.get(d) for d in bar_dates]
    else:
        close_unadj = [None] * n

    now_utc = datetime.now(UTC)
    out = pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "exchange": [exchange] * n,
            "bar_date": bar_dates,
            "open": [float(v) for v in adj["Open"]],
            "high": [float(v) for v in adj["High"]],
            "low": [float(v) for v in adj["Low"]],
            "close": [float(v) for v in adj["Close"]],
            "volume": [int(v) for v in adj["Volume"]],
            "close_unadj": close_unadj,
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
    universe: str | None = None,
    exchange: str | None = None,
) -> pd.DataFrame:
    """Fetch one symbol's daily bars over [start, end] inclusive.

    Two yfinance calls are made: one with ``auto_adjust=True`` for the
    canonical adjusted OHLC, and a second with ``auto_adjust=False`` to
    populate ``close_unadj``. The second call is best-effort — if it fails
    or returns nothing, ``close_unadj`` is None.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    yf_symbol = to_yfinance_symbol(symbol, universe or "")

    adjusted = _download(yf_symbol, start, end, auto_adjust=True)
    try:
        unadjusted = _download(yf_symbol, start, end, auto_adjust=False)
    except Exception as exc:  # noqa: BLE001 — fall back to None on any error
        log.warning(f"unadjusted yfinance pull failed for {symbol}: {exc!r}")
        unadjusted = None

    ex = exchange or _exchange_for({}, universe)
    return _to_ohlcv_frame(adjusted, unadjusted, symbol=symbol, exchange=ex)


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
    """Backfill daily bars from yfinance for current members of ``universe``."""
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    if symbols is None:
        members = _membership_for_universe(universe)
        symbols_list = members["symbol"].tolist()
        exchange_map = dict(
            zip(members["symbol"], members["exchange"], strict=True)
        )
    else:
        symbols_list = list(symbols)
        # Fill exchange from membership when available, fallback later.
        members = _membership_for_universe(universe)
        exchange_map = dict(
            zip(members["symbol"], members["exchange"], strict=True)
        )

    if not symbols_list:
        log.warning(f"yfinance backfill: no symbols for universe={universe}")
        return _empty_result(0)

    log.info(
        f"starting yfinance backfill of {len(symbols_list)} symbols in {universe} "
        f"[{start}..{end}]"
    )

    succeeded = failed = rows_written = 0
    failures: list[dict] = []

    with get_conn(duckdb_path) as duck:
        for i, sym in enumerate(symbols_list):
            try:
                if i > 0 and i % _BATCH_SIZE == 0:
                    log.debug(f"yfinance batch boundary at {i}; sleeping {_BATCH_SLEEP_SEC}s")
                    _sleep(_BATCH_SLEEP_SEC)
                elif i > 0:
                    _sleep(_PER_SYMBOL_SLEEP_SEC)

                ex = exchange_map.get(sym) or _exchange_for({}, universe)
                df = fetch_daily_bars(
                    sym, start, end, universe=universe, exchange=ex
                )
                if df.empty:
                    failed += 1
                    failures.append({"symbol": sym, "error": "no bars returned"})
                    continue
                n = upsert_ohlcv(df, conn=duck)
                rows_written += n
                succeeded += 1
                log.info(f"completed {sym} ({n} bars)")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.error(f"failed {sym}: {exc!r}")
                failures.append({"symbol": sym, "error": repr(exc)})

    log.info(
        f"yfinance backfill complete: requested={len(symbols_list)} "
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
    """Pull missing tail bars from yfinance for current members of ``universe``."""
    members = _membership_for_universe(universe)
    symbols_list = members["symbol"].tolist()
    exchange_map = dict(zip(members["symbol"], members["exchange"], strict=True))

    if not symbols_list:
        log.warning(f"yfinance daily_update: no symbols for universe={universe}")
        return _empty_result(0)

    today = date.today()
    succeeded = failed = rows_written = 0
    failures: list[dict] = []

    log.info(
        f"starting yfinance daily update for {universe} "
        f"({len(symbols_list)} symbols, end={today})"
    )

    with get_conn(duckdb_path) as duck:
        for i, sym in enumerate(symbols_list):
            try:
                last = _last_bar_date_for_symbol(sym, duck)
                start = (
                    last + timedelta(days=1)
                    if last
                    else today - timedelta(days=60)
                )
                if start > today:
                    succeeded += 1
                    continue
                if i > 0 and i % _BATCH_SIZE == 0:
                    _sleep(_BATCH_SLEEP_SEC)
                elif i > 0:
                    _sleep(_PER_SYMBOL_SLEEP_SEC)

                ex = exchange_map.get(sym) or _exchange_for({}, universe)
                df = fetch_daily_bars(
                    sym, start, today, universe=universe, exchange=ex
                )
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
        f"yfinance daily_update complete: requested={len(symbols_list)} "
        f"succeeded={succeeded} failed={failed} rows_written={rows_written}"
    )
    return {
        "requested": len(symbols_list),
        "succeeded": succeeded,
        "failed": failed,
        "rows_written": rows_written,
        "failures": failures,
    }
