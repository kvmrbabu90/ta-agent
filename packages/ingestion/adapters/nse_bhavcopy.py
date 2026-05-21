"""NSE bhavcopy (daily equity OHLC archive) ingestion adapter.

NSE publishes a daily bhavcopy ZIP for every trading day going back to ~2010.
The URL format is:
    https://archives.nseindia.com/content/historical/EQUITIES/{YYYY}/{MMM}/
        cm{DDMMMYYYY}bhav.csv.zip

where MMM is the 3-letter uppercase month code.

This adapter:
  - Fetches the archive day by day with a session that holds the bot-
    mitigation cookies set by nseindia.com.
  - Parses the ZIP -> CSV, filters to SERIES='EQ' (cash equities, no
    bonds/ETFs/derivatives), keeps OPEN/HIGH/LOW/CLOSE/TOTTRDQTY.
  - Optionally filters to a universe's symbol list before upserting.
  - Upserts into ``ohlcv_daily`` with source='bhavcopy', exchange='NSE'.

Used to extend NIFTY100 history back to 2014 (Kite Connect only gives
us 2021+, which left the strict WF with only 3.3 years OOS).

Rate-limited 0.5s/req to be polite to NSE infrastructure. ~2750 trading
days × 0.5s = ~25 min for a full 2014-2026 backfill.
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import requests

from packages.common.logging import log
from packages.ingestion.storage import upsert_ohlcv

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/all-reports",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

_MONTHS = ("", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

_RATE_LIMIT_SECONDS = 0.5


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    # Visit homepage so Akamai sets bot-mitigation cookies. The actual
    # response is 403 but the Set-Cookie still comes through.
    try:
        s.get("https://www.nseindia.com/", timeout=10)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"nse_bhavcopy: failed to seed session cookies: {exc!r}")
    return s


def _bhavcopy_url_old(d: date) -> str:
    """Old ZIP format. Empirically reliable for ~2010-mid-2024; sporadic
    later (NSE quietly stopped writing the old format around mid-2024)."""
    mmm = _MONTHS[d.month]
    yyyy = d.year
    return (
        f"https://archives.nseindia.com/content/historical/EQUITIES/"
        f"{yyyy}/{mmm}/cm{d.day:02d}{mmm}{yyyy}bhav.csv.zip"
    )


def _bhavcopy_url_modern(d: date) -> str:
    """Modern CSV format (sec_bhavdata_full_DDMMYYYY.csv). Available
    from ~2020 onward and continuing. Used as a fallback when the old
    ZIP returns 404."""
    return (
        f"https://archives.nseindia.com/products/content/"
        f"sec_bhavdata_full_{d.day:02d}{d.month:02d}{d.year}.csv"
    )


def _parse_bhavcopy_csv(csv_bytes: bytes, on_date: date) -> pd.DataFrame:
    """Parse either format. Old: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE,
    LAST, PREVCLOSE, TOTTRDQTY, TOTTRDVAL, TIMESTAMP, TOTALTRADES, ISIN.
    Modern: SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE,
    LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, ...
    """
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()
    if df.empty:
        return df
    # Detect format by column presence.
    if "OPEN" in df.columns:
        open_col, high_col, low_col, close_col, vol_col = (
            "OPEN", "HIGH", "LOW", "CLOSE", "TOTTRDQTY",
        )
    elif "OPEN_PRICE" in df.columns:
        open_col, high_col, low_col, close_col, vol_col = (
            "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY",
        )
    else:
        log.warning(f"nse_bhavcopy: unknown CSV columns {list(df.columns)}")
        return pd.DataFrame()
    out = pd.DataFrame({
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "bar_date": on_date,
        "open": pd.to_numeric(df[open_col], errors="coerce"),
        "high": pd.to_numeric(df[high_col], errors="coerce"),
        "low": pd.to_numeric(df[low_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
        "volume": pd.to_numeric(df[vol_col], errors="coerce").fillna(0).astype("int64"),
    })
    out["close_unadj"] = out["close"]  # bhavcopy is split-unadjusted at source
    out["exchange"] = "NSE"
    out["source"] = "bhavcopy"
    out["ingested_at"] = pd.Timestamp.utcnow()
    # Drop rows where any OHLC is null
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def fetch_bhavcopy(
    on_date: date, *, session: requests.Session | None = None
) -> pd.DataFrame:
    """Download one day's bhavcopy and return it as a canonical OHLCV frame.
    Tries old ZIP first, falls back to modern CSV if 404. Empty DataFrame
    if NSE returns 404 from both (non-trading day or missing date).
    """
    own_session = session is None
    if own_session:
        session = _build_session()

    # First: old ZIP format
    csv_bytes: bytes | None = None
    try:
        r = session.get(_bhavcopy_url_old(on_date), timeout=20)
        if r.status_code == 200 and len(r.content) >= 1000:
            try:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    csv_bytes = z.read(z.namelist()[0])
            except zipfile.BadZipFile:
                csv_bytes = None
    except Exception as exc:  # noqa: BLE001
        log.warning(f"nse_bhavcopy: {on_date} old-fmt fetch error {exc!r}")

    # Fallback: modern CSV format
    if csv_bytes is None:
        try:
            r = session.get(_bhavcopy_url_modern(on_date), timeout=20)
            if r.status_code == 200 and len(r.content) >= 1000:
                csv_bytes = r.content
        except Exception as exc:  # noqa: BLE001
            log.warning(f"nse_bhavcopy: {on_date} modern-fmt fetch error {exc!r}")

    if csv_bytes is None:
        if own_session:
            session.close()
        return pd.DataFrame()

    df = _parse_bhavcopy_csv(csv_bytes, on_date)
    if own_session:
        session.close()
    return df


def _upsert_with_retry(
    df: pd.DataFrame, *, attempts: int = 90, delay_s: float = 20.0
) -> None:
    """upsert_ohlcv but retry on DuckDB file-lock conflicts.

    The SP500 strict WF holds the DuckDB file with write access during
    training-dataset assembly (briefly, once per retrain). If we collide
    with it, we wait and try again. 90 attempts × 20s = 30 minutes of
    retry window — generous enough to outlast any single SP500 retrain
    (~75 min) we might collide with.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            upsert_ohlcv(df)
            return
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Cannot open file" not in msg and "IO Error" not in msg \
                    and "already open" not in msg:
                raise  # not a lock error
            last_exc = exc
            log.warning(
                f"nse_bhavcopy: upsert attempt {i+1}/{attempts} blocked by "
                f"DuckDB writer; sleeping {delay_s:.0f}s"
            )
            time.sleep(delay_s)
    if last_exc is not None:
        raise last_exc  # pragma: no cover


def backfill(
    start: date,
    end: date,
    *,
    symbols: Iterable[str] | None = None,
    flush_every_days: int = 90,
) -> dict:
    """Iterate every calendar day in [start, end], fetch bhavcopy, filter to
    ``symbols`` (if provided), buffer in memory, and bulk-upsert into
    ``ohlcv_daily`` periodically.

    Buffered writes (not per-day) so a concurrent DuckDB writer can't
    starve us — and so we don't hold a DuckDB write lock for the whole
    25-minute fetch. Default flush cadence is ~3 calendar months of data
    at a time, which is small enough to retry cheaply if the writer is
    busy.

    Returns a summary dict {requested, succeeded, failed, rows_written}.
    """
    sym_set = set(symbols) if symbols else None
    session = _build_session()
    counters = {"requested": 0, "succeeded": 0, "failed": 0, "rows_written": 0}
    d = start
    n_days = (end - start).days + 1
    buffer: list[pd.DataFrame] = []
    days_since_flush = 0

    def _flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        big = pd.concat(buffer, ignore_index=True)
        _upsert_with_retry(big)
        counters["rows_written"] += len(big)
        log.info(
            f"nse_bhavcopy: flushed {len(big):,} rows to ohlcv_daily "
            f"(cum {counters['rows_written']:,})"
        )
        buffer = []

    while d <= end:
        counters["requested"] += 1
        if d.weekday() < 5:
            df = fetch_bhavcopy(d, session=session)
            if df.empty:
                counters["failed"] += 1
            else:
                if sym_set is not None:
                    df = df[df["symbol"].isin(sym_set)]
                if not df.empty:
                    buffer.append(df)
                    counters["succeeded"] += 1
                else:
                    counters["failed"] += 1
            time.sleep(_RATE_LIMIT_SECONDS)
            days_since_flush += 1
        if days_since_flush >= flush_every_days:
            _flush()
            days_since_flush = 0
        if counters["requested"] % 100 == 0:
            log.info(
                f"nse_bhavcopy: {counters['requested']}/{n_days} days processed "
                f"(succeeded={counters['succeeded']} failed={counters['failed']} "
                f"rows_buffered={sum(len(b) for b in buffer)} "
                f"rows_written={counters['rows_written']})"
            )
        d += timedelta(days=1)
    _flush()  # final flush
    session.close()
    log.info(
        f"nse_bhavcopy backfill complete: requested={counters['requested']} "
        f"succeeded={counters['succeeded']} failed={counters['failed']} "
        f"rows_written={counters['rows_written']}"
    )
    return counters


__all__ = ["fetch_bhavcopy", "backfill"]
