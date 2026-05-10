"""SEC EDGAR 8-K filings adapter (free, official).

Source: https://data.sec.gov/submissions/CIK{cik}.json — returns the full
filing history for a company in one JSON. No paid plan required.

We use 8-Ks (Form 8-K = "Current Report") as a free historical proxy for
news/events. Each 8-K corresponds to a "material event" the SEC requires
companies to disclose within 4 business days. Categories include:
    Item 1.01  Material agreements (M&A, partnerships)
    Item 1.02  Termination of material agreements
    Item 2.01  Completion of acquisition / disposition
    Item 2.02  Results of operations (earnings releases — overlap w/ EPS data)
    Item 3.02  Unregistered equity sales (placements)
    Item 5.02  Departure / appointment of officers (CEO/CFO turnover)
    Item 5.03  Articles / bylaws amendments
    Item 7.01  Regulation FD disclosure
    Item 8.01  Other events

EDGAR rate-limits at 10 requests/second per IP. SP500 has 503 symbols so
the full backfill takes ~50 seconds.

Schema (DuckDB):
    edgar_filings (
        cik             VARCHAR  NOT NULL,
        symbol          VARCHAR  NOT NULL,    -- our canonical ticker (the SP500 symbol)
        accession       VARCHAR  NOT NULL,    -- e.g. '0001193125-25-123456'
        filing_date     DATE     NOT NULL,
        form_type       VARCHAR  NOT NULL,    -- '8-K', '10-Q', '10-K', etc.
        items           VARCHAR,              -- comma-separated 8-K item codes ('2.02,5.02')
        primary_doc     VARCHAR,              -- HTML doc URL (for debugging)
        source          VARCHAR  NOT NULL,    -- 'edgar'
        ingested_at     TIMESTAMP NOT NULL,
        PRIMARY KEY (accession)
    )
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime

import duckdb
import pandas as pd
import requests

from packages.common.logging import log
from packages.ingestion.storage import _connect

_EDGAR_DDL = """
CREATE TABLE IF NOT EXISTS edgar_filings (
    cik           VARCHAR   NOT NULL,
    symbol        VARCHAR   NOT NULL,
    accession     VARCHAR   NOT NULL,
    filing_date   DATE      NOT NULL,
    form_type     VARCHAR   NOT NULL,
    items         VARCHAR,
    primary_doc   VARCHAR,
    source        VARCHAR   NOT NULL,
    ingested_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (accession)
);
CREATE INDEX IF NOT EXISTS edgar_symbol_date
    ON edgar_filings (symbol, filing_date);
CREATE INDEX IF NOT EXISTS edgar_form_type
    ON edgar_filings (form_type);
"""

_SOURCE = "edgar"
_USER_AGENT = "ta-agent local research konda@example.com"
_PER_REQUEST_SLEEP_SEC = 0.11  # SEC limit is 10 req/sec; we sleep 110ms to be safe


@contextmanager
def _edgar_conn(path: str | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = _connect(path)
    try:
        conn.execute(_EDGAR_DDL)
        yield conn
    finally:
        conn.close()


# --- CIK lookup --------------------------------------------------------------


def _fetch_company_tickers() -> dict[str, str]:
    """Pull the SEC's master ticker -> CIK map.

    Returns a dict mapping uppercase ticker -> 10-digit zero-padded CIK string.
    The endpoint is small (~1MB) and updated daily.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    out: dict[str, str] = {}
    for _, row in raw.items():
        ticker = str(row.get("ticker", "")).upper().strip()
        cik_int = row.get("cik_str")
        if not ticker or cik_int is None:
            continue
        out[ticker] = f"{int(cik_int):010d}"
    return out


def _to_edgar_ticker(symbol: str) -> str:
    """Normalize our canonical ticker to EDGAR's. SEC uses dot-share-class
    style ('BRK.B', not 'BRK-B' like yfinance)."""
    return symbol.replace("-", ".").upper()


# --- Filings fetch -----------------------------------------------------------


def _fetch_submissions_json(cik: str) -> dict | None:
    """Fetch the recent submissions metadata for one CIK.

    Returns None on 404 (CIK doesn't exist or recently delisted).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _parse_filings(submissions: dict, symbol: str, cik: str) -> pd.DataFrame:
    """Extract a flat DataFrame of recent filings from the submissions JSON.

    Submissions JSON has shape:
        {
          "filings": {
             "recent": {
               "accessionNumber": [...],
               "filingDate": [...],
               "form": [...],
               "items": [...],
               "primaryDocument": [...],
               ...
             },
             "files": [{"name": "CIK..-submissions-001.json", ...}, ...]
          },
          ...
        }

    "recent" holds the most recent ~1000 filings. Older filings live in the
    files[] paginated submissions; we don't fetch those — 1000 filings covers
    the last 5-15 years for most companies which is what we need.
    """
    filings = submissions.get("filings", {}).get("recent", {})
    if not filings:
        return pd.DataFrame()

    n = len(filings.get("accessionNumber", []))
    if n == 0:
        return pd.DataFrame()

    rows = []
    for i in range(n):
        accession = filings["accessionNumber"][i]
        try:
            filing_date = date.fromisoformat(filings["filingDate"][i])
        except (KeyError, ValueError):
            continue
        rows.append({
            "cik": cik,
            "symbol": symbol,
            "accession": accession,
            "filing_date": filing_date,
            "form_type": filings.get("form", [None] * n)[i],
            "items": filings.get("items", [None] * n)[i] or None,
            "primary_doc": filings.get("primaryDocument", [None] * n)[i],
        })
    return pd.DataFrame(rows)


# --- Upsert ------------------------------------------------------------------


def upsert_filings(
    df: pd.DataFrame, conn: duckdb.DuckDBPyConnection | None = None
) -> int:
    if df.empty:
        return 0
    own = conn is None
    if own:
        conn = _connect()
        conn.execute(_EDGAR_DDL)
    try:
        now_utc = datetime.now(UTC)
        out = df.copy()
        if "source" not in out.columns:
            out["source"] = _SOURCE
        if "ingested_at" not in out.columns:
            out["ingested_at"] = now_utc

        conn.register("filings_in", out)
        try:
            conn.execute(
                """
                INSERT INTO edgar_filings (
                    cik, symbol, accession, filing_date, form_type,
                    items, primary_doc, source, ingested_at
                )
                SELECT cik, symbol, accession, filing_date, form_type,
                       items, primary_doc, source, ingested_at
                FROM filings_in
                ON CONFLICT (accession) DO UPDATE SET
                    cik = EXCLUDED.cik,
                    symbol = EXCLUDED.symbol,
                    filing_date = EXCLUDED.filing_date,
                    form_type = EXCLUDED.form_type,
                    items = EXCLUDED.items,
                    primary_doc = EXCLUDED.primary_doc,
                    source = EXCLUDED.source,
                    ingested_at = EXCLUDED.ingested_at
                """
            )
        finally:
            conn.unregister("filings_in")
        return len(out)
    finally:
        if own:
            conn.close()


# --- Top-level backfill ------------------------------------------------------


def backfill_filings(
    universe: str = "SP500",
    *,
    symbols: list[str] | None = None,
    duckdb_path: str | None = None,
) -> dict:
    """Fetch recent filings for every current member of ``universe``.

    Returns the standard {requested, succeeded, failed, rows_written, failures}
    dict. Each successful symbol typically yields 200-1000 filing rows.
    """
    from packages.ingestion.universe.membership import members_on

    if symbols is None:
        members = members_on(universe, datetime.now(UTC).date())
        symbols = members["symbol"].tolist()
    if not symbols:
        log.warning(f"edgar backfill: no symbols for {universe}")
        return {"requested": 0, "succeeded": 0, "failed": 0, "rows_written": 0, "failures": []}

    log.info("edgar backfill: fetching ticker -> CIK map")
    ticker_to_cik = _fetch_company_tickers()
    log.info(f"edgar backfill: SEC ticker map has {len(ticker_to_cik)} symbols")

    log.info(f"edgar backfill: {len(symbols)} {universe} symbols")
    succeeded = failed = rows_written = 0
    failures: list[dict] = []

    with _edgar_conn(duckdb_path) as conn:
        for i, sym in enumerate(symbols):
            try:
                edgar_sym = _to_edgar_ticker(sym)
                cik = ticker_to_cik.get(edgar_sym)
                if cik is None:
                    failed += 1
                    failures.append({"symbol": sym, "error": "no CIK in SEC ticker map"})
                    continue

                if i > 0:
                    time.sleep(_PER_REQUEST_SLEEP_SEC)
                submissions = _fetch_submissions_json(cik)
                if submissions is None:
                    failed += 1
                    failures.append({"symbol": sym, "error": f"submissions 404 for CIK {cik}"})
                    continue

                df = _parse_filings(submissions, sym, cik)
                if df.empty:
                    failed += 1
                    failures.append({"symbol": sym, "error": "no filings parsed"})
                    continue

                n = upsert_filings(df, conn=conn)
                rows_written += n
                succeeded += 1
                if (i + 1) % 50 == 0:
                    log.info(f"  ... {i + 1}/{len(symbols)} done ({rows_written} rows)")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append({"symbol": sym, "error": repr(exc)})
                log.error(f"failed {sym}: {exc!r}")

    log.info(
        f"edgar backfill complete: succeeded={succeeded} failed={failed} "
        f"rows_written={rows_written}"
    )
    return {
        "requested": len(symbols),
        "succeeded": succeeded,
        "failed": failed,
        "rows_written": rows_written,
        "failures": failures,
    }


def load_filings(
    symbols: list[str] | None = None,
    *,
    form_types: list[str] | None = None,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """Return filings as a DataFrame, optionally filtered."""
    with _edgar_conn(duckdb_path) as conn:
        sql = "SELECT * FROM edgar_filings WHERE 1=1"
        params: list = []
        if symbols is not None:
            sql += " AND symbol = ANY(?)"
            params.append(symbols)
        if form_types is not None:
            sql += " AND form_type = ANY(?)"
            params.append(form_types)
        sql += " ORDER BY symbol, filing_date"
        return conn.execute(sql, params).df()


def has_filings_data(*, duckdb_path: str | None = None) -> bool:
    with _edgar_conn(duckdb_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM edgar_filings").fetchone()
        return bool(row and row[0] > 0)
