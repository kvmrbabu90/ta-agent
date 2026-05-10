"""SEC EDGAR fundamentals adapter — pulls quarterly/annual XBRL facts.

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

Free, official, no API key required. Each request returns the FULL XBRL
fact history for one company in a single ~3-5 MB JSON. Standard GAAP
concepts (us-gaap namespace) — same definitions across all SEC filers.

Concepts we extract (with fallback chains for cross-era compatibility):
    NetIncomeLoss
    StockholdersEquity
    Assets
    LongTermDebt -> LongTermDebtNoncurrent
    GrossProfit
    Revenues -> RevenueFromContractWithCustomerExcludingAssessedTax
                (the latter superseded the former around 2018 for most filers)

Schema (sec_fundamentals table, DuckDB):
    cik           VARCHAR  NOT NULL
    symbol        VARCHAR  NOT NULL
    concept       VARCHAR  NOT NULL    -- canonical concept name (we normalize fallback chains)
    period_end    DATE     NOT NULL    -- fiscal period end
    fp            VARCHAR  NOT NULL    -- 'FY', 'Q1', 'Q2', 'Q3', 'Q4'
    fy            INTEGER              -- fiscal year (per the filer's calendar)
    value         DOUBLE   NOT NULL
    form          VARCHAR  NOT NULL    -- '10-K' or '10-Q'
    accession     VARCHAR  NOT NULL
    filed_date    DATE     NOT NULL    -- WHEN the data became public (CRITICAL for PIT correctness)
    source        VARCHAR  NOT NULL
    ingested_at   TIMESTAMP NOT NULL
    PRIMARY KEY (cik, concept, period_end, fp, accession)

Causality contract: every downstream feature MUST filter by
``filed_date <= bar_date`` to avoid look-ahead bias. The period_end
itself is NOT enough — companies file weeks/months AFTER the period
ends, and the data wasn't public until then.
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
from packages.ingestion.sec_filings import _fetch_company_tickers, _to_edgar_ticker
from packages.ingestion.storage import _connect

_FUNDAMENTALS_DDL = """
CREATE TABLE IF NOT EXISTS sec_fundamentals (
    cik           VARCHAR   NOT NULL,
    symbol        VARCHAR   NOT NULL,
    concept       VARCHAR   NOT NULL,
    period_end    DATE      NOT NULL,
    fp            VARCHAR   NOT NULL,
    fy            INTEGER,
    value         DOUBLE    NOT NULL,
    form          VARCHAR   NOT NULL,
    accession     VARCHAR   NOT NULL,
    filed_date    DATE      NOT NULL,
    source        VARCHAR   NOT NULL,
    ingested_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (cik, concept, period_end, fp, accession)
);
CREATE INDEX IF NOT EXISTS sec_fund_symbol_concept
    ON sec_fundamentals (symbol, concept, filed_date);
"""

_SOURCE = "edgar"
_USER_AGENT = "ta-agent local research konda@example.com"
_PER_REQUEST_SLEEP_SEC = 0.11  # SEC rate limit ~10 req/sec; we sleep 110ms

# Canonical concept name -> ordered fallback list of XBRL concept names.
# When extracting, we use the first fallback that has data.
_CONCEPT_FALLBACKS: dict[str, list[str]] = {
    "NetIncomeLoss":      ["NetIncomeLoss"],
    "StockholdersEquity": ["StockholdersEquity"],
    "Assets":             ["Assets"],
    "LongTermDebt":       ["LongTermDebt", "LongTermDebtNoncurrent"],
    "GrossProfit":        ["GrossProfit"],
    "Revenues":           ["Revenues",
                           "RevenueFromContractWithCustomerExcludingAssessedTax",
                           "SalesRevenueNet"],
}


@contextmanager
def _fund_conn(path: str | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = _connect(path)
    try:
        conn.execute(_FUNDAMENTALS_DDL)
        yield conn
    finally:
        conn.close()


def _fetch_companyfacts(cik: str) -> dict | None:
    """Fetch the full XBRL fact history for one CIK. None on 404."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _extract_concept(
    facts: dict, canonical: str, *, fallbacks: list[str]
) -> list[dict]:
    """Return all USD observations for ``canonical`` concept across the
    fallback list, deduplicating by (period_end, fp).

    For companies that switched concepts (e.g. 'Revenues' -> 'RevenueFromContract...'
    around 2018), this surfaces the union of observations. Within a (period_end,
    fp) collision, the LATER-listed fallback wins — so the canonical fallback
    list should be ordered older-name-first so newer reporting wins on overlap.
    """
    us_gaap = facts.get("us-gaap", {})
    rows: list[dict] = []
    seen: dict[tuple, int] = {}  # (period_end, fp) -> index in rows
    for name in fallbacks:
        concept_data = us_gaap.get(name)
        if concept_data is None:
            continue
        usd_obs = concept_data.get("units", {}).get("USD", [])
        for obs in usd_obs:
            val = obs.get("val")
            end = obs.get("end")
            filed = obs.get("filed")
            if val is None or not end or not filed:
                continue
            # `fp` and `form` can be present-but-None in the JSON, not just
            # missing. dict.get(k, default) only fills default when MISSING,
            # so we coerce None->'?' explicitly to satisfy the NOT NULL DDL.
            fp = obs.get("fp") or "?"
            form = obs.get("form") or "?"
            accn = obs.get("accn") or "?"
            key = (end, fp)
            new_row = {
                "concept": canonical,
                "period_end": end,
                "fp": fp,
                "fy": obs.get("fy"),
                "value": val,
                "form": form,
                "accession": accn,
                "filed_date": filed,
            }
            if key in seen:
                # Newer fallback overwrites earlier one for the same period.
                rows[seen[key]] = new_row
            else:
                seen[key] = len(rows)
                rows.append(new_row)
    return rows


def _parse_companyfacts(
    facts: dict, symbol: str, cik: str
) -> pd.DataFrame:
    """Extract all targeted concepts from one company's facts JSON."""
    if not facts:
        return pd.DataFrame()
    fact_data = facts.get("facts", {})
    rows: list[dict] = []
    for canonical, fallbacks in _CONCEPT_FALLBACKS.items():
        for r in _extract_concept(fact_data, canonical, fallbacks=fallbacks):
            try:
                period_end = date.fromisoformat(r["period_end"])
                filed_date = date.fromisoformat(r["filed_date"])
            except ValueError:
                continue
            rows.append({
                "cik": cik,
                "symbol": symbol,
                "concept": r["concept"],
                "period_end": period_end,
                "fp": r["fp"],
                "fy": r["fy"],
                "value": float(r["value"]),
                "form": r["form"],
                "accession": r["accession"],
                "filed_date": filed_date,
            })
    return pd.DataFrame(rows)


def upsert_fundamentals(
    df: pd.DataFrame, conn: duckdb.DuckDBPyConnection | None = None
) -> int:
    if df.empty:
        return 0
    own = conn is None
    if own:
        conn = _connect()
        conn.execute(_FUNDAMENTALS_DDL)
    try:
        now_utc = datetime.now(UTC)
        out = df.copy()
        if "source" not in out.columns:
            out["source"] = _SOURCE
        if "ingested_at" not in out.columns:
            out["ingested_at"] = now_utc

        conn.register("fund_in", out)
        try:
            conn.execute(
                """
                INSERT INTO sec_fundamentals (
                    cik, symbol, concept, period_end, fp, fy, value,
                    form, accession, filed_date, source, ingested_at
                )
                SELECT cik, symbol, concept, period_end, fp, fy, value,
                       form, accession, filed_date, source, ingested_at
                FROM fund_in
                ON CONFLICT (cik, concept, period_end, fp, accession) DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    fy = EXCLUDED.fy,
                    value = EXCLUDED.value,
                    form = EXCLUDED.form,
                    filed_date = EXCLUDED.filed_date,
                    source = EXCLUDED.source,
                    ingested_at = EXCLUDED.ingested_at
                """
            )
        finally:
            conn.unregister("fund_in")
        return len(out)
    finally:
        if own:
            conn.close()


def backfill_fundamentals(
    universe: str = "SP500",
    *,
    symbols: list[str] | None = None,
    duckdb_path: str | None = None,
) -> dict:
    """Pull companyfacts for every current member of ``universe``.

    Memory-conscious implementation: each XBRL JSON is 3-5 MB, and parsing
    yields 200-700 row DataFrames. Holding all of these alongside an open
    DuckDB write connection blew up Windows process memory on the SP500
    backfill. We close+reopen the DuckDB connection every batch_size
    symbols, drop large objects between iterations, and explicit-gc.
    """
    import gc

    from packages.ingestion.universe.membership import members_on

    if symbols is None:
        members = members_on(universe, datetime.now(UTC).date())
        symbols = members["symbol"].tolist()
    if not symbols:
        log.warning(f"sec_fundamentals: no symbols for {universe}")
        return {"requested": 0, "succeeded": 0, "failed": 0, "rows_written": 0, "failures": []}

    log.info("sec_fundamentals: fetching ticker -> CIK map")
    ticker_to_cik = _fetch_company_tickers()
    log.info(f"sec_fundamentals: ticker map has {len(ticker_to_cik)} entries")

    log.info(f"sec_fundamentals backfill: {len(symbols)} {universe} symbols")
    succeeded = failed = rows_written = 0
    failures: list[dict] = []

    batch_size = 25  # close+reopen the duckdb connection every N symbols

    def _process_one(sym: str, conn: duckdb.DuckDBPyConnection) -> tuple[int, str | None]:
        edgar_sym = _to_edgar_ticker(sym)
        cik = ticker_to_cik.get(edgar_sym)
        if cik is None:
            return 0, "no CIK"
        facts = _fetch_companyfacts(cik)
        if facts is None:
            return 0, f"companyfacts 404 (CIK {cik})"
        df = _parse_companyfacts(facts, sym, cik)
        # Drop the 4 MB JSON dict before the upsert to reduce peak memory
        del facts
        if df.empty:
            return 0, "no concepts parsed"
        n = upsert_fundamentals(df, conn=conn)
        del df
        gc.collect()
        return n, None

    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start:batch_start + batch_size]
        with _fund_conn(duckdb_path) as conn:
            for i, sym in enumerate(batch):
                try:
                    if batch_start + i > 0:
                        time.sleep(_PER_REQUEST_SLEEP_SEC)
                    n, err = _process_one(sym, conn)
                    if err is not None:
                        failed += 1
                        failures.append({"symbol": sym, "error": err})
                    else:
                        succeeded += 1
                        rows_written += n
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    failures.append({"symbol": sym, "error": repr(exc)})
                    log.error(f"failed {sym}: {exc!r}")
        log.info(
            f"  batch {batch_start//batch_size + 1}/"
            f"{(len(symbols) + batch_size - 1)//batch_size}: "
            f"{succeeded} ok, {failed} failed, {rows_written} rows total"
        )
        gc.collect()

    log.info(
        f"sec_fundamentals backfill complete: succeeded={succeeded} "
        f"failed={failed} rows_written={rows_written}"
    )
    return {
        "requested": len(symbols),
        "succeeded": succeeded,
        "failed": failed,
        "rows_written": rows_written,
        "failures": failures,
    }


def load_fundamentals(
    symbols: list[str] | None = None,
    *,
    concepts: list[str] | None = None,
    fp: list[str] | None = None,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    with _fund_conn(duckdb_path) as conn:
        sql = "SELECT * FROM sec_fundamentals WHERE 1=1"
        params: list = []
        if symbols is not None:
            sql += " AND symbol = ANY(?)"
            params.append(symbols)
        if concepts is not None:
            sql += " AND concept = ANY(?)"
            params.append(concepts)
        if fp is not None:
            sql += " AND fp = ANY(?)"
            params.append(fp)
        sql += " ORDER BY symbol, concept, filed_date"
        return conn.execute(sql, params).df()


def has_fundamentals_data(*, duckdb_path: str | None = None) -> bool:
    with _fund_conn(duckdb_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM sec_fundamentals").fetchone()
        return bool(row and row[0] > 0)
