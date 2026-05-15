"""Ingest GICS sector + industry metadata for universe members.

Source: yfinance `Ticker.info["sector"]` / `Ticker.info["industry"]`.
yfinance returns the GICS Sector (one of 11 canonical sectors) and the
GICS Sub-Industry. This is the same taxonomy S&P uses; SP500 sector
weights you see anywhere on the web are based on these.

We persist to DuckDB at `symbol_sectors`. The mapping is essentially
static — a company's sector changes maybe once a decade — so we don't
need to re-fetch on every pipeline run. Refresh quarterly is plenty.

Schema:
    symbol_sectors (
        symbol      VARCHAR  PRIMARY KEY,
        sector      VARCHAR,             -- GICS Sector ('Technology', 'Financials', ...)
        industry    VARCHAR,             -- GICS Sub-Industry ('Semiconductors', ...)
        source      VARCHAR  NOT NULL,   -- 'yfinance'
        fetched_at  TIMESTAMP NOT NULL
    )

The 11 GICS sectors (as of 2018 reorg + Real Estate split):
    Communication Services, Consumer Cyclical, Consumer Defensive,
    Energy, Financial Services, Healthcare, Industrials,
    Basic Materials, Real Estate, Technology, Utilities

Note: yfinance uses some legacy names ('Financial Services' not
'Financials'; 'Consumer Cyclical' not 'Consumer Discretionary').
We canonicalize at ingest so the feature column set is stable.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import duckdb
import pandas as pd

from packages.common.logging import log
from packages.ingestion.storage import _connect

_DDL = """
CREATE TABLE IF NOT EXISTS symbol_sectors (
    symbol      VARCHAR PRIMARY KEY,
    sector      VARCHAR,
    industry    VARCHAR,
    source      VARCHAR   NOT NULL,
    fetched_at  TIMESTAMP NOT NULL
);
"""

# Canonical GICS sector names. We normalize yfinance's labels into these
# so the one-hot feature columns are stable across pipeline runs.
CANONICAL_SECTORS = [
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Basic Materials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Unknown",
]

# yfinance's labels that don't quite match the canonical set above.
_YF_TO_CANONICAL = {
    "Financial Services": "Financial Services",
    "Financials": "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials",
    "Information Technology": "Technology",
    "Communications Services": "Communication Services",
}


def _canonicalize(sector: str | None) -> str:
    if not sector:
        return "Unknown"
    s = sector.strip()
    return _YF_TO_CANONICAL.get(s, s)


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(_DDL)


def has_sector_data(*, duckdb_path: str | None = None) -> bool:
    """Cheap probe used by the FeatureExtension to gate activation."""
    try:
        conn = _connect(duckdb_path, read_only=True)
    except Exception:  # noqa: BLE001
        return False
    try:
        # Fast path — just check existence + non-empty.
        rows = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'symbol_sectors'"
        ).fetchone()
        if not rows or rows[0] == 0:
            return False
        n = conn.execute("SELECT COUNT(*) FROM symbol_sectors").fetchone()[0]
        return int(n) > 0
    except Exception:  # noqa: BLE001
        return False
    finally:
        conn.close()


def load_sector_map(
    symbols: list[str] | None = None,
    *,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns [symbol, sector, industry] for the
    requested symbols (or all symbols if None). Sector values are canonicalized.
    """
    if not has_sector_data(duckdb_path=duckdb_path):
        return pd.DataFrame(columns=["symbol", "sector", "industry"])
    conn = _connect(duckdb_path, read_only=True)
    try:
        if symbols:
            df = conn.execute(
                "SELECT symbol, sector, industry FROM symbol_sectors WHERE symbol = ANY(?)",
                [symbols],
            ).df()
        else:
            df = conn.execute(
                "SELECT symbol, sector, industry FROM symbol_sectors"
            ).df()
    finally:
        conn.close()
    return df


def fetch_sectors_for_symbols(
    symbols: list[str],
    *,
    duckdb_path: str | None = None,
    sleep_per_request_s: float = 0.15,
) -> dict:
    """Pull sector + industry from yfinance for each symbol; upsert into DuckDB.

    yfinance's Ticker.info is rate-limited and slow (~0.5-1.5s per call).
    For SP500 (~503 symbols) expect ~3-10 minutes.

    Returns a summary dict with counts.
    """
    import yfinance as yf

    if not symbols:
        return {"requested": 0, "succeeded": 0, "failed": 0, "rows_written": 0}

    log.info(f"sector_metadata: fetching {len(symbols)} symbols from yfinance")
    rows: list[dict] = []
    failures = 0
    for i, sym in enumerate(symbols):
        try:
            info = yf.Ticker(sym).info or {}
            sector = _canonicalize(info.get("sector"))
            industry = (info.get("industry") or "Unknown").strip() or "Unknown"
            rows.append({"symbol": sym, "sector": sector, "industry": industry})
        except Exception as exc:  # noqa: BLE001
            log.warning(f"sector_metadata: {sym} failed: {exc!r}")
            failures += 1
            rows.append({"symbol": sym, "sector": "Unknown", "industry": "Unknown"})
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{len(symbols)} fetched ({failures} failures so far)")
        time.sleep(sleep_per_request_s)

    df = pd.DataFrame(rows)
    df["source"] = "yfinance"
    df["fetched_at"] = datetime.now(UTC)
    n = upsert_sectors(df, duckdb_path=duckdb_path)
    log.info(
        f"sector_metadata: done. requested={len(symbols)} "
        f"failed={failures} rows_written={n}"
    )
    return {
        "requested": len(symbols),
        "succeeded": len(symbols) - failures,
        "failed": failures,
        "rows_written": n,
    }


def upsert_sectors(
    df: pd.DataFrame, *, duckdb_path: str | None = None,
) -> int:
    if df.empty:
        return 0
    conn = _connect(duckdb_path)
    try:
        _ensure_schema(conn)
        conn.register("incoming_sectors", df)
        try:
            conn.execute(
                """
                INSERT INTO symbol_sectors (symbol, sector, industry, source, fetched_at)
                SELECT symbol, sector, industry, source, fetched_at FROM incoming_sectors
                ON CONFLICT (symbol) DO UPDATE SET
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    source = EXCLUDED.source,
                    fetched_at = EXCLUDED.fetched_at
                """
            )
        finally:
            conn.unregister("incoming_sectors")
        return len(df)
    finally:
        conn.close()


__all__ = [
    "CANONICAL_SECTORS",
    "fetch_sectors_for_symbols",
    "has_sector_data",
    "load_sector_map",
    "upsert_sectors",
]
