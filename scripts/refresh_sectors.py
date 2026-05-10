"""Fetch GICS sectors for the current SP500 universe via yfinance and cache.

    python -m scripts.refresh_sectors --universe SP500

Output: ``data/processed/sectors_{universe}.parquet`` with columns
(symbol, sector, industry, fetched_at).

yfinance Ticker.info is the only free source for GICS sector that doesn't
require API keys. It rate-limits aggressively, so this script is meant to be
run rarely (sectors are stable). The downstream feature group reads the
parquet — it does NOT live-fetch.

If a symbol returns no sector (yfinance occasionally fails), it is written
with sector='UNKNOWN' so the feature group can still see it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import click
import pandas as pd
import yfinance as yf

from packages.common.config import PROCESSED_DIR
from packages.common.logging import log
from packages.ingestion.adapters.yfinance_adapter import to_yfinance_symbol
from packages.ingestion.universe.membership import members_on

_PER_SYMBOL_SLEEP_SEC: float = 0.5


def _fetch_one(symbol: str, yf_symbol: str) -> tuple[str, str, str]:
    try:
        info = yf.Ticker(yf_symbol).info or {}
        sector = info.get("sector") or "UNKNOWN"
        industry = info.get("industry") or "UNKNOWN"
        return symbol, sector, industry
    except Exception as exc:  # noqa: BLE001
        log.warning(f"sector fetch failed for {symbol}: {exc!r}")
        return symbol, "UNKNOWN", "UNKNOWN"


@click.command()
@click.option("--universe", default="SP500", show_default=True)
def main(universe: str) -> None:
    members = members_on(universe, datetime.now(UTC).date())
    if members.empty:
        raise click.UsageError(f"no current members for universe={universe}")

    log.info(f"fetching sectors for {len(members)} {universe} symbols")
    rows: list[tuple[str, str, str]] = []
    for i, sym in enumerate(members["symbol"].tolist()):
        if i > 0:
            time.sleep(_PER_SYMBOL_SLEEP_SEC)
        yf_sym = to_yfinance_symbol(sym, universe)
        s, sector, industry = _fetch_one(sym, yf_sym)
        rows.append((s, sector, industry))
        if (i + 1) % 50 == 0:
            log.info(f"  ... {i + 1}/{len(members)} done")

    df = pd.DataFrame(rows, columns=["symbol", "sector", "industry"])
    df["fetched_at"] = datetime.now(UTC)
    out_path = Path(PROCESSED_DIR) / f"sectors_{universe.lower()}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_known = (df["sector"] != "UNKNOWN").sum()
    click.echo(f"Wrote {len(df)} rows ({n_known} with known sector) -> {out_path}")
    click.echo("\nSector distribution:")
    click.echo(df["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
