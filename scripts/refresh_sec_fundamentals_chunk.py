"""Chunked variant of refresh_sec_fundamentals — processes a slice of the
universe per invocation. Exists because the full backfill in a single
process appears to be killed by something in this environment after a
handful of iterations. Calling this in a loop from bash side-steps the
issue.

    python -m scripts.refresh_sec_fundamentals_chunk --offset 0 --limit 50
"""

from __future__ import annotations

from datetime import UTC, datetime

import click

from packages.common.logging import log
from packages.ingestion.sec_fundamentals import backfill_fundamentals
from packages.ingestion.universe.membership import members_on


@click.command()
@click.option("--universe", default="SP500", show_default=True)
@click.option("--offset", default=0, show_default=True, type=int)
@click.option("--limit", default=50, show_default=True, type=int)
@click.option("--skip-done", is_flag=True, default=True,
              help="Skip symbols already in sec_fundamentals.")
def main(universe: str, offset: int, limit: int, skip_done: bool) -> None:
    members = members_on(universe, datetime.now(UTC).date())
    all_syms = members["symbol"].tolist()
    log.info(f"universe has {len(all_syms)} symbols")

    if skip_done:
        import duckdb

        from packages.common.config import settings
        conn = duckdb.connect(settings.duckdb_path, read_only=True)
        try:
            done = {r[0] for r in conn.execute("SELECT DISTINCT symbol FROM sec_fundamentals").fetchall()}
        except duckdb.CatalogException:
            done = set()
        conn.close()
        todo = [s for s in all_syms if s not in done]
        log.info(f"{len(done)} already done, {len(todo)} remaining")
    else:
        todo = all_syms

    chunk = todo[offset:offset + limit]
    if not chunk:
        click.echo(f"empty chunk at offset {offset}; nothing to do")
        return

    log.info(f"processing chunk: offset={offset} limit={limit} actual={len(chunk)} symbols")
    log.info(f"first 3: {chunk[:3]}, last 3: {chunk[-3:]}")
    result = backfill_fundamentals(universe=universe, symbols=chunk)
    click.echo(
        f"chunk result: requested={result['requested']} ok={result['succeeded']} "
        f"failed={result['failed']} rows={result['rows_written']}"
    )


if __name__ == "__main__":
    main()
