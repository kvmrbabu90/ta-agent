"""Unified interface for index membership queries across universes."""

from __future__ import annotations

from datetime import date

import pandas as pd

from packages.common.logging import log
from packages.ingestion.storage import query_membership_at, upsert_membership
from packages.ingestion.universe.nifty100_history import build_nifty100_membership
from packages.ingestion.universe.nifty100_pit import build_nifty100_pit_membership
from packages.ingestion.universe.sp500_history import build_sp500_membership


def refresh_all_universes(*, pit_india: bool = False) -> dict[str, int]:
    """Rebuild and persist membership for all supported universes.

    Args:
        pit_india: When True, NIFTY 100 is reconstructed from
            ``configs/universes/nifty100_changes.yaml`` + today's snapshot.
            When False (default), NIFTY 100 is the current snapshot only
            (Phase A — survivorship-biased on pre-rebalance dates).

    Returns a dict of universe -> rows written.
    """
    counts: dict[str, int] = {}

    log.info("Refreshing S&P 500 membership")
    sp = build_sp500_membership()
    counts["SP500"] = upsert_membership(sp)

    if pit_india:
        log.info("Refreshing NIFTY 100 membership (PIT reconstruction)")
        ni = build_nifty100_pit_membership()
    else:
        log.info("Refreshing NIFTY 100 membership (current snapshot only)")
        ni = build_nifty100_membership()
    counts["NIFTY100"] = upsert_membership(ni)

    log.info(f"Universe refresh complete: {counts}")
    return counts


def members_on(universe: str, as_of: str | date) -> pd.DataFrame:
    """List of symbols in `universe` on date `as_of` (point-in-time)."""
    return query_membership_at(universe, str(as_of))


def all_symbols_ever(universe: str) -> list[str]:
    """All symbols that have ever been in this universe (for batch ingestion)."""
    from packages.ingestion.storage import get_conn
    with get_conn(read_only=True) as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM index_membership WHERE universe = ? ORDER BY symbol",
            [universe],
        ).fetchall()
    return [r[0] for r in rows]
