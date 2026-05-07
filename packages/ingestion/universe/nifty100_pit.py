"""Point-in-time NIFTY 100 reconstruction from a hand-curated change log.

NSE doesn't publish a machine-readable change history at the free tier
(unlike Wikipedia for SP 500). This module reconstructs PIT membership
from two inputs:

    1. ``configs/universes/nifty100_changes.yaml``  — manually transcribed
       reconstitution events (date, action, symbol).
    2. The current-day snapshot from niftyindices.com (Phase A loader).

Algorithm — walk BACKWARDS through history:

    state = today's members (snapshot from niftyindices.com)
    for event in events sorted newest -> oldest:
        if event is "add X on D":
            # Before D, X was NOT a member. Set X.start_date = D.
            # Remove X from the rolling state so older events don't see it.
        if event is "remove X on D":
            # Before D, X WAS a member. Open a historical interval ending at D.
            # Add X to the rolling state.
    For symbols still in the state at the end (no add event found), set
    start_date = RELIABLE_START — they were already members at the
    beginning of our reconstructable window.

Limitations:
    - Output quality is bounded by how complete this YAML is. With an
      empty events list, this loader is identical to Phase A (current
      members only, all start_date = RELIABLE_START).
    - Same symbol with multiple non-contiguous tenancies (added,
      removed, re-added) IS supported — each tenancy becomes a row.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from packages.common.config import CONFIGS_DIR
from packages.common.logging import log
from packages.ingestion.universe.nifty100_history import (
    RELIABLE_START,
    _fetch_current_csv,
)

_DEFAULT_CHANGES_PATH = CONFIGS_DIR / "universes" / "nifty100_changes.yaml"


def _load_changes(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and lightly validate the YAML change log."""
    p = Path(path or _DEFAULT_CHANGES_PATH)
    if not p.exists():
        log.warning(f"nifty100 changes file not found at {p}; PIT degrades to current snapshot")
        return []
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    rows = raw.get("changes") or []
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        try:
            ev_date = r["effective_date"]
            if isinstance(ev_date, str):
                ev_date = date.fromisoformat(ev_date)
            elif not isinstance(ev_date, date):
                raise ValueError(f"effective_date must be a date, got {type(ev_date).__name__}")
            action = str(r["action"]).lower()
            if action not in ("add", "remove"):
                raise ValueError(f"action must be 'add' or 'remove', got {action!r}")
            symbol = str(r["symbol"]).strip().upper()
            if not symbol:
                raise ValueError("symbol is empty")
        except Exception as exc:  # noqa: BLE001 — surface bad rows but keep going
            log.warning(f"nifty100 changes row {i} skipped: {exc!r} — entry: {r!r}")
            continue
        out.append(
            {
                "effective_date": ev_date,
                "action": action,
                "symbol": symbol,
                "reason": r.get("reason"),
                "source": r.get("source"),
            }
        )
    return out


def _validate_and_sort(events: list[dict[str, Any]], *, today: date) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for e in events:
        if e["effective_date"] > today:
            log.warning(
                f"nifty100 changes: future-dated event ignored: "
                f"{e['effective_date']} {e['action']} {e['symbol']}"
            )
            continue
        cleaned.append(e)
    cleaned.sort(key=lambda e: (e["effective_date"], e["action"]))
    return cleaned


def _reconstruct_intervals(
    current_symbols: dict[str, str],  # symbol -> company_name from snapshot
    events: list[dict[str, Any]],
    *,
    today: date,
) -> pd.DataFrame:
    """Walk events newest -> oldest and build (symbol, start, end) rows.

    Returns DataFrame with columns:
        universe, symbol, exchange, start_date, end_date, company_name
    """
    intervals: list[dict[str, Any]] = []

    # state: open "tenancies" — symbols that ARE members at the current
    # rewind point. Each entry tracks the (start_date, end_date_or_None)
    # of its CURRENT tenancy as we walk backwards.
    state: dict[str, dict[str, Any]] = {}
    for sym, name in current_symbols.items():
        state[sym] = {
            "start_date": None,  # filled later when we find the add event
            "end_date": None,    # currently a member as of today
            "company_name": name,
        }

    # Walk events newest -> oldest.
    for ev in sorted(events, key=lambda e: e["effective_date"], reverse=True):
        sym = ev["symbol"]
        action = ev["action"]
        ev_date = ev["effective_date"]

        if action == "add":
            # X joined on ev_date. Before ev_date, X was NOT a member.
            if sym not in state:
                log.warning(
                    f"nifty100 PIT: 'add {sym} {ev_date}' but {sym} not in rolling "
                    f"state — likely missing matching 'remove' event. Skipping."
                )
                continue
            # Close the current tenancy: it began on ev_date.
            tenancy = state.pop(sym)
            tenancy["start_date"] = ev_date
            intervals.append({"symbol": sym, **tenancy})

        elif action == "remove":
            # X left on ev_date. Before ev_date, X WAS a member.
            if sym in state:
                log.warning(
                    f"nifty100 PIT: 'remove {sym} {ev_date}' but {sym} is already "
                    f"in rolling state (currently a member). Likely a duplicate "
                    f"or out-of-order event. Skipping."
                )
                continue
            # Open a historical tenancy ending at ev_date. Look up the
            # company name from current_symbols if available, else best-effort.
            state[sym] = {
                "start_date": None,
                "end_date": ev_date,
                "company_name": current_symbols.get(sym),
            }

    # Anything still in `state` predates the earliest event we have.
    # It was a member at RELIABLE_START.
    for sym, tenancy in state.items():
        tenancy["start_date"] = RELIABLE_START
        intervals.append({"symbol": sym, **tenancy})

    if not intervals:
        return pd.DataFrame(
            columns=["universe", "symbol", "exchange", "start_date", "end_date", "company_name"]
        )

    df = pd.DataFrame(intervals)
    df["universe"] = "NIFTY100"
    df["exchange"] = "NSE"
    return df[["universe", "symbol", "exchange", "start_date", "end_date", "company_name"]]


def build_nifty100_pit_membership(
    *,
    changes_path: Path | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Build the PIT membership table for NIFTY 100.

    Combines today's current snapshot (Phase A loader) with the
    hand-curated change log to reconstruct historical intervals.
    """
    today = today or date.today()
    snap = _fetch_current_csv()
    if snap.empty:
        log.warning("nifty100 PIT: empty current snapshot — returning empty membership")
        return pd.DataFrame(
            columns=["universe", "symbol", "exchange", "start_date", "end_date", "company_name"]
        )
    current_symbols = dict(zip(snap["symbol"], snap["company_name"], strict=False))

    raw_events = _load_changes(changes_path)
    events = _validate_and_sort(raw_events, today=today)
    log.info(
        f"nifty100 PIT: {len(current_symbols)} current members, "
        f"{len(events)} change events"
    )
    return _reconstruct_intervals(current_symbols, events, today=today)
