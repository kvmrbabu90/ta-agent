"""Unified daily-ingest orchestration.

Order of operations:
    1. IB → SP500 daily_update
    2. Kite → NIFTY100 daily_update
    3. yfinance fallback for any symbol that failed in steps 1 or 2

The job is intentionally tolerant of single-symbol failures; only large-scale
problems (very low coverage, many exceptions) cause a non-zero exit.

Exit codes:
    0  — clean enough to call it done
    1  — unexpected exception
    2  — coverage below 90% across the run
    3  — total exception count > 50

Usable both as a CLI (``python -m jobs.daily_ingest``) and as an importable
function for APScheduler.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from typing import Any

import pandas_market_calendars as mcal

from packages.common.logging import log
from packages.ingestion.adapters import (
    ib_adapter,
    kite_adapter,
    yfinance_adapter,
)

# Indirection so tests can patch each step without dragging in real adapters.
ib_daily_update = ib_adapter.daily_update
kite_daily_update = kite_adapter.daily_update
yf_backfill_universe = yfinance_adapter.backfill_universe


_COVERAGE_FAIL_THRESHOLD = 0.90
_EXCEPTION_COUNT_FAIL_THRESHOLD = 50


def _is_any_market_open(as_of: date | None = None) -> bool:
    """True if either NYSE or NSE is a trading day on ``as_of``."""
    when = as_of or date.today()
    for code in ("NYSE", "XNSE"):
        try:
            cal = mcal.get_calendar(code)
        except Exception as exc:  # noqa: BLE001 — calendar code unknown
            log.warning(f"market calendar {code} unavailable: {exc!r}")
            continue
        sched = cal.schedule(start_date=when.isoformat(), end_date=when.isoformat())
        if not sched.empty:
            return True
    return False


def _failed_symbols(result: dict) -> list[str]:
    return [f["symbol"] for f in result.get("failures", [])]


def _summarize(per_source: dict[str, dict]) -> dict:
    requested = sum(r.get("requested", 0) for r in per_source.values())
    succeeded = sum(r.get("succeeded", 0) for r in per_source.values())
    failed = sum(r.get("failed", 0) for r in per_source.values())
    rows_written = sum(r.get("rows_written", 0) for r in per_source.values())
    coverage = (succeeded / requested) if requested else 1.0
    return {
        "requested": requested,
        "succeeded": succeeded,
        "failed": failed,
        "rows_written": rows_written,
        "coverage": coverage,
        "per_source": per_source,
    }


def run(*, today: date | None = None, skip_market_check: bool = False) -> dict[str, Any]:
    """Execute the full ingest run. Returns the summary dict."""
    today = today or date.today()

    if not skip_market_check and not _is_any_market_open(today):
        log.info(f"daily_ingest: {today} is not a trading day for NYSE or NSE; nothing to do")
        return {"skipped": True, "reason": "non-trading-day", "today": today.isoformat()}

    per_source: dict[str, dict] = {}

    log.info("daily_ingest: starting IB / SP500 update")
    try:
        per_source["ib_sp500"] = ib_daily_update("SP500")
    except Exception as exc:  # noqa: BLE001
        log.error(f"ib_daily_update raised: {exc!r}")
        per_source["ib_sp500"] = {
            "requested": 0, "succeeded": 0, "failed": 1, "rows_written": 0,
            "failures": [{"symbol": "<all>", "error": repr(exc)}],
        }

    log.info("daily_ingest: starting Kite / NIFTY100 update")
    try:
        per_source["kite_nifty100"] = kite_daily_update("NIFTY100")
    except Exception as exc:  # noqa: BLE001
        log.error(f"kite_daily_update raised: {exc!r}")
        per_source["kite_nifty100"] = {
            "requested": 0, "succeeded": 0, "failed": 1, "rows_written": 0,
            "failures": [{"symbol": "<all>", "error": repr(exc)}],
        }

    # yfinance fallback for symbols that the primary source missed.
    fallback_window_start = today - timedelta(days=10)
    for primary_key, universe in (("ib_sp500", "SP500"), ("kite_nifty100", "NIFTY100")):
        retry_symbols = _failed_symbols(per_source.get(primary_key, {}))
        if not retry_symbols:
            continue
        log.info(
            f"daily_ingest: yfinance fallback for {len(retry_symbols)} symbols "
            f"in {universe}"
        )
        try:
            per_source[f"yf_{universe.lower()}"] = yf_backfill_universe(
                universe,
                fallback_window_start,
                today,
                symbols=retry_symbols,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(f"yfinance fallback for {universe} raised: {exc!r}")
            per_source[f"yf_{universe.lower()}"] = {
                "requested": len(retry_symbols), "succeeded": 0,
                "failed": len(retry_symbols), "rows_written": 0,
                "failures": [{"symbol": "<all>", "error": repr(exc)}],
            }

    summary = _summarize(per_source)
    log.info(
        f"daily_ingest: requested={summary['requested']} succeeded={summary['succeeded']} "
        f"failed={summary['failed']} rows_written={summary['rows_written']} "
        f"coverage={summary['coverage']:.1%}"
    )
    return summary


def determine_exit_code(summary: dict[str, Any]) -> int:
    if summary.get("skipped"):
        return 0
    if summary["failed"] > _EXCEPTION_COUNT_FAIL_THRESHOLD:
        return 3
    if summary["requested"] > 0 and summary["coverage"] < _COVERAGE_FAIL_THRESHOLD:
        return 2
    return 0


def main() -> int:
    try:
        summary = run()
        return determine_exit_code(summary)
    except Exception as exc:  # noqa: BLE001
        log.exception(f"daily_ingest crashed: {exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
