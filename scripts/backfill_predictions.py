"""Backfill predictions_log over a historical date range.

Usage:

    python -m scripts.backfill_predictions --start 2025-05-11 --end 2026-05-08
    python -m scripts.backfill_predictions --months 12
    python -m scripts.backfill_predictions --months 12 --skip-existing

Runs `predict_with_features` on every NYSE trading day in the range,
re-using a single in-memory model load. Feature build dominates the
per-day cost (~50s/date in this codebase) so we can't easily speed
this up without batching the feature pipeline.

This is intended as a one-shot to populate the predictions log so the
paper-trading stop-loss optimizer has enough history to tune against.
The backfill uses TODAY's model on every historical date — i.e. NOT a
proper walk-forward (which would retrain at each date). That's an
acceptable simplification for stop-loss param search; do not interpret
the resulting backtest as an honest out-of-sample.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from packages.common.config import settings
from packages.common.logging import log
from packages.inference.predict import (
    build_inference_features,
    load_models_for_universe,
    predict_with_features,
)
from packages.inference.tracking import log_predictions
from packages.ingestion.universe.membership import members_on


def _nyse_trading_days(start: date, end: date) -> list[date]:
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    if sched.empty:
        return []
    return [d.date() for d in sched.index.to_pydatetime()]


def _existing_dates(universe: str) -> set[date]:
    """Return the set of as_of dates already in predictions_log for the universe."""
    conn = sqlite3.connect(settings.predictions_sqlite_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT as_of FROM predictions_log WHERE universe = ?",
            (universe,),
        ).fetchall()
    finally:
        conn.close()
    out: set[date] = set()
    for (s,) in rows:
        try:
            out.add(date.fromisoformat(s))
        except (ValueError, TypeError):
            continue
    return out


def backfill(
    universe: str,
    start: date,
    end: date,
    *,
    skip_existing: bool = True,
) -> dict:
    log.info(f"backfill: {universe} {start} -> {end} (skip_existing={skip_existing})")
    models = load_models_for_universe(universe)
    log.info(
        f"backfill: loaded models reg={models.reg_version} cls={models.cls_version}"
    )
    needed = sorted(
        set(models.reg_meta["feature_cols"]) | set(models.cls_meta["feature_cols"])
    )

    days = _nyse_trading_days(start, end)
    log.info(f"backfill: {len(days)} NYSE trading days in range")

    skip_set: set[date] = _existing_dates(universe) if skip_existing else set()
    if skip_existing:
        log.info(f"backfill: {len(skip_set)} dates already logged; will skip")

    results = {
        "total_days": len(days),
        "skipped": 0,
        "predicted": 0,
        "empty": 0,
        "errors": 0,
        "logged": 0,
        "elapsed_s": 0.0,
    }
    t_start = time.monotonic()

    for i, d in enumerate(days, start=1):
        if skip_existing and d in skip_set:
            results["skipped"] += 1
            continue
        t_day = time.monotonic()
        try:
            members = members_on(universe, d)
            if members.empty:
                log.warning(f"  [{i}/{len(days)}] {d}: no universe members; skip")
                results["empty"] += 1
                continue
            feats = build_inference_features(universe, d, feature_cols=needed)
            if feats.empty:
                log.warning(f"  [{i}/{len(days)}] {d}: empty features; skip")
                results["empty"] += 1
                continue
            feats = feats[feats["symbol"].isin(members["symbol"])]
            if feats.empty:
                results["empty"] += 1
                continue
            preds = predict_with_features(feats, models)
            if preds.empty:
                results["empty"] += 1
                continue
            n_logged = log_predictions(preds)
            results["predicted"] += 1
            results["logged"] += int(n_logged)
            elapsed = time.monotonic() - t_day
            log.info(
                f"  [{i}/{len(days)}] {d}: predicted {len(preds)} symbols, "
                f"logged {n_logged} ({elapsed:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001
            log.exception(f"  [{i}/{len(days)}] {d}: FAILED {exc!r}")
            results["errors"] += 1
            continue

    results["elapsed_s"] = time.monotonic() - t_start
    log.info(f"backfill complete: {results}")
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="SP500")
    p.add_argument("--start", type=date.fromisoformat, default=None)
    p.add_argument("--end", type=date.fromisoformat, default=None)
    p.add_argument(
        "--months", type=int, default=None,
        help="Convenience: backfill the last N months ending yesterday.",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip dates that already have rows in predictions_log "
        "(default True). Use --no-skip-existing to force re-predict.",
    )
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=True)
    args = p.parse_args()

    if args.months is not None:
        end = args.end or (date.today() - timedelta(days=1))
        start = end - timedelta(days=args.months * 30)
    else:
        start = args.start
        end = args.end
    if not start or not end:
        p.error("must provide --months OR (--start AND --end)")

    result = backfill(args.universe, start, end, skip_existing=args.skip_existing)
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
