"""Continuous safety-snapshot of the strict-WF predictions database.

Run this as a sidecar process alongside the WF. Every INTERVAL_SEC it
takes an online (read-only-while-writer-runs) snapshot of
``predictions.sqlite`` using SQLite's backup API, stored under a
timestamped name in ``backups/``. A rolling window of MAX_KEEP
snapshots is retained.

Usage:
    python -m scripts.backup_predictions_loop \
        --src data/processed/walkforward_10yr_strict/predictions.sqlite \
        --interval 1800 --keep 24

Defaults match the strict-WF setup: snapshot every 30 minutes, keep
the last 24 (rolling 12-hour window). Survives:
  - WF process restarts (this loop runs independently)
  - Brief DB-lock collisions (sqlite3.backup() handles them gracefully)
  - Power loss (each snapshot is a complete file, atomically committed)
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from packages.common.logging import log


def take_snapshot(src: str, backup_dir: Path) -> Path | None:
    """One snapshot. Uses sqlite3.backup() so it's safe while writer is open."""
    src_path = Path(src)
    if not src_path.exists():
        return None
    if src_path.stat().st_size < 50_000:
        # Empty schema or near-empty — not worth backing up.
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = backup_dir / f"predictions_{ts}.sqlite"
    try:
        # SQLite online backup: copies pages while the writer continues.
        # If the writer is in the middle of a transaction, the backup
        # gets the pre-transaction state — never corrupt, always
        # consistent.
        src_conn = sqlite3.connect(src, timeout=30.0)
        try:
            dst_conn = sqlite3.connect(str(dst))
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()
    except sqlite3.Error as exc:
        log.warning(f"snapshot failed: {exc!r}")
        # If the .sqlite file was partially created, remove it.
        if dst.exists() and dst.stat().st_size < 1000:
            dst.unlink(missing_ok=True)
        return None
    return dst


def prune(backup_dir: Path, keep: int) -> int:
    """Keep the most recent ``keep`` snapshots; delete the rest. Returns
    count pruned."""
    snapshots = sorted(
        backup_dir.glob("predictions_2*.sqlite"),
        key=lambda p: p.name,
    )
    if len(snapshots) <= keep:
        return 0
    to_remove = snapshots[:-keep]
    for f in to_remove:
        f.unlink(missing_ok=True)
    return len(to_remove)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="data/processed/walkforward_10yr_strict/predictions.sqlite",
        help="Path to the predictions sqlite to snapshot.",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=1800,
        help="Seconds between snapshots. Default 1800 (30 min).",
    )
    p.add_argument(
        "--keep",
        type=int,
        default=24,
        help="Number of snapshots to retain (oldest pruned). Default 24.",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Take a single snapshot and exit (for one-shot pre-restart safety).",
    )
    args = p.parse_args()

    src = Path(args.src)
    backup_dir = src.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"backup loop: src={args.src} interval={args.interval}s "
        f"keep={args.keep} backup_dir={backup_dir}"
    )

    while True:
        snap = take_snapshot(args.src, backup_dir)
        if snap is not None:
            size_mb = snap.stat().st_size / 1e6
            pruned = prune(backup_dir, args.keep)
            log.info(
                f"snapshot {snap.name} ({size_mb:.1f} MB) — pruned {pruned} old"
            )
        else:
            log.info("snapshot skipped (src missing or near-empty)")
        if args.once:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
