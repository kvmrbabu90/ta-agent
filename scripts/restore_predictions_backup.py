"""Restore the strict-WF predictions database from the most recent
safety snapshot.

Use this if predictions.sqlite gets corrupted, accidentally deleted,
or wiped by some other process. Picks the newest file in ``backups/``
that's >= MIN_SIZE bytes, validates it's a real SQLite database (can
open + count rows), then copies it over the current file. Atomic: the
new file is staged first, then renamed into place.

Usage:
    # Show what backups are available, then prompt before restoring:
    python -m scripts.restore_predictions_backup

    # Restore from a specific snapshot without prompting:
    python -m scripts.restore_predictions_backup --from backups/predictions_20260524_163800.sqlite

    # Inspect snapshots without restoring anything:
    python -m scripts.restore_predictions_backup --list
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from packages.common.logging import log

MIN_VALID_SIZE = 100_000  # 100 KB — anything smaller is just schema


def validate_snapshot(path: Path) -> tuple[bool, str]:
    """Returns (is_valid, summary_string). A valid snapshot is one we
    can open as SQLite and that has at least one row in predictions_log."""
    if not path.exists():
        return False, "file missing"
    if path.stat().st_size < MIN_VALID_SIZE:
        return False, f"too small ({path.stat().st_size} bytes)"
    try:
        c = sqlite3.connect(str(path))
        try:
            row = c.execute(
                "SELECT COUNT(*), MIN(as_of), MAX(as_of), "
                "COUNT(DISTINCT strftime('%Y-%m', as_of)) "
                "FROM predictions_log"
            ).fetchone()
            n, first, last, months = row
        finally:
            c.close()
    except sqlite3.Error as exc:
        return False, f"not a valid sqlite: {exc!r}"
    if n == 0:
        return False, "empty predictions_log"
    return True, f"{n:,} rows · {months} retrains · {first} → {last}"


def list_snapshots(backup_dir: Path) -> list[Path]:
    return sorted(
        list(backup_dir.glob("predictions_2*.sqlite"))
        + list(backup_dir.glob("predictions_pre_startup_*.sqlite")),
        key=lambda p: p.name,
        reverse=True,  # newest first
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        default="data/processed/walkforward_10yr_strict/predictions.sqlite",
        help="The predictions.sqlite to restore (default: strict-WF SP500).",
    )
    p.add_argument(
        "--from",
        dest="from_path",
        default=None,
        help="Specific snapshot path to restore from. Defaults to the newest valid one.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available snapshots and exit without restoring.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Don't prompt for confirmation before overwriting the target.",
    )
    args = p.parse_args()

    target = Path(args.target)
    backup_dir = target.parent / "backups"

    if not backup_dir.exists():
        log.error(f"no backup directory at {backup_dir}")
        return 1

    snapshots = list_snapshots(backup_dir)
    if not snapshots:
        log.error(f"no snapshots in {backup_dir}")
        return 1

    print(f"Found {len(snapshots)} snapshot(s) in {backup_dir}:")
    print(f"  {'name':<55}{'size':>10}  status")
    for snap in snapshots:
        ok, summary = validate_snapshot(snap)
        size_mb = snap.stat().st_size / 1e6
        marker = "✓" if ok else "✗"
        print(f"  {snap.name:<55}{size_mb:>8.1f}MB  {marker} {summary}")

    if args.list:
        return 0

    # Pick a snapshot.
    if args.from_path:
        chosen = Path(args.from_path)
        if not chosen.is_absolute():
            chosen = backup_dir / chosen.name
    else:
        # Newest valid one.
        chosen = next((s for s in snapshots if validate_snapshot(s)[0]), None)
        if chosen is None:
            log.error("no VALID snapshot found (all empty or corrupt)")
            return 1

    ok, summary = validate_snapshot(chosen)
    if not ok:
        log.error(f"chosen snapshot is invalid: {summary}")
        return 1

    print()
    print(f"Plan: restore {chosen.name} → {target}")
    print(f"      ({summary})")
    if target.exists():
        cur_size = target.stat().st_size
        cur_ok, cur_summary = validate_snapshot(target)
        print(f"      current target: {cur_size/1e6:.1f}MB ({cur_summary})")
    else:
        print(f"      current target: does not exist")

    if not args.yes:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans != "y":
            print("aborted")
            return 0

    # Stage to a sibling .restoring file, then rename atomically.
    staging = target.with_suffix(target.suffix + ".restoring")
    shutil.copy2(chosen, staging)
    # Stash the existing file before clobbering it (in case the user
    # changes their mind).
    if target.exists():
        stash = target.with_name(
            f"{target.stem}_PRE_RESTORE_{datetime.now():%Y%m%d_%H%M%S}{target.suffix}"
        )
        shutil.move(target, stash)
        log.info(f"existing target moved aside → {stash.name}")
    staging.rename(target)
    log.info(f"restored {chosen.name} → {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
