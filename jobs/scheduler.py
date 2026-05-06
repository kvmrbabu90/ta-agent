"""APScheduler-based daily orchestrator.

Run with:

    python -m jobs.scheduler

Schedules (all stored in UTC; comments give the local intent):

    Daily ingest, US        weekdays  22:30 UTC = 17:30 ET (winter, after US close)
    Daily ingest, India     weekdays  10:30 UTC = 16:00 IST (after NSE close)
    Daily predict, US       weekdays  23:00 UTC = 18:00 ET
    Daily predict, India    weekdays  11:00 UTC = 16:30 IST
    Settlement catch-up     weekdays  23:30 UTC = 18:30 ET (safety net)
    Monthly retrain         1st-3rd of month, weekdays only, 07:00 UTC ≈ 02:00 ET
                            (cron will fire on day=1..3 mon-fri, on the FIRST match
                            APScheduler runs the job; subsequent days re-fire but
                            monthly_retrain is idempotent within a calendar day)
    Quarterly universe refresh  Jan/Apr/Jul/Oct 5th at 07:00 UTC

Scheduling strategy:
    Each scheduled function is wrapped in ``_safe_run`` which logs start +
    duration + exceptions. The scheduler must never crash on a single job
    failure — it's the noisiest log producer in the system.

DST note:
    We pin to UTC, so US and IN local times shift by ±1h across DST. The
    cost is acceptable: jobs may fire 60 minutes earlier in summer, well
    after market close in either timezone. If you want strict local times,
    swap the timezone arg below to "America/New_York" / "Asia/Kolkata"
    and split the schedule across two schedulers.
"""

from __future__ import annotations

import signal
import sys
import time
import traceback
from collections.abc import Callable
from datetime import date, datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from packages.common.logging import log

# ---------------------------------------------------------------------------
# Job wrappers — keep imports lazy so a failing optional dependency doesn't
# bring down scheduler startup.
# ---------------------------------------------------------------------------


def _safe_run(name: str, fn: Callable[[], object]) -> None:
    """Run a scheduled job with bookend logging and full exception capture."""
    started = time.monotonic()
    log.info(f"[scheduler] {name}: starting")
    try:
        result = fn()
        elapsed = time.monotonic() - started
        log.info(f"[scheduler] {name}: completed in {elapsed:.1f}s result={result!r:.300s}")
    except Exception as exc:
        elapsed = time.monotonic() - started
        log.error(
            f"[scheduler] {name}: FAILED after {elapsed:.1f}s exc={exc!r}\n"
            f"{traceback.format_exc()}"
        )


def _job_us_ingest() -> None:
    from packages.ingestion.adapters.ib_adapter import daily_update

    _safe_run("us_ingest", lambda: daily_update("SP500"))


def _job_india_ingest() -> None:
    from packages.ingestion.adapters.kite_adapter import daily_update

    _safe_run("india_ingest", lambda: daily_update("NIFTY100"))


def _job_daily_predict() -> None:
    from jobs.daily_predict import run as predict_run

    _safe_run("daily_predict", lambda: predict_run())


def _job_settlement_catchup() -> None:
    from packages.inference.tracking import settle_predictions

    _safe_run(
        "settlement_catchup",
        lambda: settle_predictions(as_of=date.today(), horizon_days=5),
    )


def _job_monthly_retrain() -> None:
    from jobs.monthly_retrain import run as retrain_run

    _safe_run("monthly_retrain", lambda: retrain_run())


def _job_universe_refresh() -> None:
    from packages.ingestion.universe.membership import refresh_all_universes

    _safe_run("universe_refresh", lambda: refresh_all_universes())


# ---------------------------------------------------------------------------
# Scheduler construction
# ---------------------------------------------------------------------------


def make_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler(timezone="UTC")

    # --- Daily ingest -----------------------------------------------------
    sched.add_job(
        _job_us_ingest,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=30, timezone="UTC"),
        id="us_ingest",
        name="US daily ingest (IB → SP500)",
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        _job_india_ingest,
        CronTrigger(day_of_week="mon-fri", hour=10, minute=30, timezone="UTC"),
        id="india_ingest",
        name="India daily ingest (Kite → NIFTY100)",
        max_instances=1,
        coalesce=True,
    )

    # --- Daily predict ----------------------------------------------------
    # daily_predict orchestrates BOTH universes (SP500 + NIFTY100) and skips
    # automatically on non-trading days, so two firings per day cover both.
    sched.add_job(
        _job_daily_predict,
        CronTrigger(day_of_week="mon-fri", hour=11, minute=0, timezone="UTC"),
        id="daily_predict_in",
        name="Daily predict (post-IST close)",
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        _job_daily_predict,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=0, timezone="UTC"),
        id="daily_predict_us",
        name="Daily predict (post-ET close)",
        max_instances=1,
        coalesce=True,
    )

    # --- Settlement catch-up ----------------------------------------------
    sched.add_job(
        _job_settlement_catchup,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=30, timezone="UTC"),
        id="settlement_catchup",
        name="Settle pending predictions (safety net)",
        max_instances=1,
        coalesce=True,
    )

    # --- Monthly retrain --------------------------------------------------
    # First weekday of each month at 07:00 UTC. APScheduler doesn't expose
    # a "first business day" trigger, so we widen day=1-3 + day_of_week=mon-fri,
    # which guarantees the FIRST trigger lands on the first weekday. The job
    # is idempotent within a date — cheap to fire harmlessly if it overlaps.
    sched.add_job(
        _job_monthly_retrain,
        CronTrigger(day="1-3", day_of_week="mon-fri", hour=7, minute=0, timezone="UTC"),
        id="monthly_retrain",
        name="Monthly retrain (compare + promote)",
        max_instances=1,
        coalesce=True,
    )

    # --- Quarterly universe refresh ---------------------------------------
    sched.add_job(
        _job_universe_refresh,
        CronTrigger(month="1,4,7,10", day=5, hour=7, minute=0, timezone="UTC"),
        id="universe_refresh",
        name="Quarterly universe membership refresh",
        max_instances=1,
        coalesce=True,
    )

    return sched


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _install_signal_handlers(sched: BlockingScheduler) -> None:
    def _shutdown(signum: int, _frame: object) -> None:
        log.info(f"[scheduler] received signal {signum}; shutting down gracefully")
        sched.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)


def _print_schedule(sched: BlockingScheduler) -> None:
    log.info("[scheduler] registered jobs:")
    for job in sched.get_jobs():
        log.info(f"  - id={job.id} name={job.name!r} trigger={job.trigger}")


def main() -> int:
    sched = make_scheduler()
    _install_signal_handlers(sched)
    _print_schedule(sched)
    log.info(f"[scheduler] starting at {datetime.utcnow().isoformat()}Z (UTC)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("[scheduler] interrupted; bye")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
