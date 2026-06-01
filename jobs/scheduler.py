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

# Live paper account "go-live" date — anything older is excluded so the
# dashboard equity curve always starts at the configured starting cash on
# this date. Update to re-baseline (e.g. after a strategy change worth
# marking fresh).
LIVE_PAPER_START_DATE = date(2026, 6, 2)
# Starting cash for the live paper run. Pinned here so the simulator
# re-builds the run with the right baseline on every pipeline tick (the
# backtest() function rewrites paper_runs.default.starting_cash from
# StrategyConfig on each fire — leaving this as the StrategyConfig
# default would silently revert any DB-side manual rebase).
LIVE_PAPER_STARTING_CASH = 200_000.0

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


def _job_us_yfinance_refresh() -> None:
    """Refresh SP500 daily bars via yfinance — works without IB Gateway.

    Used by the CT-anchored 8 AM / 5 PM triggers below. Lighter-weight
    than the IB path: no socket, no auth, just rate-limited HTTP. Each
    symbol's missing tail is computed and pulled.
    """
    from packages.ingestion.adapters.yfinance_adapter import daily_update

    _safe_run("us_yf_refresh", lambda: daily_update("SP500"))


def _job_spy_refresh() -> None:
    """Pull the most recent SPY bars (benchmark for /performance + paper trade)."""
    from datetime import date, timedelta

    from packages.ingestion.adapters.yfinance_adapter import fetch_daily_bars
    from packages.ingestion.storage import get_conn, upsert_ohlcv

    def _refresh_spy() -> dict:
        end = date.today()
        start = end - timedelta(days=10)
        df = fetch_daily_bars("SPY", start, end, universe="", exchange="NYSE")
        if df.empty:
            return {"refreshed": 0}
        with get_conn() as conn:
            n = upsert_ohlcv(df, conn=conn)
        return {"refreshed": int(n)}

    _safe_run("spy_refresh", _refresh_spy)


def _job_news_classify() -> None:
    """Audit-only LLM classifier over today's top long picks.

    Runs after daily_predict so today's predictions exist; runs BEFORE
    paper_backtest in the pipeline ordering, but the paper engine does
    NOT consume verdicts yet — we accumulate paired (verdict, realized
    5d return) data first, then decide whether to act on it.

    Soft-fails: a missing Ollama process or model shouldn't break the
    pipeline. The pipeline's other steps proceed regardless.
    """
    from packages.news import classify_top_picks
    from packages.news.classifier import healthcheck

    def _run() -> dict:
        hc = healthcheck()
        if not hc.get("ok"):
            log.warning(f"news_classify: skipped (ollama unhealthy: {hc})")
            return {"skipped": True, "reason": "ollama_unhealthy", **hc}
        return classify_top_picks(universe="SP500", top_n=10)

    _safe_run("news_classify", _run)


def _job_paper_backtest() -> None:
    """Replay the model's predictions through the paper-trading engine,
    refreshing the equity curve, open positions, and trade log.

    `start_date` is pinned to ``LIVE_PAPER_START_DATE`` (see module top)
    so the live paper account starts from a known "today $1000" baseline
    and doesn't drag in stale historical predictions. Backfilled
    predictions (12-month window) are only used by the grid-search
    optimizer, which runs unbounded."""
    from packages.paper_trading import StrategyConfig, backtest

    _safe_run(
        "paper_backtest",
        lambda: backtest(StrategyConfig(
            run_id="default",
            start_date=LIVE_PAPER_START_DATE,
            starting_cash=LIVE_PAPER_STARTING_CASH,
        )),
    )


def _job_us_ct_pipeline() -> None:
    """Combined CT-anchored job: refresh OHLCV (SP500 + SPY), generate
    today's predictions, settle anything that's matured, then re-run the
    paper backtest. Wired to fire twice daily at 8 AM CT and 5 PM CT.

    Each step is wrapped in its own _safe_run so a failing step doesn't
    short-circuit the rest.
    """
    log.info("[scheduler] us_ct_pipeline: starting")
    _job_us_yfinance_refresh()
    _job_spy_refresh()
    _job_daily_predict()
    _job_settlement_catchup()
    _job_news_classify()   # audit-only LLM verdicts over today's longs
    _job_paper_backtest()
    # Drift check fires AFTER settlement so realized_returns are fresh.
    # It's cheap (a few SQL queries + IC compute) and has its own
    # cooldown logic — safe to run on every pipeline tick.
    _job_drift_check()
    log.info("[scheduler] us_ct_pipeline: complete")


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
    """Cheap monthly retrain — refreshes weights, reuses Optuna-tuned
    hyperparameters from the most recent quarterly tune. ~5 min/universe."""
    from jobs.monthly_retrain import run as retrain_run

    _safe_run("monthly_retrain", lambda: retrain_run(do_tune=False))


def _job_quarterly_retune() -> None:
    """Expensive quarterly Optuna re-tune — 20 trials × ~10 min/trial =
    ~3 hours/universe. Refreshes the hyperparameters that the monthly
    retrains then reuse for the next 3 months."""
    from jobs.monthly_retrain import run as retrain_run

    _safe_run("quarterly_retune", lambda: retrain_run(do_tune=True, n_trials=20))


def _job_drift_check() -> None:
    """Daily check: if the deployed model's recent rank-IC has degraded
    below threshold, fire an off-cycle retrain. Catches regime changes
    between scheduled monthly retrains.

    Triggers on the SP500 model only — NIFTY100 lacks enough live data.

    HOLD-OFF lock file: when ``data/processed/.no_retrain.lock`` exists,
    the drift check still runs (so we keep logging rank-IC), but any
    drift-triggered retrain is suppressed. Used to keep the GPU free
    for a long-running strict-WF backtest. Delete the lock file to
    re-enable emergency retrains."""
    import os
    from jobs.monthly_retrain import run as retrain_run
    from packages.inference.drift import check_drift

    LOCK_PATH = "data/processed/.no_retrain.lock"

    def _check_and_maybe_retrain() -> dict:
        verdict = check_drift("SP500")
        if not verdict.get("drifted"):
            return {"drifted": False, **verdict}
        if os.path.exists(LOCK_PATH):
            log.warning(
                f"drift_check: SP500 has drifted, but {LOCK_PATH} is set — "
                f"suppressing emergency retrain. rank_ic={verdict.get('rank_ic')}, "
                f"threshold={verdict.get('threshold')}. Delete the lock to re-enable."
            )
            return {"drifted": True, "retrain_suppressed_by_lock": True, "verdict": verdict}
        log.warning(
            f"drift_check: SP500 model has drifted "
            f"(rank_ic={verdict.get('rank_ic'):.4f} over {verdict.get('n_dates')} days, "
            f"threshold={verdict.get('threshold')}); firing emergency retrain"
        )
        retrain_result = retrain_run(universes=("SP500",), do_tune=False)
        return {"drifted": True, "verdict": verdict, "retrain": retrain_result}

    _safe_run("drift_check", _check_and_maybe_retrain)


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

    # --- Daily predict ----------------------------------------------------
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

    # --- Monthly retrain (cheap) -----------------------------------------
    # First weekday of each month at 07:00 UTC. APScheduler doesn't expose
    # a "first business day" trigger, so we widen day=1-3 + day_of_week=mon-fri,
    # which guarantees the FIRST trigger lands on the first weekday. The job
    # is idempotent within a date — cheap to fire harmlessly if it overlaps.
    # Reuses cached Optuna hyperparameters; ~5 min/universe.
    sched.add_job(
        _job_monthly_retrain,
        CronTrigger(day="1-3", day_of_week="mon-fri", hour=7, minute=0, timezone="UTC"),
        id="monthly_retrain",
        name="Monthly retrain (cached hyperparams)",
        max_instances=1,
        coalesce=True,
    )

    # --- Quarterly Optuna re-tune (expensive) ----------------------------
    # First weekday of Jan/Apr/Jul/Oct at 09:00 UTC. Re-runs the full
    # Optuna search to refresh hyperparameters. ~3 hours/universe; the
    # later hour avoids overlapping with the cheap monthly job.
    sched.add_job(
        _job_quarterly_retune,
        CronTrigger(
            month="1,4,7,10", day="1-3", day_of_week="mon-fri",
            hour=9, minute=0, timezone="UTC",
        ),
        id="quarterly_retune",
        name="Quarterly Optuna re-tune (20 trials)",
        max_instances=1,
        coalesce=True,
    )

    # --- Daily drift check -----------------------------------------------
    # Runs after the post-close US pipeline at 17:00 CT (= 22:00 UTC),
    # well after settlement_catchup has updated realized_returns. If
    # the deployed SP500 model's rank-IC over the last 20 settled days
    # is below threshold, fires an off-cycle retrain.
    sched.add_job(
        _job_drift_check,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=15, timezone="UTC"),
        id="drift_check",
        name="Daily drift detector (rank-IC monitor)",
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

    # --- US CT-anchored full pipeline (8:35 AM + 5 PM CT) -----------------
    # Uses America/Chicago tz so DST is automatic. Runs the full chain on
    # each tick: yfinance OHLCV refresh -> SPY refresh -> daily_predict ->
    # settlement catch-up -> paper_trading backtest. Each step is logged
    # individually; a failure in one step doesn't abort the rest.
    #
    # Morning trigger sits 5 min after the equity open (08:30 CT) so
    # yfinance has reliably published today's OPEN bar by the time the
    # paper backtest reads it. daily_predict still builds features off
    # yesterday's complete bar — today's intraday bar is ignored.
    sched.add_job(
        _job_us_ct_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=35, timezone="America/Chicago"),
        id="us_pipeline_8am_ct",
        name="US pipeline (08:35 CT post-open)",
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        _job_us_ct_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=17, minute=0, timezone="America/Chicago"),
        id="us_pipeline_5pm_ct",
        name="US pipeline (17:00 CT post-close)",
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
