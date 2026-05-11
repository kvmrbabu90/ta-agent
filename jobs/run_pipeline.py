"""Run the full US daily pipeline ONCE and exit.

    python -m jobs.run_pipeline

Steps (each independently logged):
  1. yfinance daily_update for SP500 (pulls missing tail bars)
  2. SPY benchmark refresh (last 10 days)
  3. daily_predict for SP500
  4. settle any predictions whose 5-day horizon has matured
  5. paper-trading backtest replay (refreshes equity curve / positions)

Use this with Windows Task Scheduler (or any external cron) to fire the
pipeline at 8 AM CT and 5 PM CT each weekday — instead of running the
long-lived APScheduler process. Both approaches do the same work; this
one is more robust on Windows because it doesn't require a persistent
process or third-party service-wrapper (nssm/winsw).

Exit code:
  0 if every step ran without raising. Each step uses the scheduler's
  _safe_run wrapper, so a step's exception is logged and the next step
  still runs. The script returns 0 even if individual steps logged errors
  (intentional — Task Scheduler then doesn't mark the whole run as failed).
"""

from __future__ import annotations

import sys
from datetime import datetime

from packages.common.logging import log


def main() -> int:
    started = datetime.utcnow().isoformat()
    log.info(f"[run_pipeline] starting at {started}Z (UTC)")

    # Reuse the same pipeline function the long-lived scheduler uses, so
    # the two paths can never diverge.
    from jobs.scheduler import _job_us_ct_pipeline
    _job_us_ct_pipeline()

    log.info("[run_pipeline] complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
