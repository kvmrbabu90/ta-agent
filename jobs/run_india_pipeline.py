"""Run the India daily pipeline ONCE and exit.

    python -m jobs.run_india_pipeline

India counterpart to ``run_pipeline.py`` (US). Wired by the Windows
Scheduled Task ``ta-agent-pipeline-india-am-ct`` to fire daily at
06:00 AM CT (~04:30-05:30 PM IST, 1-2 hours after NSE close).

Steps:
  1. Kite daily_update for NIFTY100
  2. daily_predict (NIFTY100 path; US is overnight and gets skipped)
  3. settle predictions whose horizon matured
  4. paper backtest replay
  5. drift check

Exit code: always 0 (intentional — Task Scheduler shouldn't flag the
whole run as failed when one step has a soft error like "kite token
expired"; per-step errors are logged separately by ``_safe_run``).
"""

from __future__ import annotations

import sys
from datetime import datetime

from packages.common.logging import log


def main() -> int:
    started = datetime.utcnow().isoformat()
    log.info(f"[run_india_pipeline] starting at {started}Z (UTC)")

    from jobs.scheduler import _job_india_ct_pipeline
    _job_india_ct_pipeline()

    log.info("[run_india_pipeline] complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
