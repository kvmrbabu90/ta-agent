"""Run news classification over today's top picks and exit.

    python -m jobs.news_classify
    python -m jobs.news_classify --universe SP500 --top-n 10
    python -m jobs.news_classify --as-of 2026-05-11 --top-n 15

Used as a step in the daily scheduled pipeline (after daily_predict,
before paper_backtest). Standalone for ad-hoc reruns / debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from packages.common.logging import log
from packages.news import classify_top_picks
from packages.news.classifier import DEFAULT_MODEL, healthcheck


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="SP500")
    p.add_argument(
        "--as-of", type=date.fromisoformat, default=None,
        help="Default: today.",
    )
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    hc = healthcheck(model=args.model)
    if not hc.get("ok"):
        log.error(f"news_classify: ollama healthcheck failed: {hc}")
        return 1

    try:
        summary = classify_top_picks(
            universe=args.universe,
            as_of=args.as_of,
            top_n=args.top_n,
            model=args.model,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception(f"news_classify crashed: {exc!r}")
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
