"""Daily prediction orchestration.

Workflow:
    1. Settle yesterday's (and older) predictions whose horizon has now closed.
    2. For each universe, run predict_universe.
    3. Idempotently log to SQLite (re-running on same day overwrites
       prediction columns but preserves any already-realized fields).
    4. Print top-N long / short picks.

Importable function ``run`` so APScheduler / cron / manual CLI all share
the same code path.
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Any

import pandas_market_calendars as mcal

from packages.common.logging import log
from packages.inference.predict import predict_universe
from packages.inference.ranker import rank_predictions
from packages.inference.tracking import log_predictions, settle_predictions

_UNIVERSES = (("SP500", "NYSE"), ("NIFTY100", "XNSE"))


def _is_trading_day(market_code: str, when: date) -> bool:
    try:
        cal = mcal.get_calendar(market_code)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"market calendar {market_code} unavailable: {exc!r}")
        return False
    sched = cal.schedule(start_date=when.isoformat(), end_date=when.isoformat())
    return not sched.empty


def run(
    today: date | None = None,
    *,
    universes: tuple[tuple[str, str], ...] = _UNIVERSES,
    horizon_days: int = 5,
    top_n: int = 10,
) -> dict[str, Any]:
    today = today or date.today()
    log.info(f"daily_predict: as_of={today}")

    settled = settle_predictions(as_of=today, horizon_days=horizon_days)
    log.info(f"daily_predict: settled {settled} previously open predictions")

    summary: dict[str, Any] = {
        "as_of": today.isoformat(),
        "settled": settled,
        "per_universe": {},
    }

    for universe, market in universes:
        if not _is_trading_day(market, today):
            log.info(f"daily_predict: {today} not a trading day for {market}; skipping {universe}")
            summary["per_universe"][universe] = {"skipped": "non-trading-day"}
            continue

        log.info(f"daily_predict: predicting {universe}")
        try:
            preds = predict_universe(universe, as_of=today)
        except FileNotFoundError as exc:
            log.error(f"no model for {universe}: {exc!r}")
            summary["per_universe"][universe] = {"error": "no-model"}
            continue
        except Exception as exc:  # noqa: BLE001
            log.exception(f"predict_universe({universe}) crashed: {exc!r}")
            summary["per_universe"][universe] = {"error": repr(exc)}
            continue

        if preds.empty:
            log.warning(f"{universe}: no predictions produced")
            summary["per_universe"][universe] = {"n_predictions": 0}
            continue

        n_logged = log_predictions(preds)
        ranked = rank_predictions(preds, top_n=top_n)

        log.info(f"{universe}: logged {n_logged} predictions")
        long_view = ranked["long_picks"][
            ["symbol", "predicted_return", "top_quintile_proba"]
        ].head(top_n)
        short_view = ranked["short_picks"][
            ["symbol", "predicted_return", "bottom_quintile_proba"]
        ].head(top_n)
        log.info(f"{universe} top {top_n} long picks:\n{long_view.to_string(index=False)}")
        log.info(f"{universe} top {top_n} short picks:\n{short_view.to_string(index=False)}")

        summary["per_universe"][universe] = {
            "n_predictions": len(preds),
            "n_logged": n_logged,
            "long_picks": ranked["long_picks"].to_dict("records")[:top_n],
            "short_picks": ranked["short_picks"].to_dict("records")[:top_n],
            "n_high_confidence_long": len(ranked["high_confidence_long"]),
            "n_high_confidence_short": len(ranked["high_confidence_short"]),
        }

    return summary


def determine_exit_code(summary: dict[str, Any]) -> int:
    """Non-zero only when *every* universe failed."""
    per = summary.get("per_universe", {})
    if not per:
        return 1
    if all(("error" in v or v.get("n_predictions") == 0) for v in per.values()):
        return 1
    return 0


def main() -> int:
    try:
        summary = run()
        return determine_exit_code(summary)
    except Exception as exc:  # noqa: BLE001
        log.exception(f"daily_predict crashed: {exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
