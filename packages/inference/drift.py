"""Model-drift detector.

Watches the deployed model's recent rank-IC and flags when it has
degraded enough to warrant an off-cycle retrain. Designed to be safe:
two-condition trigger so a single noisy day doesn't fire a retrain.

Trigger logic:
    1. Pull the last `lookback_days` (default 20) of SETTLED predictions
       from `predictions_log` for `universe`. A row is settled when its
       5-day forward window has matured and `realized_return IS NOT NULL`.
    2. Compute mean daily rank-IC across those dates.
    3. Trigger drifted=True iff:
         (mean rank-IC < `hard_threshold`) OR
         (mean rank-IC < `soft_threshold` AND last `consecutive_days`
          daily rank-ICs are all < soft_threshold)

       The hard threshold catches catastrophic drift in one check; the
       soft+consecutive condition catches slow degradation while
       tolerating one or two noisy days.

Cooldown:
    The detector also rejects a trigger if the most recent model in the
    registry is younger than `min_model_age_days`. Without this, we'd
    re-retrain immediately if the freshly-retrained model still shows
    poor IC for the first few days post-deploy.

Calibration reference (SP500, May 2026):
    CV-time rank-IC at training      : 0.0167
    Recent live 90-day rank-IC       : -0.006
    Random baseline                  : 0
    Hard threshold (always-trigger)  : -0.005
    Soft threshold (with confirmation): 0.005
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from packages.common.config import MODELS_DIR, settings
from packages.common.logging import log
from packages.modeling.evaluate import rank_information_coefficient
from packages.modeling.registry import find_latest_model_dir

DEFAULT_LOOKBACK_DAYS = 20
DEFAULT_HARD_THRESHOLD = -0.005
DEFAULT_SOFT_THRESHOLD = 0.005
DEFAULT_CONSECUTIVE_DAYS = 10
DEFAULT_MIN_MODEL_AGE_DAYS = 14


def _load_recent_predictions(
    universe: str,
    *,
    lookback_days: int,
    sqlite_path: str | None = None,
) -> pd.DataFrame:
    """Pull the last `lookback_days` of settled predictions."""
    p = sqlite_path or settings.predictions_sqlite_path
    cutoff = date.today() - timedelta(days=lookback_days * 3)  # widen for non-trading days
    conn = sqlite3.connect(p)
    try:
        df = pd.read_sql_query(
            """
            SELECT as_of, symbol, predicted_return, realized_return
            FROM predictions_log
            WHERE universe = ?
              AND realized_return IS NOT NULL
              AND as_of >= ?
            """,
            conn, params=[universe, cutoff.isoformat()],
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    return df


def _daily_rank_ics(df: pd.DataFrame) -> dict[date, float]:
    """Compute one rank-IC per as_of date. Returns {date: rank_ic}."""
    out: dict[date, float] = {}
    for d, group in df.groupby("as_of"):
        if len(group) < 5:
            continue
        ric = rank_information_coefficient(
            group["realized_return"].to_numpy(),
            group["predicted_return"].to_numpy(),
        )
        if not np.isnan(ric):
            out[d] = float(ric)
    return out


def _model_age_days(universe: str) -> int | None:
    """Days since the most recent regression model was promoted. None if absent."""
    try:
        d = find_latest_model_dir(universe, "regression", MODELS_DIR)
    except FileNotFoundError:
        return None
    # Model dirs are named like SP500_regression_20260510_061125 — pull the date.
    name = d.name
    parts = name.split("_")
    for tok in parts:
        if len(tok) == 8 and tok.isdigit():
            try:
                trained_on = datetime.strptime(tok, "%Y%m%d").date()
                return (date.today() - trained_on).days
            except ValueError:
                pass
    # Fallback: filesystem mtime.
    try:
        ts = Path(d).stat().st_mtime
        return (date.today() - datetime.fromtimestamp(ts).date()).days
    except OSError:
        return None


def check_drift(
    universe: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    hard_threshold: float = DEFAULT_HARD_THRESHOLD,
    soft_threshold: float = DEFAULT_SOFT_THRESHOLD,
    consecutive_days: int = DEFAULT_CONSECUTIVE_DAYS,
    min_model_age_days: int = DEFAULT_MIN_MODEL_AGE_DAYS,
    sqlite_path: str | None = None,
) -> dict:
    """Decide whether the deployed model has drifted enough to warrant retraining.

    Returns a dict:
        drifted        bool
        rank_ic        mean daily rank-IC over the lookback window
        n_dates        # of dates evaluated
        threshold      threshold actually triggered (hard or soft)
        consecutive_below   # of recent days below soft_threshold
        model_age_days     # of days since last promoted model
        cooldown       True if a fresh model is too young to fairly judge
        reason         human-readable explanation
    """
    age = _model_age_days(universe)
    if age is not None and age < min_model_age_days:
        return {
            "drifted": False, "rank_ic": None, "n_dates": 0,
            "threshold": None, "consecutive_below": 0, "model_age_days": age,
            "cooldown": True,
            "reason": f"model is only {age}d old (min_age={min_model_age_days}d)",
        }

    df = _load_recent_predictions(universe, lookback_days=lookback_days, sqlite_path=sqlite_path)
    if df.empty:
        return {
            "drifted": False, "rank_ic": None, "n_dates": 0,
            "threshold": None, "consecutive_below": 0, "model_age_days": age,
            "cooldown": False,
            "reason": "no settled predictions in window",
        }
    daily = _daily_rank_ics(df)
    if not daily:
        return {
            "drifted": False, "rank_ic": None, "n_dates": 0,
            "threshold": None, "consecutive_below": 0, "model_age_days": age,
            "cooldown": False,
            "reason": "no daily rank-IC could be computed (groups too small)",
        }
    sorted_dates = sorted(daily.keys())[-lookback_days:]
    ics = [daily[d] for d in sorted_dates]
    mean_ric = float(np.mean(ics))

    if mean_ric < hard_threshold:
        return {
            "drifted": True, "rank_ic": mean_ric, "n_dates": len(ics),
            "threshold": hard_threshold, "consecutive_below": len(ics),
            "model_age_days": age, "cooldown": False,
            "reason": f"mean rank-IC {mean_ric:.4f} < hard threshold {hard_threshold}",
        }

    # Soft + consecutive check.
    consecutive = 0
    for ric in reversed(ics):
        if ric < soft_threshold:
            consecutive += 1
        else:
            break

    if mean_ric < soft_threshold and consecutive >= consecutive_days:
        return {
            "drifted": True, "rank_ic": mean_ric, "n_dates": len(ics),
            "threshold": soft_threshold, "consecutive_below": consecutive,
            "model_age_days": age, "cooldown": False,
            "reason": (
                f"mean rank-IC {mean_ric:.4f} < soft threshold {soft_threshold} "
                f"AND last {consecutive} consecutive days below soft threshold"
            ),
        }

    return {
        "drifted": False, "rank_ic": mean_ric, "n_dates": len(ics),
        "threshold": None, "consecutive_below": consecutive,
        "model_age_days": age, "cooldown": False,
        "reason": (
            f"mean rank-IC {mean_ric:.4f} above thresholds "
            f"(hard={hard_threshold}, soft={soft_threshold})"
        ),
    }


__all__ = ["check_drift", "DEFAULT_HARD_THRESHOLD", "DEFAULT_SOFT_THRESHOLD"]


# Silence the unused-import linter for `log`; reserved for future signal.
_ = log
