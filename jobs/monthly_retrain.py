"""Monthly retraining workflow.

For each universe:
    1. Load the latest training dataset (or rebuild it).
    2. Read CV metrics from the currently-deployed model (if any).
    3. Run a small Optuna search + final-model train.
    4. Compare new mean_daily_rank_ic to the deployed baseline.
    5. Promote the new model (save_model) only if it's at least 80% of
       the baseline IC, or there is no baseline. Otherwise log a warning
       and retain the old.
    6. Persist a JSON report under data/models/retrain_reports/{date}.json.

Idempotent within a calendar day — the report file is overwritten if you
re-run; the registry only ever appends a new model directory when we
explicitly call save_model.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.config import MODELS_DIR
from packages.common.logging import log
from packages.labels.dataset import build_training_dataset
from packages.modeling.calibrate import calibrate_classifier
from packages.modeling.registry import ModelMetadata, load_latest_model, save_model
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import (
    TrainConfig,
    train_final_model,
    train_with_cv,
)
from packages.modeling.tune import tune_hyperparameters

_RETAIN_THRESHOLD = 0.80  # new model's IC must be >= 0.8 * baseline to promote
# Classification target is forward-return QUINTILE → 5 classes. Used to
# subtract the random-guess accuracy floor (1/5 = 0.20) before applying the
# 80%-of-baseline gate, so the gate compares EDGE over random rather than
# raw accuracy (raw accuracy floors at 0.20, making an 80%-of-raw-accuracy
# gate unable to ever reject a near-random model).
_QUINTILE_N_CLASSES = 5
_DEFAULT_HORIZON = 5
_DEFAULT_LOOKBACK_YEARS = 10
_TUNE_TRIALS = 20

_NON_FEATURE_COLS = {
    "symbol", "bar_date", "in_universe", "fwd_return_5d", "fwd_quintile_5d",
}


def _baseline_metric(universe: str, target: str) -> float | None:
    """Return the deployed model's comparison metric, or None if absent.

    The promotion gate compares the candidate's CV metric against the
    incumbent's. The metric differs by target — and crucially, so does the
    metadata key it's stored under:
      regression     → mean_daily_rank_ic
      classification → val_accuracy

    BUG HISTORY: this used to look for ``mean_daily_rank_ic`` unconditionally.
    Classification metadata never carries that key (it stores val_accuracy),
    so the lookup always returned None for classification → _should_promote
    saw "no baseline" → the classification head promoted EVERY retrain
    unchecked, with no protection against a degraded model. Now keyed by
    target so classification is gated like regression.
    """
    metric_key = "val_accuracy" if target == "classification" else "mean_daily_rank_ic"
    try:
        _, metadata = load_latest_model(universe, target)
    except FileNotFoundError:
        return None
    cv_metrics = metadata.get("cv_metrics", {}) or {}
    mean_block = cv_metrics.get("mean") if isinstance(cv_metrics, dict) else None
    if isinstance(mean_block, dict) and metric_key in mean_block:
        v = mean_block.get(metric_key)
        return float(v) if v is not None else None
    # Older metadata: metric stored at top level of cv_metrics.
    if isinstance(cv_metrics, dict) and metric_key in cv_metrics:
        v = cv_metrics.get(metric_key)
        return float(v) if v is not None else None
    return None


def _should_promote(new_ic: float | None, baseline_ic: float | None) -> tuple[bool, str]:
    """Return (promote?, reason) per the 80%-of-baseline rule."""
    if new_ic is None:
        return False, "new model produced no IC (training/eval failed?)"
    if baseline_ic is None:
        return True, "no baseline model — promoting first model"
    if new_ic >= baseline_ic * _RETAIN_THRESHOLD:
        return True, f"new IC {new_ic:.4f} >= {_RETAIN_THRESHOLD:.0%} of baseline {baseline_ic:.4f}"
    return False, f"new IC {new_ic:.4f} < {_RETAIN_THRESHOLD:.0%} of baseline {baseline_ic:.4f}"


def _train_one_target(
    df: pd.DataFrame,
    universe: str,
    target: str,
    *,
    horizon_days: int,
    final_train_end: date,
    do_tune: bool,
    n_trials: int,
) -> dict[str, Any]:
    """Train + (optionally) calibrate for one (universe, target). Returns the
    summary dict; promotion happens upstream in run_universe()."""
    label_col = (
        f"fwd_return_{horizon_days}d" if target == "regression" else f"fwd_quintile_{horizon_days}d"
    )
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    splitter = PurgedWalkForwardSplit(
        n_folds=5,
        horizon_days=horizon_days,
        embargo_days=horizon_days,
    )
    base_cfg = TrainConfig(objective=target)

    if do_tune:
        tuned_cfg, _study = tune_hyperparameters(
            df, feature_cols, label_col, splitter, base_cfg, n_trials=n_trials
        )
    else:
        tuned_cfg = base_cfg

    cv = train_with_cv(df, feature_cols, label_col, splitter, tuned_cfg)
    booster, _final_metrics = train_final_model(
        df, feature_cols, label_col, tuned_cfg, final_train_end
    )

    model_to_save = booster
    if target == "classification":
        es_cutoff = final_train_end - timedelta(days=60)
        cal_start = es_cutoff - timedelta(days=30)
        df_local = df.copy()
        df_local["bar_date"] = pd.to_datetime(df_local["bar_date"]).dt.date
        cal_mask = (
            (df_local["bar_date"] > cal_start)
            & (df_local["bar_date"] <= es_cutoff)
            & (df_local["in_universe"].fillna(False).astype(bool))
            & (df_local[label_col].notna())
        )
        if cal_mask.sum() > 0:
            X_cal = df_local.loc[cal_mask, feature_cols]
            y_cal = df_local.loc[cal_mask, label_col].astype(int)
            model_to_save = calibrate_classifier(booster, X_cal, y_cal)

    return {
        "label_col": label_col,
        "feature_cols": feature_cols,
        "config": asdict(tuned_cfg),
        "cv_metrics": cv["mean_metrics"],
        "cv_std": cv["std_metrics"],
        "model": model_to_save,
        "horizon_days": horizon_days,
    }


def run_universe(
    universe: str,
    *,
    horizon_days: int = _DEFAULT_HORIZON,
    lookback_years: int = _DEFAULT_LOOKBACK_YEARS,
    final_train_end: date | None = None,
    do_tune: bool = True,
    n_trials: int = _TUNE_TRIALS,
) -> dict[str, Any]:
    """Retrain both targets for one universe and return a summary."""
    fte = final_train_end or (date.today() - timedelta(days=60))
    start = fte - timedelta(days=365 * lookback_years)

    log.info(f"[retrain] {universe}: building dataset [{start} .. {fte}]")
    df = build_training_dataset(
        universe, start, fte, horizon_days=horizon_days
    )
    if df.empty:
        return {"universe": universe, "skipped": "empty-dataset"}

    summary: dict[str, Any] = {
        "universe": universe,
        "final_train_end": fte.isoformat(),
        "horizon_days": horizon_days,
        "results": {},
    }

    for target in ("regression", "classification"):
        log.info(f"[retrain] {universe}/{target}: training")
        baseline_metric = _baseline_metric(universe, target)
        try:
            outcome = _train_one_target(
                df, universe, target,
                horizon_days=horizon_days,
                final_train_end=fte,
                do_tune=do_tune,
                n_trials=n_trials,
            )
        except Exception as exc:
            log.exception(f"[retrain] {universe}/{target}: training failed: {exc!r}")
            summary["results"][target] = {"error": repr(exc)}
            continue

        # Per-target comparison metric. Regression uses rank-IC directly
        # (naturally centered at 0, so the 80%-of-baseline rule is meaningful).
        # Classification uses val_accuracy, but compared as EDGE OVER RANDOM
        # (accuracy − 1/n_classes) so the same 80% rule actually bites — a
        # raw-accuracy gate can't, because 80% of a ~0.24 baseline (0.19) sits
        # below the 0.20 random floor.
        if target == "regression":
            new_metric = outcome["cv_metrics"].get("mean_daily_rank_ic")
            promote, reason = _should_promote(new_metric, baseline_metric)
        else:
            new_metric = outcome["cv_metrics"].get("val_accuracy")
            floor = 1.0 / _QUINTILE_N_CLASSES
            new_edge = (new_metric - floor) if new_metric is not None else None
            base_edge = (baseline_metric - floor) if baseline_metric is not None else None
            promote, reason = _should_promote(new_edge, base_edge)
            # Make the reason readable in raw-accuracy terms too.
            if new_metric is not None and baseline_metric is not None:
                reason += (
                    f" [acc {new_metric:.4f} vs baseline {baseline_metric:.4f}, "
                    f"edge-over-random basis]"
                )

        # Reported values are the raw comparison metrics (rank-IC / accuracy)
        # for human readability; the promote DECISION used the edge basis
        # for classification.
        new_ic = new_metric
        baseline_ic = baseline_metric
        log.info(
            f"[retrain] {universe}/{target}: baseline={baseline_ic} new={new_ic} "
            f"promote={promote} ({reason})"
        )

        if promote:
            metadata = ModelMetadata(
                universe=universe,
                target=target,
                horizon_days=horizon_days,
                feature_cols=outcome["feature_cols"],
                train_start=start,
                train_end=fte,
                config=outcome["config"],
                cv_metrics={
                    "mean": outcome["cv_metrics"],
                    "std": outcome["cv_std"],
                },
            )
            save_dir = save_model(outcome["model"], metadata)
            promoted_path = str(save_dir)
        else:
            promoted_path = None

        summary["results"][target] = {
            "baseline_ic": baseline_ic,
            "new_ic": new_ic,
            "promoted": promote,
            "reason": reason,
            "promoted_path": promoted_path,
        }

    return summary


def run(
    *,
    universes: tuple[str, ...] = ("SP500", "NIFTY100"),
    do_tune: bool = False,
    n_trials: int = _TUNE_TRIALS,
    final_train_end: date | None = None,
) -> dict[str, Any]:
    """Top-level entry. Writes a retrain report under data/models/retrain_reports/.

    `do_tune` defaults to False — the monthly cadence reuses the most-recently
    Optuna-tuned hyperparameters, retraining only the model weights against
    fresh data. Pass do_tune=True (called from the QUARTERLY job) to re-run
    the full Optuna search; that's expensive (~3 hours) but rare (4×/year).
    """
    today = date.today()
    out_dir = Path(MODELS_DIR) / "retrain_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "today": today.isoformat(),
        "universes": {},
    }
    for u in universes:
        try:
            report["universes"][u] = run_universe(
                u,
                final_train_end=final_train_end,
                do_tune=do_tune,
                n_trials=n_trials,
            )
        except Exception as exc:
            log.exception(f"[retrain] {u}: top-level failure: {exc!r}")
            report["universes"][u] = {"error": repr(exc)}

    report["finished_at"] = datetime.utcnow().isoformat() + "Z"
    report_path = out_dir / f"{today.isoformat()}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"[retrain] wrote report → {report_path}")
    return report


def _is_first_business_day_of_month(today: date | None = None) -> bool:
    """True iff today is the first Mon–Fri of its calendar month.

    Used by Windows Task Scheduler invocations that fire weekly but
    should only actually retrain on the first business day of each
    month. (Windows lacks a native cron-style "first weekday of month"
    trigger; the simplest correct answer is to fire weekly and gate
    inside the wrapper.)
    """
    today = today or date.today()
    if today.weekday() >= 5:  # Sat/Sun
        return False
    # Walk back to the start of the month, find the first weekday.
    for d in range(1, today.day + 1):
        candidate = date(today.year, today.month, d)
        if candidate.weekday() < 5:
            return candidate == today
    return False


def _is_first_business_day_of_quarter(today: date | None = None) -> bool:
    today = today or date.today()
    if today.month not in (1, 4, 7, 10):
        return False
    return _is_first_business_day_of_month(today)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--do-tune", action="store_true",
                   help="Re-run Optuna search (expensive, ~3h/universe). "
                        "Default: False — reuse cached hyperparameters.")
    p.add_argument("--n-trials", type=int, default=_TUNE_TRIALS)
    p.add_argument("--only-if-first-business-day-of-month", action="store_true",
                   help="Bail unless today is the 1st Mon–Fri of the calendar month. "
                        "Used by Windows Task Scheduler weekly-firing invocations.")
    p.add_argument("--only-if-first-business-day-of-quarter", action="store_true",
                   help="Bail unless today is the 1st Mon–Fri of Jan/Apr/Jul/Oct. "
                        "Used by Windows Task Scheduler weekly-firing invocations.")
    args = p.parse_args()
    if args.only_if_first_business_day_of_quarter and not _is_first_business_day_of_quarter():
        log.info("monthly_retrain: today is not first business day of quarter; skipping")
        return 0
    if args.only_if_first_business_day_of_month and not _is_first_business_day_of_month():
        log.info("monthly_retrain: today is not first business day of month; skipping")
        return 0
    run(do_tune=args.do_tune, n_trials=args.n_trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
