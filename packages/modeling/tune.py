"""Optuna-based hyperparameter tuning.

Search space:
    learning_rate, num_leaves, feature_fraction, bagging_fraction,
    min_data_in_leaf, lambda_l1, lambda_l2.

Objective:
    Regression     → maximize mean daily rank-IC across CV folds.
    Classification → maximize mean fold accuracy.

Trial constraints (Phase A1) — reject hyperparams that produce degenerate
"good-on-paper, bad-in-reality" winners:

    1. min_best_iter: every fold must train for at least N boosting rounds.
       A run with best_iter=1 across folds means LightGBM is fitting one tree
       and stopping; the metric is meaningless.
    2. min_fold_size: drop folds whose train_size is below the threshold from
       the mean calculation. Otherwise Optuna optimizes the mean by picking
       hyperparams that overfit the smallest fold.
    3. max_fold_dominance_z: reject trials where one fold's metric is more
       than ``z`` standard deviations above the mean of other folds. This
       catches the "fold-0-anomaly" pattern from May 9 where one fold scored
       0.19 rank-IC while others were near zero.

The TPE sampler is seeded for reproducibility. Studies can be persisted to
SQLite for resumption (``storage_path``); without it the study is in-memory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import optuna
import pandas as pd

from packages.common.logging import log
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_with_cv


@dataclass
class TuneConstraints:
    """Trial-level rejection criteria. Trials that violate any constraint
    return -inf so Optuna's TPE sampler learns to avoid that region.

    Why these specific defaults:
        - min_best_iter=5 catches the [1,1,1,1,1] degenerate case (where
          LightGBM stops after a single tree on every fold) but allows the
          normal early-stopping pattern of [1, 1, 18, 15, 4] that healthy
          training on the small first folds produces with our purged
          walk-forward CV. min_best_iter=50 was too strict — it rejected
          even the prior production-grade hyperparams.
        - min_fold_train_size=200_000 drops fold 0 (~138k rows). Fold 0 is
          where the May-9 fold-0-overfit pathology lived (rank-IC=0.19 on a
          tiny train set).
        - max_fold_dominance_z=2.5 catches the OTHER half of that pathology
          (one fold dramatically beating the mean of the others). The
          combination of "drop tiny fold" + "reject extreme outlier folds"
          is what blocks the degenerate Optuna winner from May 9.
    """

    # Reject trials where any retained fold has best_iter < this value.
    # Calibrated low (was 5 originally): the prior production hp itself
    # produces best_iter pattern [1, 1, 18, 15, 4] across folds. Setting
    # min_best_iter=5 rejected normal training. min_best_iter=2 catches
    # the truly degenerate "every fold stops at iter 1" case while allowing
    # healthy walk-forward CV behavior.
    min_best_iter: int = 2

    # Drop folds with train_size below this from the mean calculation.
    # Set to 0 to disable.
    min_fold_train_size: int = 200_000

    # Reject trials where one fold's per-fold metric is > mean(others) +
    # max_fold_dominance_z * std(others). Set to inf to disable.
    max_fold_dominance_z: float = 2.5


def _objective_metric(
    cv_result: dict,
    objective: str,
    constraints: TuneConstraints | None = None,
) -> float:
    """Compute the constrained Optuna objective.

    Returns -inf when any constraint is violated (so TPE learns to avoid
    that region). Otherwise returns the mean metric across the *retained*
    folds.
    """
    fold_metrics = cv_result.get("fold_metrics", [])
    if not fold_metrics:
        return float("nan")

    metric_key = (
        "mean_daily_rank_ic" if objective == "regression" else "val_accuracy"
    )

    # Drop folds smaller than min_fold_train_size (Phase-A1 constraint #2).
    if constraints is not None and constraints.min_fold_train_size > 0:
        retained = [
            m for m in fold_metrics
            if m.get("train_size", 0) >= constraints.min_fold_train_size
        ]
    else:
        retained = list(fold_metrics)
    if not retained:
        return float("-inf")

    # Constraint #1 — every retained fold must train for ≥ min_best_iter.
    if constraints is not None and constraints.min_best_iter > 1:
        too_short = [m for m in retained if m.get("best_iter", 0) < constraints.min_best_iter]
        if too_short:
            return float("-inf")

    values = [m.get(metric_key, float("nan")) for m in retained]
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) == 0:
        return float("-inf")

    # Constraint #3 — fold-dominance check.
    if (
        constraints is not None
        and np.isfinite(constraints.max_fold_dominance_z)
        and len(finite) >= 3
    ):
        best = max(finite)
        others = [v for v in finite if v != best]
        # If multiple folds tie for best, pick the index of any of them.
        if others:
            mu = float(np.mean(others))
            sigma = float(np.std(others, ddof=1)) if len(others) > 1 else 0.0
            if sigma > 0 and (best - mu) / sigma > constraints.max_fold_dominance_z:
                return float("-inf")

    return float(np.mean(finite))


def tune_hyperparameters(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    splitter: PurgedWalkForwardSplit,
    base_config: TrainConfig,
    *,
    n_trials: int = 50,
    timeout_seconds: int | None = 3600,
    study_name: str = "ta_agent",
    storage_path: str | None = None,
    constraints: TuneConstraints | None = None,
    seeds: tuple[int, ...] | None = None,
    n_jobs: int = 1,
) -> tuple[TrainConfig, optuna.Study]:
    """Run Optuna search; return (best_config, study).

    ``constraints`` is opt-in for Phase-A1 trial gating. Pass an instance
    of ``TuneConstraints`` to enforce min_best_iter, min_fold_train_size,
    and max_fold_dominance_z. Pass ``None`` for the legacy unconstrained
    objective (used by older tests).

    ``seeds`` (Phase E): when provided, each trial trains the model under
    EVERY seed in the tuple and the objective is the MEAN across seeds.
    This breaks the lucky-seed pathology where a single-seed objective
    optimized hyperparams that worked great for seed 42 and collapsed on
    seeds 43+44. Each multi-seed trial is N× slower than single-seed.
    Pass ``None`` (default) for the legacy single-seed behavior.
    """

    def _run_one_trial(cfg: TrainConfig) -> dict:
        try:
            return train_with_cv(df, feature_cols, label_col, splitter, cfg)
        except Exception as exc:  # noqa: BLE001 — Optuna trial failure should not abort study
            log.warning(f"trial cfg failed: {exc!r}")
            return {"fold_metrics": [], "mean_metrics": {}}

    def _objective(trial: optuna.Trial) -> float:
        cfg = replace(
            base_config,
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 500),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        )

        if seeds is None:
            cv = _run_one_trial(cfg)
            return _objective_metric(cv, base_config.objective, constraints)

        # Multi-seed: average objective across seeds. If ANY seed produces
        # -inf (constraint violation), the whole trial is rejected.
        per_seed_objective: list[float] = []
        for s in seeds:
            seed_cfg = replace(cfg, seed=s)
            cv = _run_one_trial(seed_cfg)
            obj = _objective_metric(cv, base_config.objective, constraints)
            if not np.isfinite(obj):
                # Single-seed constraint violation is enough to reject the trial.
                return float("-inf")
            per_seed_objective.append(obj)

        mean_obj = float(np.mean(per_seed_objective))
        # Stash per-seed values in the trial's user_attrs for inspection.
        trial.set_user_attr("per_seed_objectives", per_seed_objective)
        trial.set_user_attr("min_seed_objective", float(min(per_seed_objective)))
        return mean_obj

    sampler = optuna.samplers.TPESampler(seed=base_config.seed)
    storage = f"sqlite:///{storage_path}" if storage_path else None
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        gc_after_trial=True,
        n_jobs=n_jobs,
    )

    try:
        best_val = study.best_value
    except ValueError:
        # No trial completed (e.g. CV splitter raised on every trial because
        # train data is too small for the configured min_train_size_days).
        # Falling back to base_config is safer than failing the whole retrain.
        log.warning(
            "Optuna: no trials completed — likely CV/data issue. "
            "Returning base_config unchanged."
        )
        return base_config, study
    if not np.isfinite(best_val):
        log.warning("Optuna found no finite-objective trial; returning base_config unchanged")
        return base_config, study

    best_cfg = replace(base_config, **study.best_params)
    return best_cfg, study
