"""Optuna-based hyperparameter tuning.

Search space:
    learning_rate, num_leaves, feature_fraction, bagging_fraction,
    min_data_in_leaf, lambda_l1, lambda_l2.

Objective:
    Regression     → maximize mean daily rank-IC across CV folds.
    Classification → maximize mean fold accuracy.

The TPE sampler is seeded for reproducibility. Studies can be persisted to
SQLite for resumption (``storage_path``); without it the study is in-memory.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import optuna
import pandas as pd

from packages.common.logging import log
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_with_cv


def _objective_metric(cv_result: dict, objective: str) -> float:
    if objective == "regression":
        return float(cv_result["mean_metrics"].get("mean_daily_rank_ic", float("nan")))
    return float(cv_result["mean_metrics"].get("val_accuracy", float("nan")))


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
) -> tuple[TrainConfig, optuna.Study]:
    """Run Optuna search; return (best_config, study)."""

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
        try:
            cv = train_with_cv(df, feature_cols, label_col, splitter, cfg)
        except Exception as exc:  # noqa: BLE001 — Optuna trial failure should not abort study
            log.warning(f"trial {trial.number} failed: {exc!r}")
            return float("-inf")
        return _objective_metric(cv, base_config.objective)

    sampler = optuna.samplers.TPESampler(seed=base_config.seed)
    storage = f"sqlite:///{storage_path}" if storage_path else None
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(_objective, n_trials=n_trials, timeout=timeout_seconds, gc_after_trial=True)

    if not np.isfinite(study.best_value):
        log.warning("Optuna found no finite-objective trial; returning base_config unchanged")
        return base_config, study

    best_cfg = replace(base_config, **study.best_params)
    return best_cfg, study
