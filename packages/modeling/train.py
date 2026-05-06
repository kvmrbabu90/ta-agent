"""LightGBM training: per-fold, per-CV, and final-model production builds.

Always uses the native ``lgb.train`` API (the sklearn wrapper hides too
much). Always uses the modern ``lgb.early_stopping`` callback (the
``early_stopping_rounds`` keyword arg is deprecated). Never shuffles —
time order is sacred.

Filtering rule:
    Before training, rows are filtered to `in_universe == True` AND
    label is not null. This is the only place we drop rows; downstream
    train/val splits work on the filtered, time-sorted frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from packages.common.logging import log
from packages.modeling.evaluate import evaluate_predictions
from packages.modeling.splits import PurgedWalkForwardSplit


@dataclass
class TrainConfig:
    objective: Literal["regression", "classification"] = "regression"
    n_classes: int = 5
    num_boost_round: int = 5000
    early_stopping_rounds: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 63
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_data_in_leaf: int = 100
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    seed: int = 42
    verbose: int = -1
    is_unbalance: bool = False  # classification only


def _to_lgb_params(cfg: TrainConfig) -> dict[str, Any]:
    common = {
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "lambda_l1": cfg.lambda_l1,
        "lambda_l2": cfg.lambda_l2,
        "seed": cfg.seed,
        "verbose": cfg.verbose,
    }
    if cfg.objective == "regression":
        return {**common, "objective": "regression", "metric": "l2"}
    if cfg.objective == "classification":
        return {
            **common,
            "objective": "multiclass",
            "num_class": cfg.n_classes,
            "metric": "multi_logloss",
            "is_unbalance": cfg.is_unbalance,
        }
    raise ValueError(f"unknown objective: {cfg.objective}")


def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN — LightGBM handles NaN natively."""
    X = X.replace([np.inf, -np.inf], np.nan)
    # Coerce object/bool columns to float so LightGBM accepts them.
    for c in X.columns:
        if X[c].dtype == object or X[c].dtype == bool:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _filter_trainable(
    df: pd.DataFrame, label_col: str, *, require_in_universe: bool = True
) -> pd.DataFrame:
    """Drop rows with NaN labels and (optionally) out-of-universe rows."""
    out = df
    if require_in_universe and "in_universe" in df.columns:
        out = out[out["in_universe"].fillna(False).astype(bool)]
    out = out[out[label_col].notna()]
    return out.sort_values("bar_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------


def train_one_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: TrainConfig,
) -> tuple[lgb.Booster, dict]:
    """Train a single LightGBM model on one CV fold."""
    X_train_c = _clean_features(X_train)
    X_val_c = _clean_features(X_val)

    if config.objective == "classification":
        n_seen = int(pd.Series(y_train).nunique())
        if n_seen < config.n_classes:
            log.warning(
                f"train_one_fold: only {n_seen}/{config.n_classes} classes "
                f"present in training set; LightGBM will still train but "
                f"calibration / probability metrics may degrade"
            )

    dtrain = lgb.Dataset(X_train_c, label=np.asarray(y_train), free_raw_data=False)
    dval = lgb.Dataset(
        X_val_c, label=np.asarray(y_val), reference=dtrain, free_raw_data=False
    )

    params = _to_lgb_params(config)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=config.num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_metric_name = next(iter(booster.best_score["val"].keys()))
    metrics = {
        "best_iter": int(booster.best_iteration),
        "best_val_metric_name": val_metric_name,
        "best_val_metric": float(booster.best_score["val"][val_metric_name]),
    }
    return booster, metrics


def _val_predictions(
    booster: lgb.Booster, X_val: pd.DataFrame, *, objective: str
) -> np.ndarray:
    """Predict on val. For classification returns probas (n, n_classes);
    for regression returns a 1-D array."""
    raw = booster.predict(_clean_features(X_val), num_iteration=booster.best_iteration)
    return np.asarray(raw)


# ---------------------------------------------------------------------------
# Full CV
# ---------------------------------------------------------------------------


def train_with_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    splitter: PurgedWalkForwardSplit,
    config: TrainConfig,
) -> dict[str, Any]:
    """Run purged walk-forward CV. For each fold, train + collect val metrics."""
    df_clean = _filter_trainable(df, label_col)
    if df_clean.empty:
        raise ValueError("no trainable rows after filtering (in_universe + non-null label)")

    fold_metrics: list[dict] = []
    models: list[lgb.Booster] = []
    val_predictions: list[dict] = []

    for k, (train_idx, val_idx) in enumerate(splitter.split(df_clean)):
        X_train = df_clean.iloc[train_idx][feature_cols]
        y_train = df_clean.iloc[train_idx][label_col]
        X_val = df_clean.iloc[val_idx][feature_cols]
        y_val = df_clean.iloc[val_idx][label_col]
        val_dates = df_clean.iloc[val_idx]["bar_date"]

        booster, m = train_one_fold(X_train, y_train, X_val, y_val, config)

        preds = _val_predictions(booster, X_val, objective=config.objective)

        # Compute objective-appropriate val diagnostics.
        if config.objective == "regression":
            eval_dict = evaluate_predictions(
                y_true_returns=y_val.reset_index(drop=True),
                y_pred_returns=pd.Series(preds).reset_index(drop=True),
                bar_dates=val_dates.reset_index(drop=True),
            )
            m.update(
                {
                    "fold": k,
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                    "mean_daily_ic": eval_dict["mean_daily_ic"],
                    "mean_daily_rank_ic": eval_dict["mean_daily_rank_ic"],
                    "mean_decile_spread": eval_dict["mean_decile_spread"],
                    "hit_rate": eval_dict["hit_rate"],
                    "t_stat_daily_ic": eval_dict["t_stat_daily_ic"],
                }
            )
        else:
            # Classification: track multi-class accuracy as a fold-level summary.
            class_pred = preds.argmax(axis=1) if preds.ndim == 2 else preds
            acc = float((class_pred == y_val.to_numpy().astype(int)).mean())
            m.update(
                {
                    "fold": k,
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                    "val_accuracy": acc,
                }
            )

        fold_metrics.append(m)
        models.append(booster)
        val_predictions.append(
            {
                "fold": k,
                "y_true": y_val.to_numpy(),
                "y_pred": preds,
                "bar_dates": val_dates.to_numpy(),
            }
        )

    # Aggregate.
    aggregate_keys = (
        ["mean_daily_ic", "mean_daily_rank_ic", "mean_decile_spread", "hit_rate"]
        if config.objective == "regression"
        else ["val_accuracy"]
    )
    mean_metrics = {
        k: float(np.nanmean([m.get(k, np.nan) for m in fold_metrics]))
        for k in aggregate_keys
    }
    std_metrics = {
        k: float(np.nanstd([m.get(k, np.nan) for m in fold_metrics], ddof=0))
        for k in aggregate_keys
    }

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "models": models,
        "val_predictions": val_predictions,
    }


# ---------------------------------------------------------------------------
# Final / production model
# ---------------------------------------------------------------------------


def train_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    config: TrainConfig,
    final_train_end: date,
    *,
    early_stopping_holdout_days: int = 60,
) -> tuple[lgb.Booster, dict]:
    """Train the production model on all data up through ``final_train_end``.

    Splits the very tail of that window (last ``early_stopping_holdout_days``)
    into a held-out validation set used purely for early-stopping. This is
    chronologically distinct from any later evaluation slice.
    """
    df_clean = _filter_trainable(df, label_col)
    df_clean["bar_date"] = pd.to_datetime(df_clean["bar_date"]).dt.date
    df_clean = df_clean[df_clean["bar_date"] <= final_train_end]
    if df_clean.empty:
        raise ValueError("no rows with bar_date <= final_train_end")

    es_cutoff = final_train_end - pd.Timedelta(days=early_stopping_holdout_days)
    es_cutoff_date = es_cutoff.date() if hasattr(es_cutoff, "date") else es_cutoff

    train_mask = df_clean["bar_date"] <= es_cutoff_date
    val_mask = df_clean["bar_date"] > es_cutoff_date

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError(
            f"early_stopping_holdout_days={early_stopping_holdout_days} produced "
            f"empty train or val split"
        )

    X_train = df_clean.loc[train_mask, feature_cols]
    y_train = df_clean.loc[train_mask, label_col]
    X_val = df_clean.loc[val_mask, feature_cols]
    y_val = df_clean.loc[val_mask, label_col]

    booster, metrics = train_one_fold(X_train, y_train, X_val, y_val, config)
    metrics.update(
        {
            "train_size": int(train_mask.sum()),
            "es_holdout_size": int(val_mask.sum()),
            "es_cutoff_date": es_cutoff_date.isoformat(),
            "final_train_end": final_train_end.isoformat(),
            "config": asdict(config),
        }
    )
    return booster, metrics


__all__ = [
    "TrainConfig",
    "train_one_fold",
    "train_with_cv",
    "train_final_model",
    "_filter_trainable",
    "_clean_features",
    "_to_lgb_params",
    "replace",  # re-export from dataclasses for tune.py convenience
]
