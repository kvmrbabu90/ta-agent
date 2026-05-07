"""End-to-end training CLI.

    python -m scripts.train_models --universe SP500 --target regression \\
        --dataset data/processed/training_sp500.parquet --tune --n-trials 50

Steps:
    1. Load the dataset parquet (output of scripts.build_dataset).
    2. Run baseline purged walk-forward CV with default config.
    3. If --tune, run Optuna -> get best config -> rerun CV with it.
    4. Train the final production model on all data up to (today - 60d).
    5. For classification, calibrate on a held-out tail slice.
    6. Save model + metadata via the registry; print a summary.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import pandas as pd

from packages.common.config import MODELS_DIR
from packages.common.logging import log
from packages.modeling.calibrate import calibrate_classifier
from packages.modeling.registry import ModelMetadata, save_model
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import (
    TrainConfig,
    train_final_model,
    train_with_cv,
)
from packages.modeling.tune import tune_hyperparameters

_NON_FEATURE_COLS = {
    "symbol", "bar_date", "in_universe", "fwd_return_5d", "fwd_quintile_5d",
}


@click.command()
@click.option("--universe", default="SP500", show_default=True)
@click.option(
    "--target",
    type=click.Choice(["regression", "classification"]),
    required=True,
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--horizon", default=5, show_default=True, type=int)
@click.option("--n-classes", default=5, show_default=True, type=int)
@click.option("--n-folds", default=5, show_default=True, type=int)
@click.option("--embargo-days", default=5, show_default=True, type=int)
@click.option("--min-train-days", default=504, show_default=True, type=int)
@click.option("--tune", is_flag=True, help="Run Optuna search.")
@click.option("--n-trials", default=50, show_default=True, type=int)
@click.option(
    "--tune-timeout-seconds",
    default=3600,
    show_default=True,
    type=int,
    help="Wall-clock cap for the Optuna study. Use 0 to disable (only n_trials limits).",
)
@click.option(
    "--final-train-end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Default: today - 60 days.",
)
@click.option(
    "--early-stopping-holdout-days", default=60, show_default=True, type=int
)
@click.option(
    "--calibration-holdout-days",
    default=30,
    show_default=True,
    type=int,
    help="Classification only: last N days of training reserved for calibration.",
)
@click.option(
    "--output-base",
    type=click.Path(file_okay=False, path_type=Path),
    default=MODELS_DIR,
    show_default=True,
)
def main(
    universe: str,
    target: str,
    dataset: Path,
    horizon: int,
    n_classes: int,
    n_folds: int,
    embargo_days: int,
    min_train_days: int,
    tune: bool,
    n_trials: int,
    tune_timeout_seconds: int,
    final_train_end: datetime | None,
    early_stopping_holdout_days: int,
    calibration_holdout_days: int,
    output_base: Path,
) -> None:
    log.info(f"loading dataset {dataset}")
    df = pd.read_parquet(dataset)

    label_col = (
        f"fwd_return_{horizon}d" if target == "regression" else f"fwd_quintile_{horizon}d"
    )
    if label_col not in df.columns:
        raise click.UsageError(f"dataset is missing label column: {label_col}")

    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    log.info(f"features: {len(feature_cols)} | label: {label_col}")

    splitter = PurgedWalkForwardSplit(
        n_folds=n_folds,
        horizon_days=horizon,
        embargo_days=embargo_days,
        min_train_size_days=min_train_days,
    )

    base_cfg = TrainConfig(objective=target, n_classes=n_classes)

    log.info("running baseline CV")
    baseline = train_with_cv(df, feature_cols, label_col, splitter, base_cfg)
    log.info(f"baseline mean metrics: {baseline['mean_metrics']}")

    final_cfg = base_cfg
    final_cv = baseline
    if tune:
        timeout = None if tune_timeout_seconds <= 0 else tune_timeout_seconds
        log.info(
            f"running Optuna with n_trials={n_trials} "
            f"timeout={'unlimited' if timeout is None else f'{timeout}s'}"
        )
        final_cfg, study = tune_hyperparameters(
            df,
            feature_cols,
            label_col,
            splitter,
            base_cfg,
            n_trials=n_trials,
            timeout_seconds=timeout,
            study_name=f"{universe}_{target}",
        )
        log.info(f"best params: {study.best_params}")
        final_cv = train_with_cv(df, feature_cols, label_col, splitter, final_cfg)
        log.info(f"tuned mean metrics: {final_cv['mean_metrics']}")

    fte = (final_train_end.date() if final_train_end else date.today() - timedelta(days=60))
    log.info(f"training final model with final_train_end={fte}")
    final_booster, fm = train_final_model(
        df,
        feature_cols,
        label_col,
        final_cfg,
        fte,
        early_stopping_holdout_days=early_stopping_holdout_days,
    )
    log.info(f"final model: best_iter={fm['best_iter']} train_size={fm['train_size']}")

    model_to_save = final_booster
    if target == "classification":
        # Calibrate on a slice strictly before the early-stopping holdout window.
        es_cutoff = fte - timedelta(days=early_stopping_holdout_days)
        cal_start = es_cutoff - timedelta(days=calibration_holdout_days)
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
            model_to_save = calibrate_classifier(final_booster, X_cal, y_cal)
            log.info(f"calibrated classifier on {cal_mask.sum()} rows")
        else:
            log.warning(
                "calibration window has 0 eligible rows; saving uncalibrated booster"
            )

    df_dates = pd.to_datetime(df["bar_date"]).dt.date
    metadata = ModelMetadata(
        universe=universe,
        target=target,
        horizon_days=horizon,
        feature_cols=feature_cols,
        train_start=df_dates.min(),
        train_end=fte,
        config=asdict(final_cfg),
        cv_metrics={
            "mean": final_cv["mean_metrics"],
            "std": final_cv["std_metrics"],
            "fold_metrics": [
                {k: v for k, v in m.items() if k != "best_val_metric_name"}
                for m in final_cv["fold_metrics"]
            ],
        },
    )

    out_dir = save_model(model_to_save, metadata, base_dir=output_base)
    click.echo(f"Saved model to: {out_dir}")
    click.echo(f"CV mean metrics: {final_cv['mean_metrics']}")


if __name__ == "__main__":
    main()
