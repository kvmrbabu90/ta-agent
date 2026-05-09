"""Train final regression + classification models on the +earnings panel,
using the prior production hyperparams (no Optuna re-tune).

Why no re-tune: a controlled re-tune with the new earnings features picked
hyperparams that overfit fold 0 (best_iter=1, fold-0 rank-IC=0.19, far above
the README's 'suspicious' threshold). The prior hyperparams are known-stable
across multiple validation runs, so reusing them with the new feature set
ships the validated +earnings improvement (positive decile-spread sign-flip
on 2/3 seeds, +0.14 pp hit rate on 3/3 seeds) without the re-tune surprise.

Outputs:
  data/models/SP500_regression_<ts>/    final regression model + CV metrics
  data/models/SP500_classification_<ts>/ final classification model (calibrated)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
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

NON_FEAT = {"symbol", "bar_date", "in_universe", "fwd_return_5d", "fwd_quintile_5d"}

# Prior production hyperparams (lifted from the deployed models — see report).
_REG_HP = {
    "learning_rate": 0.017668989141957626,
    "num_leaves": 117,
    "feature_fraction": 0.8624490178509115,
    "bagging_fraction": 0.755888486511419,
    "min_data_in_leaf": 79,
    "lambda_l1": 4.099702482742569,
    "lambda_l2": 4.742843845050359e-08,
    "seed": 42,
    "is_unbalance": False,
}

_CLF_HP = {
    "learning_rate": 0.03217559775850123,
    "num_leaves": 66,
    "feature_fraction": 0.5702405360582566,
    "bagging_fraction": 0.9258229964032386,
    "min_data_in_leaf": 103,
    "lambda_l1": 6.761799315301321e-07,
    "lambda_l2": 8.163996912176187e-07,
    "seed": 42,
    "is_unbalance": False,
}


def _train_one(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    universe: str,
    fte: date,
    splitter: PurgedWalkForwardSplit,
    early_stopping_holdout_days: int,
    calibration_holdout_days: int,
) -> Path:
    label_col = "fwd_return_5d" if target == "regression" else "fwd_quintile_5d"
    hp = _REG_HP if target == "regression" else _CLF_HP
    cfg = TrainConfig(
        objective=target,
        n_classes=5,
        learning_rate=hp["learning_rate"],
        num_leaves=hp["num_leaves"],
        feature_fraction=hp["feature_fraction"],
        bagging_fraction=hp["bagging_fraction"],
        min_data_in_leaf=hp["min_data_in_leaf"],
        lambda_l1=hp["lambda_l1"],
        lambda_l2=hp["lambda_l2"],
        seed=hp["seed"],
        is_unbalance=hp["is_unbalance"],
    )

    log.info(f"running CV ({target}) with prior hyperparams")
    cv = train_with_cv(df, feature_cols, label_col, splitter, cfg)
    log.info(f"CV mean: {cv['mean_metrics']}")

    log.info(f"training final {target} model with final_train_end={fte}")
    booster, fm = train_final_model(
        df, feature_cols, label_col, cfg, fte,
        early_stopping_holdout_days=early_stopping_holdout_days,
    )
    log.info(f"final {target}: best_iter={fm['best_iter']} train_size={fm['train_size']}")

    model_to_save = booster
    if target == "classification":
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
            x_cal = df_local.loc[cal_mask, feature_cols]
            y_cal = df_local.loc[cal_mask, label_col].astype(int)
            model_to_save = calibrate_classifier(booster, x_cal, y_cal)
            log.info(f"calibrated classifier on {cal_mask.sum()} rows")

    df_dates = pd.to_datetime(df["bar_date"]).dt.date
    metadata = ModelMetadata(
        universe=universe,
        target=target,
        horizon_days=5,
        feature_cols=feature_cols,
        train_start=df_dates.min(),
        train_end=fte,
        config=asdict(cfg),
        cv_metrics={
            "mean": cv["mean_metrics"],
            "std": cv["std_metrics"],
            "fold_metrics": [
                {k: v for k, v in m.items() if k != "best_val_metric_name"}
                for m in cv["fold_metrics"]
            ],
        },
    )
    out_dir = save_model(model_to_save, metadata, base_dir=MODELS_DIR)
    log.info(f"saved {target} model to {out_dir}")
    return out_dir


@click.command()
@click.option("--universe", default="SP500", show_default=True)
@click.option(
    "--dataset",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/processed/training_sp500_2010-01-01_2026-05-06.parquet"),
    show_default=True,
)
@click.option("--final-train-end", default="2026-03-09", show_default=True)
def main(universe: str, dataset: Path, final_train_end: str) -> None:
    df = pd.read_parquet(dataset)
    feature_cols = [c for c in df.columns if c not in NON_FEAT]
    n_earn = sum(1 for c in feature_cols if c.startswith("earnings__"))
    log.info(f"{len(feature_cols)} features ({n_earn} earnings)")

    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=504
    )
    fte = date.fromisoformat(final_train_end)

    reg_dir = _train_one(
        df, feature_cols, "regression", universe, fte, splitter,
        early_stopping_holdout_days=60, calibration_holdout_days=30,
    )
    clf_dir = _train_one(
        df, feature_cols, "classification", universe, fte, splitter,
        early_stopping_holdout_days=60, calibration_holdout_days=30,
    )

    click.echo("\n=== SUMMARY ===")
    for label, p in [("regression", reg_dir), ("classification", clf_dir)]:
        md = json.loads((p / "metadata.json").read_text())
        click.echo(f"  {label:14s} {p.name}")
        click.echo(f"    cv mean: {md['cv_metrics']['mean']}")


if __name__ == "__main__":
    main()
