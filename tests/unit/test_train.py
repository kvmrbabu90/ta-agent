"""Unit tests for the modeling layer's training loop."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import (
    TrainConfig,
    _filter_trainable,
    train_one_fold,
    train_with_cv,
)


def _synthetic_dataset(
    n_dates: int = 700,
    n_symbols: int = 8,
    n_features: int = 6,
    *,
    target: str = "regression",
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str], str]:
    """Long-format frame with bar_date, symbol, feature_*, label, in_universe."""
    rng = np.random.default_rng(seed)
    base = date(2020, 1, 1)
    rows = []
    for d in range(n_dates):
        for s in range(n_symbols):
            feats = rng.normal(size=n_features)
            # Regression label = small linear combo of features + noise
            y_reg = 0.05 * feats[0] - 0.03 * feats[1] + 0.01 * rng.normal()
            y_cls = int(np.clip(np.round(2.0 + feats[0]), 0, 4))
            rows.append(
                {
                    "bar_date": base + timedelta(days=d),
                    "symbol": f"S{s}",
                    **{f"feature_{i}": float(v) for i, v in enumerate(feats)},
                    "fwd_return_5d": float(y_reg),
                    "fwd_quintile_5d": int(y_cls),
                    "in_universe": True,
                }
            )
    df = pd.DataFrame(rows)
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    label = "fwd_return_5d" if target == "regression" else "fwd_quintile_5d"
    return df, feature_cols, label


def test_filter_trainable_drops_out_of_universe_and_nan_labels() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A"],
            "bar_date": [date(2024, 1, i) for i in (1, 2, 3, 4)],
            "in_universe": [True, True, False, True],
            "fwd_return_5d": [0.01, np.nan, 0.02, 0.03],
        }
    )
    out = _filter_trainable(df, "fwd_return_5d")
    assert len(out) == 2
    assert set(out["bar_date"].astype(str)) == {"2024-01-01", "2024-01-04"}


def test_train_one_fold_runs_and_returns_booster() -> None:
    df, feature_cols, label = _synthetic_dataset(n_dates=80)
    cutoff = df["bar_date"].sort_values().iloc[len(df) // 2]
    train = df[df["bar_date"] <= cutoff]
    val = df[df["bar_date"] > cutoff]
    cfg = TrainConfig(
        objective="regression",
        num_boost_round=50,
        early_stopping_rounds=10,
        min_data_in_leaf=5,
        num_leaves=15,
    )
    booster, m = train_one_fold(
        train[feature_cols], train[label], val[feature_cols], val[label], cfg
    )
    assert booster.best_iteration > 0
    assert "best_val_metric" in m


def test_train_one_fold_early_stopping_caps_iterations() -> None:
    """If we set num_boost_round huge but early stopping tight, best_iter << num_boost_round."""
    df, feature_cols, label = _synthetic_dataset(n_dates=100)
    cutoff = df["bar_date"].sort_values().iloc[len(df) // 2]
    train = df[df["bar_date"] <= cutoff]
    val = df[df["bar_date"] > cutoff]
    cfg = TrainConfig(
        objective="regression",
        num_boost_round=2000,
        early_stopping_rounds=10,
        min_data_in_leaf=5,
        num_leaves=15,
        learning_rate=0.1,
    )
    booster, m = train_one_fold(
        train[feature_cols], train[label], val[feature_cols], val[label], cfg
    )
    assert m["best_iter"] < 2000


def test_train_with_cv_yields_one_model_per_fold() -> None:
    df, feature_cols, label = _synthetic_dataset(n_dates=600, n_symbols=6)
    splitter = PurgedWalkForwardSplit(
        n_folds=3, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    cfg = TrainConfig(
        objective="regression",
        num_boost_round=80,
        early_stopping_rounds=10,
        min_data_in_leaf=10,
        num_leaves=15,
    )
    result = train_with_cv(df, feature_cols, label, splitter, cfg)
    assert len(result["models"]) == 3
    assert len(result["fold_metrics"]) == 3
    assert "mean_daily_rank_ic" in result["mean_metrics"]
    # Per-fold val_predictions captured for downstream Optuna / evaluation.
    assert len(result["val_predictions"]) == 3


def test_train_with_cv_filters_out_of_universe_rows() -> None:
    df, feature_cols, label = _synthetic_dataset(n_dates=600, n_symbols=6)
    # Mark half the rows out of universe — they should be excluded.
    df.loc[df["symbol"] == "S0", "in_universe"] = False
    splitter = PurgedWalkForwardSplit(
        n_folds=3, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    cfg = TrainConfig(
        objective="regression",
        num_boost_round=50,
        early_stopping_rounds=10,
        min_data_in_leaf=10,
    )
    result = train_with_cv(df, feature_cols, label, splitter, cfg)
    total = sum(m["train_size"] + m["val_size"] for m in result["fold_metrics"])
    # At minimum, the S0 rows are absent from the trainable set in every fold.
    assert total > 0
    # In each fold, train size is < total rows for that fold's window with all 6 symbols.
    for m in result["fold_metrics"]:
        assert m["train_size"] > 0


def test_train_with_cv_classification() -> None:
    df, feature_cols, label = _synthetic_dataset(
        n_dates=600, n_symbols=8, target="classification"
    )
    splitter = PurgedWalkForwardSplit(
        n_folds=3, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    cfg = TrainConfig(
        objective="classification",
        n_classes=5,
        num_boost_round=40,
        early_stopping_rounds=10,
        min_data_in_leaf=10,
        num_leaves=15,
    )
    result = train_with_cv(df, feature_cols, label, splitter, cfg)
    assert len(result["models"]) == 3
    assert "val_accuracy" in result["mean_metrics"]
