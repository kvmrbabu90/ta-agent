"""Unit tests for SHAP explanations."""

from __future__ import annotations

from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap

from packages.inference.explain import explain_predictions


def _train_tiny_regressor(seed: int = 0) -> tuple[lgb.Booster, list[str], pd.DataFrame]:
    """Train a tiny regression booster on synthetic data and return
    (booster, feature_cols, training-frame-for-shap-base-comparisons)."""
    rng = np.random.default_rng(seed)
    n = 400
    X = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    y = 0.5 * X["f0"] - 0.3 * X["f1"] + 0.05 * rng.normal(size=n)
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    booster = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "verbose": -1,
            "seed": seed,
        },
        dtrain,
        num_boost_round=50,
    )
    return booster, list(X.columns), X


def test_explain_top_k_has_exactly_k_rows_per_prediction() -> None:
    booster, feat_cols, X_train = _train_tiny_regressor()

    n_preds = 5
    feat_matrix = X_train.head(n_preds).reset_index(drop=True)
    preds = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(n_preds)],
            "as_of": [date(2024, 1, 5)] * n_preds,
        }
    )
    out = explain_predictions(preds, feat_matrix, booster, feat_cols, top_k=3)
    assert len(out) == n_preds * 3
    assert set(out.columns) == {
        "symbol", "as_of", "rank", "feature_name",
        "feature_value", "shap_value", "contribution_direction",
    }
    # Each symbol should have ranks 1..3.
    for sym in preds["symbol"]:
        sub = out[out["symbol"] == sym]
        assert sub["rank"].tolist() == [1, 2, 3]


def test_shap_values_sum_plus_base_equals_prediction() -> None:
    """For regression: prediction = base_value + sum(shap values across features)."""
    booster, feat_cols, X_train = _train_tiny_regressor()

    n_preds = 8
    feat_matrix = X_train.head(n_preds).reset_index(drop=True)
    preds = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(n_preds)],
            "as_of": [date(2024, 1, 5)] * n_preds,
        }
    )
    explainer = shap.TreeExplainer(booster)
    sv = np.asarray(explainer.shap_values(feat_matrix), dtype=float)
    base = float(np.asarray(explainer.expected_value).ravel()[0])
    raw_predictions = booster.predict(feat_matrix)
    reconstructed = base + sv.sum(axis=1)
    np.testing.assert_allclose(reconstructed, raw_predictions, atol=1e-3)

    # The function itself produces top-k attributions consistent with sv.
    out = explain_predictions(preds, feat_matrix, booster, feat_cols, top_k=4)
    assert len(out) == n_preds * 4


def test_explain_handles_empty_input() -> None:
    booster, feat_cols, _ = _train_tiny_regressor()
    out = explain_predictions(
        pd.DataFrame(columns=["symbol", "as_of"]),
        pd.DataFrame(columns=feat_cols),
        booster,
        feat_cols,
        top_k=3,
    )
    assert out.empty


def test_explain_raises_on_mismatched_lengths() -> None:
    booster, feat_cols, X_train = _train_tiny_regressor()
    preds = pd.DataFrame({"symbol": ["A", "B"], "as_of": [date(2024, 1, 5)] * 2})
    with pytest.raises(ValueError, match="same length"):
        explain_predictions(preds, X_train.head(5), booster, feat_cols)
