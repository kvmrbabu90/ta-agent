"""SHAP feature attributions for daily predictions.

Returns the top-K features per prediction by absolute SHAP value, with
the feature value, the SHAP contribution, and a sign tag.

For multi-class classifiers we explain the TOP-quintile class (n_classes-1)
since that's the class our high-confidence-long picks key off. Regression
boosters get a single SHAP array.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from packages.modeling.calibrate import CalibratedModel
from packages.modeling.train import _clean_features


def _underlying_booster(model: lgb.Booster | CalibratedModel) -> lgb.Booster:
    if isinstance(model, CalibratedModel):
        return model.booster
    return model


def _explainer(model: lgb.Booster | CalibratedModel) -> shap.TreeExplainer:
    return shap.TreeExplainer(_underlying_booster(model))


def explain_predictions(
    predictions: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    model: lgb.Booster | CalibratedModel,
    feature_cols: list[str],
    *,
    top_k: int = 5,
    explain_class: int | None = None,
) -> pd.DataFrame:
    """Compute top-K SHAP attributions for each row in ``predictions``.

    feature_matrix must contain the same rows as predictions in the same
    order, with at least the columns named in feature_cols. The function
    aligns to feature_cols and runs TreeExplainer in a single call.

    For classification the SHAP values are explained for ``explain_class``
    (default = highest class index, i.e. the top quintile).
    """
    if predictions.empty or feature_matrix.empty:
        return pd.DataFrame(
            columns=[
                "symbol", "as_of", "rank", "feature_name",
                "feature_value", "shap_value", "contribution_direction",
            ]
        )
    if len(predictions) != len(feature_matrix):
        raise ValueError(
            f"predictions ({len(predictions)}) and feature_matrix "
            f"({len(feature_matrix)}) must have the same length"
        )

    X = _clean_features(feature_matrix[feature_cols])
    explainer = _explainer(model)
    shap_values: Any = explainer.shap_values(X)

    if isinstance(shap_values, list):
        # Multi-class: pick the requested class.
        klass = (
            explain_class if explain_class is not None else len(shap_values) - 1
        )
        klass = max(0, min(klass, len(shap_values) - 1))
        sv_arr = np.asarray(shap_values[klass], dtype=float)
    else:
        sv_arr = np.asarray(shap_values, dtype=float)
        if sv_arr.ndim == 3:
            # New SHAP API may return (n, n_features, n_classes) for multiclass.
            klass = (
                explain_class if explain_class is not None else sv_arr.shape[2] - 1
            )
            klass = max(0, min(klass, sv_arr.shape[2] - 1))
            sv_arr = sv_arr[:, :, klass]

    out_rows: list[dict] = []
    feat_names = list(feature_cols)
    for i, (_, pred_row) in enumerate(predictions.iterrows()):
        sv = sv_arr[i]
        order = np.argsort(-np.abs(sv))[:top_k]
        for rank, j in enumerate(order, start=1):
            fname = feat_names[j]
            shap_val = float(sv[j])
            feat_val = X.iloc[i, j]
            out_rows.append(
                {
                    "symbol": pred_row["symbol"],
                    "as_of": pred_row["as_of"],
                    "rank": rank,
                    "feature_name": fname,
                    "feature_value": (None if pd.isna(feat_val) else float(feat_val)),
                    "shap_value": shap_val,
                    "contribution_direction": "positive" if shap_val >= 0 else "negative",
                }
            )

    return pd.DataFrame(out_rows)
