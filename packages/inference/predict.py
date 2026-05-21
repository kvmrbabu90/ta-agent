"""Daily prediction: load latest models, build inference features, score.

Critical contract:
    Inference features are built from the SAME ``build_feature_matrix``
    code path used at training, with end=as_of. We then take only the
    rows where bar_date == as_of. The set+order of columns passed to
    each model MUST exactly match its metadata's feature_cols.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from packages.common.config import MODELS_DIR
from packages.common.logging import log
from packages.features.pipeline import build_feature_matrix
from packages.ingestion.universe.membership import members_on
from packages.modeling.calibrate import CalibratedModel
from packages.modeling.registry import find_latest_model_dir, load_latest_model

_DEFAULT_HORIZON = 5


@dataclass
class _UniverseModels:
    universe: str
    horizon_days: int
    n_classes: int
    reg_model: lgb.Booster
    reg_meta: dict[str, Any]
    reg_version: str
    cls_model: lgb.Booster | CalibratedModel
    cls_meta: dict[str, Any]
    cls_version: str


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------


def _align_features(
    panel: pd.DataFrame, feature_cols: list[str], *, model_label: str
) -> pd.DataFrame:
    """Return a (n, len(feature_cols)) matrix in the model's expected order.
    Raises ValueError if any required feature is missing.

    Dtype-coerces any ``object``-dtype column to float64 (NaN on failure).
    This catches the NIFTY100 case where a universe has NO data for some
    optional adapter (e.g. earnings on NSE): build_feature_matrix returns
    Python ``None`` → object dtype, which LightGBM rejects, even though
    the equivalent training-parquet path stores them as float NaN. NaN is
    a valid missing-value sentinel for LightGBM.
    """
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise ValueError(
            f"feature matrix missing {len(missing)} required column(s) for "
            f"{model_label}: {missing[:10]}{'…' if len(missing) > 10 else ''}"
        )
    out = panel[feature_cols].copy()
    obj_cols = [c for c in feature_cols if out[c].dtype == object]
    for c in obj_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_inference_features(
    universe: str,
    as_of: date,
    feature_cols: list[str],
    *,
    duckdb_path: str | None = None,
    warmup_days: int = 400,
) -> pd.DataFrame:
    """Compute features through ``as_of`` and return ONE row per current member.

    A 400-day warmup window precedes ``as_of`` so all rolling features are
    populated. Symbols with no bar dated exactly ``as_of`` are silently
    dropped (likely halted / no data) — caller decides how to handle.

    Returns columns: symbol, bar_date, plus feature_cols in model order.
    """
    start = as_of - timedelta(days=warmup_days)
    panel = build_feature_matrix(universe, start, as_of, duckdb_path=duckdb_path)
    if panel.empty:
        return pd.DataFrame(columns=["symbol", "bar_date", *feature_cols])

    panel["bar_date"] = pd.to_datetime(panel["bar_date"]).dt.date
    inf = panel[panel["bar_date"] == as_of].copy()
    if inf.empty:
        log.warning(
            f"build_inference_features: no rows for as_of={as_of} in {universe}"
        )
        return pd.DataFrame(columns=["symbol", "bar_date", *feature_cols])

    aligned = _align_features(inf, feature_cols, model_label="inference")
    out = pd.DataFrame(
        {"symbol": inf["symbol"].values, "bar_date": inf["bar_date"].values}
    )
    for c in feature_cols:
        out[c] = aligned[c].values
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_models_for_universe(
    universe: str,
    *,
    base_dir: Path | str = MODELS_DIR,
) -> _UniverseModels:
    """Load the latest registered regression + classification models for ``universe``."""
    reg_dir = find_latest_model_dir(universe, "regression", base_dir)
    cls_dir = find_latest_model_dir(universe, "classification", base_dir)

    reg_model, reg_meta = load_latest_model(universe, "regression", base_dir)
    cls_model, cls_meta = load_latest_model(universe, "classification", base_dir)

    horizon = int(reg_meta.get("horizon_days", _DEFAULT_HORIZON))
    n_classes = int(cls_meta.get("config", {}).get("n_classes", 5))

    return _UniverseModels(
        universe=universe,
        horizon_days=horizon,
        n_classes=n_classes,
        reg_model=reg_model,
        reg_meta=reg_meta,
        reg_version=reg_dir.name,
        cls_model=cls_model,
        cls_meta=cls_meta,
        cls_version=cls_dir.name,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _classifier_proba(
    model: lgb.Booster | CalibratedModel, X: pd.DataFrame, n_classes: int
) -> np.ndarray:
    """Uniform proba interface across raw-Booster and CalibratedModel."""
    if isinstance(model, CalibratedModel):
        return np.asarray(model.predict_proba(X), dtype=float)
    raw = np.asarray(model.predict(X), dtype=float)
    if raw.ndim == 1:
        # Binary or 1-class collapse — treat as 2-class.
        raw = np.column_stack([1.0 - raw, raw])
    return raw


def predict_with_features(
    features: pd.DataFrame,
    models: _UniverseModels,
) -> pd.DataFrame:
    """Run regression + classification on a precomputed inference feature frame."""
    if features.empty:
        return pd.DataFrame()

    reg_cols = list(models.reg_meta["feature_cols"])
    cls_cols = list(models.cls_meta["feature_cols"])

    X_reg = _align_features(features, reg_cols, model_label="regression")
    X_cls = _align_features(features, cls_cols, model_label="classification")

    pred_return = np.asarray(models.reg_model.predict(X_reg), dtype=float)
    proba = _classifier_proba(models.cls_model, X_cls, models.n_classes)
    pred_quintile = proba.argmax(axis=1).astype(int)
    n_classes = proba.shape[1]
    top_idx = n_classes - 1
    bot_idx = 0

    return pd.DataFrame(
        {
            "universe": models.universe,
            "symbol": features["symbol"].values,
            "as_of": features["bar_date"].values,
            "horizon_days": models.horizon_days,
            "predicted_return": pred_return,
            "predicted_quintile": pred_quintile,
            "top_quintile_proba": proba[:, top_idx],
            "bottom_quintile_proba": proba[:, bot_idx],
            "model_version_regression": models.reg_version,
            "model_version_classification": models.cls_version,
        }
    )


def predict_universe(
    universe: str,
    as_of: date | None = None,
    model_dir: Path | str | None = None,
    *,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """High-level: load models, build features for current members, predict.

    Returns columns:
        universe, symbol, as_of, horizon_days, predicted_return,
        predicted_quintile, top_quintile_proba, bottom_quintile_proba,
        model_version_regression, model_version_classification.
    """
    if as_of is None:
        as_of = date.today()

    models = load_models_for_universe(
        universe, base_dir=model_dir or MODELS_DIR
    )

    # The two models may have been trained on the same feature set, but allow
    # for divergence: build the union and align per-model later.
    needed = sorted(
        set(models.reg_meta["feature_cols"]) | set(models.cls_meta["feature_cols"])
    )

    members = members_on(universe, as_of)
    if members.empty:
        log.warning(f"predict_universe: no members of {universe} on {as_of}")
        return pd.DataFrame()

    features = build_inference_features(
        universe, as_of, feature_cols=needed, duckdb_path=duckdb_path
    )
    if features.empty:
        log.warning(f"predict_universe: empty inference features for {universe}/{as_of}")
        return pd.DataFrame()

    # Drop rows that aren't current members (shouldn't happen, but defensive).
    features = features[features["symbol"].isin(members["symbol"])]
    if features.empty:
        return pd.DataFrame()

    return predict_with_features(features, models)
