"""Unit tests for predict_universe and feature alignment."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from packages.inference import predict as predict_mod
from packages.inference.predict import (
    _align_features,
    _UniverseModels,
    predict_universe,
    predict_with_features,
)

# ---------------------------------------------------------------------------
# _align_features (the leakage-prevention check)
# ---------------------------------------------------------------------------


def test_align_features_reorders_columns() -> None:
    panel = pd.DataFrame({"c": [1, 2], "a": [3, 4], "b": [5, 6]})
    out = _align_features(panel, ["a", "b", "c"], model_label="test")
    assert list(out.columns) == ["a", "b", "c"]
    assert out["a"].tolist() == [3, 4]


def test_align_features_raises_on_missing() -> None:
    panel = pd.DataFrame({"a": [1], "c": [2]})
    with pytest.raises(ValueError, match="missing 1 required"):
        _align_features(panel, ["a", "b", "c"], model_label="test")


# ---------------------------------------------------------------------------
# predict_with_features (model orchestration without I/O)
# ---------------------------------------------------------------------------


def _fake_models(reg_features=("a", "b"), cls_features=("a", "b"), n_classes=5):
    reg = MagicMock(name="reg_booster")
    reg.predict.return_value = np.array([0.01, -0.02])

    cls = MagicMock(name="cls_booster")
    # Multiclass predict returns (n, n_classes) probas.
    proba = np.tile([0.1, 0.1, 0.1, 0.2, 0.5], (2, 1))
    proba[1] = [0.5, 0.2, 0.1, 0.1, 0.1]
    cls.predict.return_value = proba

    return _UniverseModels(
        universe="SP500",
        horizon_days=5,
        n_classes=n_classes,
        reg_model=reg,
        reg_meta={"feature_cols": list(reg_features), "horizon_days": 5},
        reg_version="SP500_regression_20240101_120000",
        cls_model=cls,
        cls_meta={
            "feature_cols": list(cls_features),
            "config": {"n_classes": n_classes},
        },
        cls_version="SP500_classification_20240101_120000",
    )


def test_predict_with_features_returns_canonical_schema() -> None:
    models = _fake_models()
    features = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "bar_date": [date(2024, 1, 5), date(2024, 1, 5)],
            "a": [0.1, 0.2],
            "b": [0.3, 0.4],
        }
    )
    out = predict_with_features(features, models)
    expected_cols = {
        "universe", "symbol", "as_of", "horizon_days",
        "predicted_return", "predicted_quintile",
        "top_quintile_proba", "bottom_quintile_proba",
        "model_version_regression", "model_version_classification",
    }
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == 2
    assert (out["universe"] == "SP500").all()
    # Top-quintile proba pulled from class index 4.
    assert out["top_quintile_proba"].iloc[0] == pytest.approx(0.5)
    # AAA's argmax was class 4 (top), BBB's was class 0 (bottom).
    assert out["predicted_quintile"].iloc[0] == 4
    assert out["predicted_quintile"].iloc[1] == 0


def test_predict_with_features_passes_columns_in_model_order() -> None:
    """If model expects [a, b] but feature matrix has [b, a, c], we still pass [a, b]."""
    models = _fake_models(reg_features=("a", "b"), cls_features=("a", "b"))
    features = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "bar_date": [date(2024, 1, 5)] * 2,
            "b": [0.3, 0.4],
            "a": [0.1, 0.2],
            "c": [99.0, 99.0],  # extra — must be ignored
        }
    )
    predict_with_features(features, models)

    reg_call_X = models.reg_model.predict.call_args.args[0]
    assert list(reg_call_X.columns) == ["a", "b"]
    assert reg_call_X["a"].tolist() == [0.1, 0.2]


def test_predict_with_features_raises_on_missing_feature() -> None:
    models = _fake_models(reg_features=("a", "b", "c"))
    features = pd.DataFrame(
        {"symbol": ["A"], "bar_date": [date(2024, 1, 5)], "a": [0.1], "b": [0.2]}
    )
    with pytest.raises(ValueError, match="missing"):
        predict_with_features(features, models)


# ---------------------------------------------------------------------------
# predict_universe end-to-end (heavily mocked)
# ---------------------------------------------------------------------------


def test_predict_universe_full_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _fake_models()
    monkeypatch.setattr(
        predict_mod, "load_models_for_universe", lambda u, base_dir=None: models
    )
    monkeypatch.setattr(
        predict_mod, "members_on",
        lambda universe, as_of: pd.DataFrame({"symbol": ["AAA", "BBB"]}),
    )

    def _fake_features(universe, as_of, feature_cols, *, duckdb_path=None, warmup_days=400):
        return pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "bar_date": [as_of, as_of],
                "a": [0.1, 0.2],
                "b": [0.3, 0.4],
            }
        )

    monkeypatch.setattr(predict_mod, "build_inference_features", _fake_features)

    out = predict_universe("SP500", as_of=date(2024, 1, 5))
    assert len(out) == 2
    assert (out["as_of"] == date(2024, 1, 5)).all()
    assert "predicted_return" in out.columns


def test_predict_universe_empty_on_no_members(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _fake_models()
    monkeypatch.setattr(
        predict_mod, "load_models_for_universe", lambda u, base_dir=None: models
    )
    monkeypatch.setattr(
        predict_mod, "members_on", lambda universe, as_of: pd.DataFrame()
    )
    out = predict_universe("SP500", as_of=date(2024, 1, 5))
    assert out.empty
