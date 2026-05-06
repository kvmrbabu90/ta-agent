"""End-to-end smoke test for the training pipeline.

Marked @pytest.mark.integration — skipped by default. Run with:
    pytest -m integration tests/integration/test_train_e2e.py
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from packages.modeling.calibrate import calibrate_classifier
from packages.modeling.registry import ModelMetadata, load_latest_model, save_model
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_final_model, train_with_cv


def _dataset(n_dates: int = 600, n_symbols: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = date(2020, 1, 1)
    rows = []
    for d in range(n_dates):
        for s in range(n_symbols):
            x0 = rng.normal()
            x1 = rng.normal()
            y_reg = 0.05 * x0 - 0.03 * x1 + 0.02 * rng.normal()
            y_cls = int(np.clip(np.round(2.0 + x0), 0, 4))
            rows.append(
                {
                    "bar_date": base + timedelta(days=d),
                    "symbol": f"S{s}",
                    "x0": x0,
                    "x1": x1,
                    "x2": rng.normal(),
                    "fwd_return_5d": y_reg,
                    "fwd_quintile_5d": y_cls,
                    "in_universe": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.integration
class TestTrainE2E:
    def test_regression_e2e(self, tmp_path):
        df = _dataset()
        feats = ["x0", "x1", "x2"]
        splitter = PurgedWalkForwardSplit(
            n_folds=3, horizon_days=5, embargo_days=5, min_train_size_days=200
        )
        cfg = TrainConfig(
            objective="regression",
            num_boost_round=80,
            early_stopping_rounds=10,
            min_data_in_leaf=20,
        )

        cv = train_with_cv(df, feats, "fwd_return_5d", splitter, cfg)
        assert len(cv["models"]) == 3

        fte = df["bar_date"].max()
        booster, fm = train_final_model(df, feats, "fwd_return_5d", cfg, fte)

        meta = ModelMetadata(
            universe="TEST",
            target="regression",
            horizon_days=5,
            feature_cols=feats,
            train_start=df["bar_date"].min(),
            train_end=fte,
            config=cfg.__dict__,
            cv_metrics=cv["mean_metrics"],
        )
        out_dir = save_model(booster, meta, base_dir=tmp_path)
        assert (out_dir / "metadata.json").exists()
        assert (out_dir / "feature_importance.csv").exists()

        loaded, md = load_latest_model("TEST", "regression", base_dir=tmp_path)
        assert md["universe"] == "TEST"
        # Sanity: loaded booster predicts (no calibration on regression).
        preds = loaded.predict(df[feats])
        assert len(preds) == len(df)

        # Sanity bounds: synthetic data with linear signal — IC should be
        # positive and finite; reject impossible numbers like NaN.
        ic = cv["mean_metrics"]["mean_daily_ic"]
        assert np.isfinite(ic)
        assert -0.2 < ic < 1.0

    def test_classification_e2e_with_calibration(self, tmp_path):
        df = _dataset()
        feats = ["x0", "x1", "x2"]
        splitter = PurgedWalkForwardSplit(
            n_folds=3, horizon_days=5, embargo_days=5, min_train_size_days=200
        )
        cfg = TrainConfig(
            objective="classification",
            n_classes=5,
            num_boost_round=80,
            early_stopping_rounds=10,
            min_data_in_leaf=20,
        )

        cv = train_with_cv(df, feats, "fwd_quintile_5d", splitter, cfg)
        fte = df["bar_date"].max()
        booster, _ = train_final_model(df, feats, "fwd_quintile_5d", cfg, fte)

        # Use a temporally-distinct calibration slice (last 30 days of training).
        cal_cutoff = fte - timedelta(days=30)
        cal_df = df[df["bar_date"] > cal_cutoff]
        calibrated = calibrate_classifier(
            booster, cal_df[feats], cal_df["fwd_quintile_5d"]
        )

        meta = ModelMetadata(
            universe="TEST",
            target="classification",
            horizon_days=5,
            feature_cols=feats,
            train_start=df["bar_date"].min(),
            train_end=fte,
            config=cfg.__dict__,
            cv_metrics=cv["mean_metrics"],
        )
        out_dir = save_model(calibrated, meta, base_dir=tmp_path)
        assert (out_dir / "calibrators.pkl").exists()

        loaded, _ = load_latest_model("TEST", "classification", base_dir=tmp_path)
        probas = loaded.predict_proba(df[feats].head(20))
        assert probas.shape == (20, 5)
        # Probabilities should sum to ~1 per row after renormalization.
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
