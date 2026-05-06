"""Probability calibration for the multi-class classification model.

Per-class isotonic regression on a held-out calibration slice. The
calibration slice MUST be later in time than the training data AND
distinct from any held-out test slice. The training script enforces
the temporal ordering; this module just fits + serializes.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from packages.modeling.train import _clean_features


class CalibratedModel:
    """Wraps a multi-class booster + per-class isotonic calibrators."""

    def __init__(
        self,
        booster: lgb.Booster,
        calibrators: list[IsotonicRegression],
        n_classes: int,
    ) -> None:
        self.booster = booster
        self.calibrators = calibrators
        self.n_classes = n_classes

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(_clean_features(X))
        raw = np.asarray(raw, dtype=float)
        if raw.ndim == 1:
            # Binary-like collapse — broadcast to (n, 2). Shouldn't happen for
            # multi-class but defensive.
            raw = np.column_stack([1.0 - raw, raw])
        out = np.zeros_like(raw)
        for c, cal in enumerate(self.calibrators):
            out[:, c] = cal.predict(raw[:, c])
        # Renormalize per row.
        sums = out.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return out / sums

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"))
        with open(directory / "calibrators.pkl", "wb") as f:
            pickle.dump(
                {"calibrators": self.calibrators, "n_classes": self.n_classes}, f
            )

    @classmethod
    def load(cls, directory: Path) -> CalibratedModel:
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        with open(directory / "calibrators.pkl", "rb") as f:
            data = pickle.load(f)  # noqa: S301 — local trust boundary
        return cls(booster, data["calibrators"], data["n_classes"])


def calibrate_classifier(
    booster: lgb.Booster,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: Literal["isotonic", "sigmoid"] = "isotonic",
) -> CalibratedModel:
    """Fit per-class isotonic regression on a held-out slice."""
    if method != "isotonic":
        raise NotImplementedError("only isotonic calibration is supported in v1")

    raw = np.asarray(booster.predict(_clean_features(X_calib)), dtype=float)
    if raw.ndim == 1:
        raw = np.column_stack([1.0 - raw, raw])
    n_classes = raw.shape[1]
    y = np.asarray(y_calib, dtype=int)
    calibrators: list[IsotonicRegression] = []
    for c in range(n_classes):
        target = (y == c).astype(int)
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw[:, c], target)
        calibrators.append(cal)
    return CalibratedModel(booster, calibrators, n_classes)
