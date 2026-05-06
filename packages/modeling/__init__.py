"""Modeling layer: purged walk-forward CV, LightGBM training, calibration,
evaluation, and a simple file-based model registry."""

from packages.modeling.calibrate import CalibratedModel, calibrate_classifier
from packages.modeling.evaluate import (
    brier_score_multiclass,
    brier_skill_score,
    decile_spread,
    evaluate_predictions,
    hit_rate,
    information_coefficient,
    rank_information_coefficient,
)
from packages.modeling.registry import (
    ModelMetadata,
    find_latest_model_dir,
    load_latest_model,
    save_model,
)
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import (
    TrainConfig,
    train_final_model,
    train_one_fold,
    train_with_cv,
)

__all__ = [
    "PurgedWalkForwardSplit",
    "TrainConfig",
    "train_one_fold",
    "train_with_cv",
    "train_final_model",
    "CalibratedModel",
    "calibrate_classifier",
    "information_coefficient",
    "rank_information_coefficient",
    "hit_rate",
    "decile_spread",
    "brier_score_multiclass",
    "brier_skill_score",
    "evaluate_predictions",
    "ModelMetadata",
    "save_model",
    "load_latest_model",
    "find_latest_model_dir",
]
