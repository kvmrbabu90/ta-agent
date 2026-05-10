"""Tests for the Optuna trial-constraint logic."""

from __future__ import annotations

import math

from packages.modeling.tune import TuneConstraints, _objective_metric


def _fold(*, rank_ic: float, accuracy: float = 0.0,
          best_iter: int = 100, train_size: int = 500_000) -> dict:
    """Build a single fold-metric dict shaped like train_with_cv emits."""
    return {
        "mean_daily_rank_ic": rank_ic,
        "val_accuracy": accuracy,
        "best_iter": best_iter,
        "train_size": train_size,
    }


def _cv(folds: list[dict]) -> dict:
    return {"fold_metrics": folds, "mean_metrics": {}}


# ---------------------------------------------------------------------------
# Unconstrained baseline
# ---------------------------------------------------------------------------


def test_no_constraints_returns_simple_mean() -> None:
    cv = _cv([_fold(rank_ic=0.01), _fold(rank_ic=0.02), _fold(rank_ic=0.03)])
    out = _objective_metric(cv, "regression", constraints=None)
    assert out == 0.02


def test_no_constraints_handles_empty_folds() -> None:
    out = _objective_metric(_cv([]), "regression", constraints=None)
    assert math.isnan(out)


# ---------------------------------------------------------------------------
# Constraint #1 — min_best_iter
# ---------------------------------------------------------------------------


def test_min_best_iter_rejects_single_round_trial() -> None:
    """A trial whose folds all stop at iter=1 is degenerate — the model is
    effectively a constant prediction."""
    cv = _cv([
        _fold(rank_ic=0.05, best_iter=1),
        _fold(rank_ic=0.05, best_iter=1),
    ])
    out = _objective_metric(cv, "regression", TuneConstraints(min_best_iter=50))
    assert out == float("-inf")


def test_min_best_iter_accepts_when_all_folds_meet_threshold() -> None:
    cv = _cv([_fold(rank_ic=0.01, best_iter=80), _fold(rank_ic=0.02, best_iter=120)])
    out = _objective_metric(cv, "regression", TuneConstraints(min_best_iter=50))
    assert out == 0.015


def test_min_best_iter_rejects_if_any_fold_below_threshold() -> None:
    """Even one stunted fold rejects the trial."""
    cv = _cv([
        _fold(rank_ic=0.01, best_iter=80),
        _fold(rank_ic=0.05, best_iter=2),  # this one
    ])
    out = _objective_metric(cv, "regression", TuneConstraints(min_best_iter=50))
    assert out == float("-inf")


# ---------------------------------------------------------------------------
# Constraint #2 — min_fold_train_size (drop, don't reject)
# ---------------------------------------------------------------------------


def test_min_fold_train_size_drops_small_folds_from_mean() -> None:
    """Fold 0 has 100k rows (small) → dropped. Mean computed on remaining 2 folds."""
    cv = _cv([
        _fold(rank_ic=0.20, train_size=100_000),  # the "fold-0 anomaly" — dropped
        _fold(rank_ic=0.01, train_size=500_000),
        _fold(rank_ic=0.02, train_size=600_000),
    ])
    out = _objective_metric(
        cv, "regression",
        TuneConstraints(min_fold_train_size=200_000, min_best_iter=1, max_fold_dominance_z=float("inf")),
    )
    assert out == 0.015  # (0.01 + 0.02) / 2 — the 0.20 anomaly was dropped


def test_all_folds_dropped_returns_neg_inf() -> None:
    """If every fold is too small, the objective is undefined."""
    cv = _cv([_fold(rank_ic=0.05, train_size=10_000)])
    out = _objective_metric(
        cv, "regression",
        TuneConstraints(min_fold_train_size=200_000, min_best_iter=1, max_fold_dominance_z=float("inf")),
    )
    assert out == float("-inf")


# ---------------------------------------------------------------------------
# Constraint #3 — max_fold_dominance_z
# ---------------------------------------------------------------------------


def test_fold_dominance_rejects_extreme_outlier() -> None:
    """One fold scoring 5σ above the others is suspicious — reject."""
    cv = _cv([
        _fold(rank_ic=0.20),  # the outlier
        _fold(rank_ic=0.01),
        _fold(rank_ic=0.012),
        _fold(rank_ic=0.014),
        _fold(rank_ic=0.011),
    ])
    out = _objective_metric(
        cv, "regression",
        TuneConstraints(min_best_iter=1, min_fold_train_size=0, max_fold_dominance_z=2.5),
    )
    assert out == float("-inf")


def test_fold_dominance_accepts_when_within_tolerance() -> None:
    """Folds clustered together — no domination — pass through."""
    cv = _cv([
        _fold(rank_ic=0.012),
        _fold(rank_ic=0.018),
        _fold(rank_ic=0.014),
        _fold(rank_ic=0.016),
        _fold(rank_ic=0.011),
    ])
    out = _objective_metric(
        cv, "regression",
        TuneConstraints(min_best_iter=1, min_fold_train_size=0, max_fold_dominance_z=2.5),
    )
    # (0.012+0.018+0.014+0.016+0.011) / 5 = 0.0142
    assert abs(out - 0.0142) < 1e-9


def test_fold_dominance_disabled_with_inf() -> None:
    """Setting max_fold_dominance_z = inf disables the check."""
    cv = _cv([
        _fold(rank_ic=0.20), _fold(rank_ic=0.01),
        _fold(rank_ic=0.01), _fold(rank_ic=0.01), _fold(rank_ic=0.01),
    ])
    out = _objective_metric(
        cv, "regression",
        TuneConstraints(min_best_iter=1, min_fold_train_size=0, max_fold_dominance_z=float("inf")),
    )
    # All 5 folds counted → mean = 0.048
    assert abs(out - 0.048) < 1e-9


# ---------------------------------------------------------------------------
# Constraint composition
# ---------------------------------------------------------------------------


def test_constraints_compose_size_drops_then_min_iter_rejects() -> None:
    """Fold 0 is too small (drop). Of the remaining folds, one has best_iter<50
    → reject the whole trial."""
    cv = _cv([
        _fold(rank_ic=0.05, train_size=100_000, best_iter=200),  # dropped (small)
        _fold(rank_ic=0.02, train_size=500_000, best_iter=100),
        _fold(rank_ic=0.02, train_size=500_000, best_iter=10),   # too few iters
    ])
    out = _objective_metric(
        cv, "regression",
        TuneConstraints(min_best_iter=50, min_fold_train_size=200_000, max_fold_dominance_z=float("inf")),
    )
    assert out == float("-inf")


def test_classification_objective_uses_val_accuracy() -> None:
    cv = _cv([
        _fold(rank_ic=0.0, accuracy=0.25, train_size=500_000, best_iter=100),
        _fold(rank_ic=0.0, accuracy=0.30, train_size=500_000, best_iter=100),
    ])
    out = _objective_metric(cv, "classification", TuneConstraints(min_best_iter=50))
    assert out == 0.275
