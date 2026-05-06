"""Unit tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.modeling.evaluate import (
    brier_score_multiclass,
    brier_skill_score,
    decile_spread,
    evaluate_predictions,
    hit_rate,
    information_coefficient,
    rank_information_coefficient,
)

# ---------------------------------------------------------------------------
# Single-shot metrics
# ---------------------------------------------------------------------------


def test_perfect_pearson_yields_ic_one() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=200)
    y_pred = y_true.copy()
    assert information_coefficient(y_true, y_pred) == pytest.approx(1.0)
    assert rank_information_coefficient(y_true, y_pred) == pytest.approx(1.0)
    assert hit_rate(y_true, y_pred) == pytest.approx(1.0)


def test_random_predictions_ic_near_zero() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=2000)
    y_pred = rng.normal(size=2000)
    ic = information_coefficient(y_true, y_pred)
    rank_ic = rank_information_coefficient(y_true, y_pred)
    assert abs(ic) < 0.1
    assert abs(rank_ic) < 0.1


def test_decile_spread_perfect_predictor_is_positive_and_large() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=1000)
    spread = decile_spread(y_true, y_true)
    assert spread > 0  # top decile higher than bottom
    # And it's notably larger than a noisy predictor's spread.
    noisy = y_true + rng.normal(scale=2.0, size=1000)
    spread_noisy = decile_spread(y_true, noisy)
    assert spread > spread_noisy


def test_hit_rate_inverted_predictions_yields_zero() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=200)
    y_pred = -y_true
    assert hit_rate(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_skill_positive_when_model_beats_uniform() -> None:
    y_true = np.array([0, 1, 2, 3, 4] * 20, dtype=int)
    n_classes = 5
    n = len(y_true)

    # Confident, mostly-correct model.
    model_proba = np.zeros((n, n_classes), dtype=float)
    for i, lbl in enumerate(y_true):
        model_proba[i, lbl] = 0.9
        model_proba[i, (lbl + 1) % n_classes] = 0.1

    # Uniform-prior baseline.
    baseline = np.full((n, n_classes), 1.0 / n_classes)

    bs_model = brier_score_multiclass(y_true, model_proba)
    bs_base = brier_score_multiclass(y_true, baseline)
    assert bs_model < bs_base
    assert brier_skill_score(y_true, model_proba, baseline) > 0


def test_brier_skill_negative_when_model_worse_than_baseline() -> None:
    y_true = np.array([0, 1, 2] * 20, dtype=int)
    n_classes = 3
    n = len(y_true)

    # Confidently wrong model.
    bad_proba = np.zeros((n, n_classes), dtype=float)
    for i, lbl in enumerate(y_true):
        wrong = (lbl + 1) % n_classes
        bad_proba[i, wrong] = 0.95
        bad_proba[i, lbl] = 0.025
        bad_proba[i, (lbl + 2) % n_classes] = 0.025

    baseline = np.full((n, n_classes), 1.0 / n_classes)
    assert brier_skill_score(y_true, bad_proba, baseline) < 0


# ---------------------------------------------------------------------------
# Per-date aggregation
# ---------------------------------------------------------------------------


def test_evaluate_predictions_mean_ic_close_to_one_for_perfect() -> None:
    rng = np.random.default_rng(3)
    n_dates = 30
    n_per_date = 50
    rows = []
    for d in range(n_dates):
        y = rng.normal(size=n_per_date)
        rows.append(pd.DataFrame({"y_true": y, "y_pred": y, "bar_date": d}))
    df = pd.concat(rows, ignore_index=True)

    metrics = evaluate_predictions(df["y_true"], df["y_pred"], df["bar_date"])
    assert metrics["mean_daily_ic"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["mean_daily_rank_ic"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["n_dates"] == n_dates


def test_evaluate_predictions_random_yields_ic_near_zero() -> None:
    rng = np.random.default_rng(4)
    n_dates = 60
    n_per_date = 30
    rows = []
    for d in range(n_dates):
        rows.append(
            pd.DataFrame(
                {
                    "y_true": rng.normal(size=n_per_date),
                    "y_pred": rng.normal(size=n_per_date),
                    "bar_date": d,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    metrics = evaluate_predictions(df["y_true"], df["y_pred"], df["bar_date"])
    assert abs(metrics["mean_daily_ic"]) < 0.05
    assert metrics["n_dates"] == n_dates
