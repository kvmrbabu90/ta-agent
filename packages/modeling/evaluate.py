"""Evaluation metrics for the modeling layer.

Per-date convention (per the PRD): IC, rank IC, and decile spread are
computed PER bar_date (cross-sectionally), then averaged across dates.
Pooling all (symbol, date) rows together mixes cross-sectional and
time-series variance and is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Single-shot metrics (operate on one cross-section or one combined sample)
# ---------------------------------------------------------------------------


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between predicted and actual values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2:
        return float("nan")
    if np.std(y_pred[valid]) == 0 or np.std(y_true[valid]) == 0:
        return float("nan")
    r, _ = pearsonr(y_true[valid], y_pred[valid])
    return float(r)


def rank_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2:
        return float("nan")
    r, _ = spearmanr(y_true[valid], y_pred[valid])
    return float(r)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction where sign(y_pred) == sign(y_true). NaN-safe."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[valid]) == np.sign(y_pred[valid])).mean())


def decile_spread(
    y_true: np.ndarray, y_pred: np.ndarray, n_buckets: int = 10
) -> float:
    """Mean(y_true | top-bucket y_pred) - Mean(y_true | bottom-bucket y_pred)."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(df) < n_buckets:
        return float("nan")
    try:
        df["bucket"] = pd.qcut(df["y_pred"], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    if df["bucket"].nunique() < 2:
        return float("nan")
    top = df[df["bucket"] == df["bucket"].max()]["y_true"].mean()
    bot = df[df["bucket"] == df["bucket"].min()]["y_true"].mean()
    return float(top - bot)


# ---------------------------------------------------------------------------
# Classification probability metrics
# ---------------------------------------------------------------------------


def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Average squared distance between one-hot true labels and predicted probas."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    n_classes = y_proba.shape[1]
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(len(y_true)), np.clip(y_true, 0, n_classes - 1)] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def brier_skill_score(
    y_true: np.ndarray, y_proba: np.ndarray, baseline_proba: np.ndarray
) -> float:
    """1 - Brier(model) / Brier(baseline). Positive = beats baseline."""
    bs = brier_score_multiclass(y_true, y_proba)
    bs_base = brier_score_multiclass(y_true, baseline_proba)
    if bs_base == 0:
        return float("nan")
    return float(1.0 - bs / bs_base)


# ---------------------------------------------------------------------------
# Per-date aggregation (the canonical evaluator)
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true_returns: pd.Series,
    y_pred_returns: pd.Series,
    bar_dates: pd.Series,
    *,
    min_obs_per_date: int = 5,
    n_buckets: int = 10,
) -> dict:
    """Compute IC and rank IC per date, then aggregate.

    Returns:
        mean_daily_ic, std_daily_ic, t_stat_daily_ic
        mean_daily_rank_ic, std_daily_rank_ic
        mean_decile_spread (averaged over dates with >= n_buckets observations)
        hit_rate (pooled — sign agreement, no leakage concern)
        n_dates (with valid IC)
    """
    df = pd.DataFrame(
        {
            "y_true": pd.Series(y_true_returns).reset_index(drop=True),
            "y_pred": pd.Series(y_pred_returns).reset_index(drop=True),
            "bar_date": pd.Series(bar_dates).reset_index(drop=True),
        }
    ).dropna(subset=["y_true", "y_pred"])

    if df.empty:
        return {
            "mean_daily_ic": float("nan"),
            "std_daily_ic": float("nan"),
            "t_stat_daily_ic": float("nan"),
            "mean_daily_rank_ic": float("nan"),
            "std_daily_rank_ic": float("nan"),
            "mean_decile_spread": float("nan"),
            "hit_rate": float("nan"),
            "n_dates": 0,
        }

    daily_ic: list[float] = []
    daily_rank_ic: list[float] = []
    daily_spread: list[float] = []
    for _, group in df.groupby("bar_date"):
        if len(group) < min_obs_per_date:
            continue
        ic = information_coefficient(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        ric = rank_information_coefficient(
            group["y_true"].to_numpy(), group["y_pred"].to_numpy()
        )
        if not np.isnan(ic):
            daily_ic.append(ic)
        if not np.isnan(ric):
            daily_rank_ic.append(ric)
        if len(group) >= n_buckets:
            sp = decile_spread(
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
                n_buckets=n_buckets,
            )
            if not np.isnan(sp):
                daily_spread.append(sp)

    def _mean(x: list[float]) -> float:
        return float(np.mean(x)) if x else float("nan")

    def _std(x: list[float]) -> float:
        return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")

    n = len(daily_ic)
    t_stat = (
        _mean(daily_ic) / (_std(daily_ic) / np.sqrt(n))
        if n > 1 and _std(daily_ic) > 0
        else float("nan")
    )

    return {
        "mean_daily_ic": _mean(daily_ic),
        "std_daily_ic": _std(daily_ic),
        "t_stat_daily_ic": float(t_stat),
        "mean_daily_rank_ic": _mean(daily_rank_ic),
        "std_daily_rank_ic": _std(daily_rank_ic),
        "mean_decile_spread": _mean(daily_spread),
        "hit_rate": hit_rate(df["y_true"].to_numpy(), df["y_pred"].to_numpy()),
        "n_dates": n,
    }
