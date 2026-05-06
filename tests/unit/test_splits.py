"""Unit tests for PurgedWalkForwardSplit — the heart of look-ahead-free CV."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from packages.modeling.splits import PurgedWalkForwardSplit


def _frame(n_dates: int, *, n_per_date: int = 3) -> pd.DataFrame:
    """A tidy long-format frame: n_per_date rows for each of n_dates dates."""
    base = date(2020, 1, 1)
    rows = []
    for i in range(n_dates):
        d = base + timedelta(days=i)
        for s in range(n_per_date):
            rows.append({"bar_date": d, "symbol": f"S{s}", "x": 1.0})
    return pd.DataFrame(rows)


def test_split_count_matches_n_folds() -> None:
    df = _frame(800)
    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    folds = list(splitter.split(df))
    assert len(folds) == 5


def test_train_strictly_before_val_each_fold() -> None:
    df = _frame(800)
    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    for train_idx, val_idx in splitter.split(df):
        train_max = df.iloc[train_idx]["bar_date"].max()
        val_min = df.iloc[val_idx]["bar_date"].min()
        assert train_max < val_min, (train_max, val_min)


def test_embargo_enforced() -> None:
    """max(train_dates) <= min(val_dates) - embargo_days (calendar gap)."""
    df = _frame(800)
    embargo = 5
    splitter = PurgedWalkForwardSplit(
        n_folds=4, horizon_days=5, embargo_days=embargo, min_train_size_days=200
    )
    for train_idx, val_idx in splitter.split(df):
        train_max = df.iloc[train_idx]["bar_date"].max()
        val_min = df.iloc[val_idx]["bar_date"].min()
        # We move val_min back by `embargo` calendar positions in the unique-date
        # sequence; with consecutive dates this is the same as days.
        assert (val_min - train_max).days >= embargo, (
            f"gap {(val_min - train_max).days} < embargo {embargo}"
        )


def test_purge_excludes_label_overlap() -> None:
    """No training row has bar_date + horizon_days inside the val window."""
    df = _frame(800)
    horizon = 5
    splitter = PurgedWalkForwardSplit(
        n_folds=4, horizon_days=horizon, embargo_days=5, min_train_size_days=200
    )
    for train_idx, val_idx in splitter.split(df):
        train_dates = df.iloc[train_idx]["bar_date"]
        val_dates = df.iloc[val_idx]["bar_date"]
        v_min, v_max = val_dates.min(), val_dates.max()
        for td in train_dates:
            label_date = td + timedelta(days=horizon)
            assert not (v_min <= label_date <= v_max), (
                f"train row {td} has label at {label_date}, inside val [{v_min}..{v_max}]"
            )


def test_walk_forward_chronological_non_overlapping() -> None:
    df = _frame(800)
    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    val_starts = []
    val_ends = []
    for _, val_idx in splitter.split(df):
        v_dates = df.iloc[val_idx]["bar_date"]
        val_starts.append(v_dates.min())
        val_ends.append(v_dates.max())
    # Each val window starts strictly after the previous one.
    for i in range(1, len(val_starts)):
        assert val_starts[i] > val_ends[i - 1], (val_starts[i], val_ends[i - 1])


def test_min_train_size_respected_on_first_fold() -> None:
    df = _frame(800)
    min_train = 250
    splitter = PurgedWalkForwardSplit(
        n_folds=4, horizon_days=5, embargo_days=5, min_train_size_days=min_train
    )
    folds = list(splitter.split(df))
    assert folds, "no folds yielded"
    train_idx, _ = folds[0]
    n_unique_train_dates = df.iloc[train_idx]["bar_date"].nunique()
    # First fold has at least min_train_size_days of training history minus
    # the purge buffer (we drop max(embargo, horizon) trailing dates).
    assert n_unique_train_dates >= min_train - 5


def test_too_few_dates_raises() -> None:
    df = _frame(50)  # < min_train + n_folds
    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    with pytest.raises(ValueError, match="insufficient unique dates"):
        list(splitter.split(df))


def test_indices_are_positional_and_disjoint() -> None:
    df = _frame(800).reset_index(drop=True)
    splitter = PurgedWalkForwardSplit(
        n_folds=3, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    for train_idx, val_idx in splitter.split(df):
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)
        assert len(set(train_idx) & set(val_idx)) == 0
        assert train_idx.max() < len(df) and val_idx.max() < len(df)
