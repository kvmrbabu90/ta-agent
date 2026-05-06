"""Purged walk-forward cross-validation for financial time series.

Why we need this (vs sklearn's TimeSeriesSplit):
    Standard time-series splits prevent val data from PRECEDING the train
    data — but they do NOT remove training rows whose label horizon
    overlaps the validation window. With a 5-day forward-return label,
    the training row at date t carries information about close[t+5];
    if t+5 lies inside the validation window, that row's label
    "knows" about validation outcomes. The model picks up on that
    leakage and validation scores are inflated.

    The fix is **purging** (drop training rows whose label peeks into
    val) and **embargoing** (drop a buffer of training rows immediately
    before val). The combination is from López de Prado.

How fold k works (k = 0 .. n_folds-1):

    Validation window = bars [v_start_idx, v_end_idx], with windows
    chosen so they are non-overlapping, contiguous, and chronological.

    Training set:
        - bar_date <= v_start_date - max(embargo_days, horizon_days)
        - The combined cutoff handles both rules at once: a row whose
          label crosses into val would have its label_date >= v_start,
          i.e. bar_date >= v_start - horizon_days.

The returned indices are POSITIONAL (use ``df.iloc[idx]``).

Default config matches the PRD: 5 folds, 5-day horizon, 5-day embargo,
2-year minimum training history.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForwardSplit:
    n_folds: int = 5
    horizon_days: int = 5
    embargo_days: int = 5
    min_train_size_days: int = 504  # ~2 trading years

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) pairs in chronological order.

        ``df`` must have a ``bar_date`` column. Indices are positional
        (compatible with ``df.iloc``)."""
        if "bar_date" not in df.columns:
            raise ValueError("df must have a 'bar_date' column")
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")

        bar_dates = pd.to_datetime(df["bar_date"]).dt.date.to_numpy()
        unique_dates = np.unique(bar_dates)
        unique_dates.sort()
        T = len(unique_dates)

        if self.min_train_size_days + self.n_folds > T:
            raise ValueError(
                f"insufficient unique dates ({T}) for "
                f"min_train_size_days={self.min_train_size_days} + n_folds={self.n_folds}"
            )

        test_dates = T - self.min_train_size_days
        val_size = test_dates // self.n_folds
        if val_size < 1:
            raise ValueError("val window size collapsed to zero — too few dates")

        purge_window = max(self.embargo_days, self.horizon_days)

        for k in range(self.n_folds):
            v_start_pos = self.min_train_size_days + k * val_size
            v_end_pos = (
                v_start_pos + val_size - 1
                if k < self.n_folds - 1
                else T - 1
            )

            train_cutoff_pos = v_start_pos - purge_window
            if train_cutoff_pos <= 0:
                continue

            v_start_date = unique_dates[v_start_pos]
            v_end_date = unique_dates[v_end_pos]
            train_max_date = unique_dates[train_cutoff_pos - 1]

            train_mask = bar_dates <= train_max_date
            val_mask = (bar_dates >= v_start_date) & (bar_dates <= v_end_date)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx
