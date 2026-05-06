"""Base classes for feature groups.

Two flavors:
    FeatureGroup       — operates on one symbol's OHLCV history at a time.
                         Used for purely time-series features (returns, RSI,
                         ATR, etc).
    PanelFeatureGroup  — operates on the full universe panel for a single date.
                         Used for cross-sectional and regime features.

Causality contract:
    Every feature value at row T must depend ONLY on data with bar_date <= T.
    No future leakage. No centered windows. No imputation that pulls forward.
    The mandatory test in tests/unit/test_features_causality.py enforces this
    end-to-end.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureGroup(ABC):
    """Per-symbol feature group: one OHLCV history in, one feature DataFrame out."""

    name: str = "base"

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Produce features for a single symbol.

        Inputs:
            ohlcv: a DataFrame for ONE symbol, sorted ascending by bar_date,
                   with at least columns: bar_date, open, high, low, close, volume.

        Output:
            A DataFrame indexed by bar_date with feature columns prefixed by
            ``{self.name}__`` to avoid collisions across groups. The index
            should match the input bar_dates exactly (warmup rows are NaN).
        """

    def _validate_input(self, ohlcv: pd.DataFrame) -> None:
        required = {"bar_date", "open", "high", "low", "close", "volume"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"{self.name}: input missing columns {missing}")
        if not ohlcv["bar_date"].is_monotonic_increasing:
            raise ValueError(f"{self.name}: input bar_date must be sorted ascending")


class PanelFeatureGroup(ABC):
    """Panel feature group: long-format panel in, long-format feature panel out."""

    name: str = "base_panel"

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Produce panel features.

        Inputs:
            panel: a long-format DataFrame keyed by (symbol, bar_date) with at
                   least columns: symbol, bar_date, open, high, low, close, volume,
                   plus any per-symbol features already computed.

        Output:
            A DataFrame keyed by (symbol, bar_date) with feature columns
            prefixed by ``{self.name}__``.
        """

    def _validate_input(self, panel: pd.DataFrame) -> None:
        required = {"symbol", "bar_date"}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"{self.name}: panel missing columns {missing}")
