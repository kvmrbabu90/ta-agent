"""Pick selection: top-N long, top-N short, and high-confidence subsets."""

from __future__ import annotations

import pandas as pd


def rank_predictions(
    predictions: pd.DataFrame,
    top_n: int = 20,
    *,
    high_confidence_threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Slice a predictions frame into long / short / high-confidence picks.

    Required columns: predicted_return, top_quintile_proba, bottom_quintile_proba.
    """
    if predictions.empty:
        empty = predictions.iloc[0:0]
        return {
            "long_picks": empty,
            "short_picks": empty,
            "high_confidence_long": empty,
            "high_confidence_short": empty,
        }

    sorted_desc = predictions.sort_values("predicted_return", ascending=False)
    sorted_asc = predictions.sort_values("predicted_return", ascending=True)

    return {
        "long_picks": sorted_desc.head(top_n).reset_index(drop=True),
        "short_picks": sorted_asc.head(top_n).reset_index(drop=True),
        "high_confidence_long": (
            predictions[predictions["top_quintile_proba"] > high_confidence_threshold]
            .sort_values("top_quintile_proba", ascending=False)
            .reset_index(drop=True)
        ),
        "high_confidence_short": (
            predictions[
                predictions["bottom_quintile_proba"] > high_confidence_threshold
            ]
            .sort_values("bottom_quintile_proba", ascending=False)
            .reset_index(drop=True)
        ),
    }
