"""Unit tests for rank_predictions."""

from __future__ import annotations

from datetime import date

import pandas as pd

from packages.inference.ranker import rank_predictions


def _frame(returns: list[float], top_probas: list[float], bot_probas: list[float]) -> pd.DataFrame:
    n = len(returns)
    return pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(n)],
            "as_of": [date(2024, 1, 5)] * n,
            "predicted_return": returns,
            "top_quintile_proba": top_probas,
            "bottom_quintile_proba": bot_probas,
        }
    )


def test_long_picks_sorted_descending_top_n() -> None:
    df = _frame(
        returns=[-0.01, 0.02, 0.05, -0.03, 0.04, 0.001],
        top_probas=[0.1, 0.4, 0.6, 0.05, 0.55, 0.2],
        bot_probas=[0.6, 0.2, 0.05, 0.7, 0.1, 0.3],
    )
    out = rank_predictions(df, top_n=3)
    long_returns = out["long_picks"]["predicted_return"].tolist()
    assert long_returns == sorted(long_returns, reverse=True)
    assert len(out["long_picks"]) == 3
    # Top long should be S2 (return 0.05).
    assert out["long_picks"].iloc[0]["symbol"] == "S2"


def test_short_picks_sorted_ascending_top_n() -> None:
    df = _frame(
        returns=[-0.01, 0.02, 0.05, -0.03, 0.04, 0.001],
        top_probas=[0.1, 0.4, 0.6, 0.05, 0.55, 0.2],
        bot_probas=[0.6, 0.2, 0.05, 0.7, 0.1, 0.3],
    )
    out = rank_predictions(df, top_n=2)
    short_returns = out["short_picks"]["predicted_return"].tolist()
    assert short_returns == sorted(short_returns)
    assert out["short_picks"].iloc[0]["symbol"] == "S3"


def test_high_confidence_filtered_by_threshold() -> None:
    df = _frame(
        returns=[0.01, 0.02, 0.03],
        top_probas=[0.45, 0.51, 0.7],
        bot_probas=[0.7, 0.4, 0.55],
    )
    out = rank_predictions(df, top_n=10, high_confidence_threshold=0.5)
    # high_confidence_long: > 0.5
    assert out["high_confidence_long"]["symbol"].tolist() == ["S2", "S1"]
    # high_confidence_short: > 0.5
    assert set(out["high_confidence_short"]["symbol"]) == {"S0", "S2"}


def test_empty_input_returns_empty_buckets() -> None:
    df = pd.DataFrame(
        columns=[
            "symbol", "as_of", "predicted_return",
            "top_quintile_proba", "bottom_quintile_proba",
        ]
    )
    out = rank_predictions(df, top_n=10)
    for k in ("long_picks", "short_picks", "high_confidence_long", "high_confidence_short"):
        assert out[k].empty
