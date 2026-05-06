"""Unit tests for forward-return and quintile-label generation."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from packages.labels.targets import (
    compute_forward_log_return,
    compute_quintile_labels,
)

# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------


def test_forward_return_arithmetic() -> None:
    closes = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0]
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * len(closes),
            "bar_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(len(closes))],
            "close": closes,
        }
    )
    out = compute_forward_log_return(df, horizon_days=5)
    col = "fwd_return_5d"
    # Index 0: log(110/100); Index 1: log(112/102); after that: NaN.
    assert out[col].iloc[0] == pytest.approx(np.log(110.0 / 100.0))
    assert out[col].iloc[1] == pytest.approx(np.log(112.0 / 102.0))
    # The last 5 rows should all be NaN
    assert out[col].iloc[-5:].isna().all()


def test_forward_return_per_symbol_no_cross_pollination() -> None:
    """Two symbols with different price trajectories must have independent forward returns."""
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 10 + ["B"] * 10,
            "bar_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(10)] * 2,
            "close": list(range(100, 110)) + list(range(200, 210)),
        }
    )
    out = compute_forward_log_return(df, horizon_days=5)
    a = out[out["symbol"] == "A"].sort_values("bar_date").reset_index(drop=True)
    b = out[out["symbol"] == "B"].sort_values("bar_date").reset_index(drop=True)

    # A's row 0: log(105/100); B's row 0: log(205/200) — must differ.
    assert a["fwd_return_5d"].iloc[0] == pytest.approx(np.log(105.0 / 100.0))
    assert b["fwd_return_5d"].iloc[0] == pytest.approx(np.log(205.0 / 200.0))


def test_forward_return_uses_trading_bars_not_calendar_days() -> None:
    """Bars are dropped on weekends; horizon=5 picks the 5th NEXT bar in sequence."""
    bar_dates = [
        date(2024, 1, 1),  # Mon
        date(2024, 1, 2),  # Tue
        date(2024, 1, 3),  # Wed
        date(2024, 1, 4),  # Thu
        date(2024, 1, 5),  # Fri
        date(2024, 1, 8),  # Mon (weekend skipped)
        date(2024, 1, 9),  # Tue
    ]
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    df = pd.DataFrame({"symbol": ["AAPL"] * 7, "bar_date": bar_dates, "close": closes})
    out = compute_forward_log_return(df, horizon_days=5)
    # Row 0 (Mon Jan 1) → 5 trading bars later is Mon Jan 8 (close 105.0).
    assert out["fwd_return_5d"].iloc[0] == pytest.approx(np.log(105.0 / 100.0))
    # Row 1 (Tue) → close at index 6 (Tue Jan 9).
    assert out["fwd_return_5d"].iloc[1] == pytest.approx(np.log(106.0 / 101.0))
    # Last 5 rows NaN
    assert out["fwd_return_5d"].iloc[-5:].isna().all()


# ---------------------------------------------------------------------------
# Quintile labels
# ---------------------------------------------------------------------------


def _membership_all_in(symbols: list[str], universe: str = "SP500") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "universe": universe,
                "symbol": s,
                "start_date": date(2010, 1, 1),
                "end_date": None,
                "company_name": s,
            }
            for s in symbols
        ]
    )


def test_quintile_labels_balanced_buckets() -> None:
    n = 100
    symbols = [f"S{i:03d}" for i in range(n)]
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.05, size=n)
    fwd = pd.DataFrame(
        {"symbol": symbols, "bar_date": [date(2024, 1, 5)] * n, "fwd_return_5d": returns}
    )
    mem = _membership_all_in(symbols)
    out = compute_quintile_labels(fwd, mem, "SP500", horizon_days=5, n_classes=5)
    counts = out["fwd_quintile_5d"].value_counts().sort_index()
    assert sorted(counts.index) == [0.0, 1.0, 2.0, 3.0, 4.0]
    # 100/5 = 20 per bucket
    assert (counts == 20).all()


def test_quintile_top_bucket_holds_largest_returns() -> None:
    n = 50
    symbols = [f"S{i:03d}" for i in range(n)]
    returns = np.linspace(-0.1, 0.1, n)
    fwd = pd.DataFrame(
        {"symbol": symbols, "bar_date": [date(2024, 1, 5)] * n, "fwd_return_5d": returns}
    )
    mem = _membership_all_in(symbols)
    out = compute_quintile_labels(fwd, mem, "SP500", horizon_days=5, n_classes=5)
    top = out[out["fwd_quintile_5d"] == 4]
    bot = out[out["fwd_quintile_5d"] == 0]
    assert top["fwd_return_5d"].min() > bot["fwd_return_5d"].max()


def test_quintile_respects_pit_membership() -> None:
    """Stocks not yet in the universe on date T must not get a quintile label."""
    symbols = ["A", "B", "C", "D", "E"]
    fwd = pd.DataFrame(
        {
            "symbol": symbols,
            "bar_date": [date(2024, 1, 5)] * 5,
            "fwd_return_5d": [-0.02, -0.01, 0.0, 0.01, 0.02],
        }
    )
    # E joined the universe AFTER 2024-01-05 → must have NaN on that date.
    mem = pd.DataFrame(
        [
            {"universe": "SP500", "symbol": "A", "start_date": date(2010, 1, 1), "end_date": None, "company_name": "A"},
            {"universe": "SP500", "symbol": "B", "start_date": date(2010, 1, 1), "end_date": None, "company_name": "B"},
            {"universe": "SP500", "symbol": "C", "start_date": date(2010, 1, 1), "end_date": None, "company_name": "C"},
            {"universe": "SP500", "symbol": "D", "start_date": date(2010, 1, 1), "end_date": None, "company_name": "D"},
            {"universe": "SP500", "symbol": "E", "start_date": date(2024, 1, 6), "end_date": None, "company_name": "E"},
        ]
    )
    out = compute_quintile_labels(fwd, mem, "SP500", horizon_days=5, n_classes=4)
    e_row = out[out["symbol"] == "E"].iloc[0]
    assert pd.isna(e_row["fwd_quintile_5d"])
    # And the non-E rows should have valid labels.
    valid = out[out["symbol"] != "E"]["fwd_quintile_5d"]
    assert valid.notna().all()


def test_quintile_does_not_use_future_membership() -> None:
    """Leakage check: stock joining T+1 should not appear in T's ranking."""
    fwd = pd.DataFrame(
        {
            "symbol": ["X", "Y", "Z"],
            "bar_date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "fwd_return_5d": [0.05, -0.02, 0.10],  # Z is highest
        }
    )
    mem = pd.DataFrame(
        [
            {"universe": "SP500", "symbol": "X", "start_date": date(2020, 1, 1), "end_date": None, "company_name": "X"},
            {"universe": "SP500", "symbol": "Y", "start_date": date(2020, 1, 1), "end_date": None, "company_name": "Y"},
            # Z joins on T+1 — must NOT participate in T's ranking.
            {"universe": "SP500", "symbol": "Z", "start_date": date(2024, 1, 2), "end_date": None, "company_name": "Z"},
        ]
    )
    out = compute_quintile_labels(fwd, mem, "SP500", horizon_days=5, n_classes=2)
    z_row = out[out["symbol"] == "Z"].iloc[0]
    assert pd.isna(z_row["fwd_quintile_5d"])
    # X and Y are the only two participants; their ranks must split 0 and 1.
    xy = out[out["symbol"].isin(["X", "Y"])]
    assert set(xy["fwd_quintile_5d"].dropna().tolist()) == {0.0, 1.0}


def test_quintile_skips_when_too_few_rankable() -> None:
    """Below n_classes distinct returns → all NaN for that date (no spurious labels)."""
    fwd = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "bar_date": [date(2024, 1, 5)] * 2,
            "fwd_return_5d": [0.01, 0.02],
        }
    )
    mem = _membership_all_in(["A", "B"])
    out = compute_quintile_labels(fwd, mem, "SP500", horizon_days=5, n_classes=5)
    assert out["fwd_quintile_5d"].isna().all()
