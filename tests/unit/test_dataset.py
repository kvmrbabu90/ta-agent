"""Unit tests for the master training-dataset assembly."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.common.config import settings
from packages.ingestion.storage import get_conn, upsert_membership, upsert_ohlcv
from packages.labels.dataset import build_training_dataset


def _seed_fake_universe(
    path: str,
    *,
    n_symbols: int = 4,
    n_days: int = 280,
    universe: str = "SP500",
    start_date: date = date(2022, 1, 3),
) -> None:
    members: list[dict] = []
    bars: list[dict] = []
    for i in range(n_symbols):
        sym = f"SYM{i}"
        members.append(
            {
                "universe": universe, "symbol": sym, "exchange": "NASDAQ",
                "start_date": start_date, "end_date": None,
                "company_name": sym,
            }
        )
        rng = np.random.default_rng(i)
        daily_ret = rng.normal(0.0005, 0.012, size=n_days)
        close = (100.0 + 10.0 * i) * np.exp(np.cumsum(daily_ret))
        open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n_days))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n_days)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n_days)))
        volume = rng.integers(1_000_000, 5_000_000, size=n_days)
        for j in range(n_days):
            bars.append(
                {
                    "symbol": sym, "exchange": "NASDAQ",
                    "bar_date": start_date + timedelta(days=j),
                    "open": float(open_[j]), "high": float(high[j]),
                    "low": float(low[j]), "close": float(close[j]),
                    "volume": int(volume[j]),
                    "close_unadj": None, "source": "test",
                    "ingested_at": datetime(2024, 1, 1),
                }
            )

    with get_conn(path) as conn:
        upsert_membership(pd.DataFrame(members), conn=conn)
        upsert_ohlcv(pd.DataFrame(bars), conn=conn)


def test_build_training_dataset_columns_and_in_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_fake_universe(db, n_symbols=4, n_days=280)

    out_file = tmp_path / "training.parquet"
    df = build_training_dataset(
        "SP500",
        date(2022, 1, 3),
        date(2022, 1, 3) + timedelta(days=279),
        horizon_days=5,
        duckdb_path=db,
        output_path=out_file,
    )

    expected_target_cols = {"fwd_return_5d", "fwd_quintile_5d", "in_universe"}
    assert expected_target_cols.issubset(set(df.columns))
    assert {"symbol", "bar_date"}.issubset(set(df.columns))
    # Output file persisted.
    assert out_file.exists()
    # All rows are in_universe given our flat membership.
    assert df["in_universe"].all()


def test_label_count_matches_in_universe_with_future_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-null label rows = in_universe rows that have at least horizon trading bars ahead."""
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    n_days = 100
    n_symbols = 4
    _seed_fake_universe(db, n_symbols=n_symbols, n_days=n_days)

    df = build_training_dataset(
        "SP500",
        date(2022, 1, 3),
        date(2022, 1, 3) + timedelta(days=n_days - 1),
        horizon_days=5,
        duckdb_path=db,
        output_path=tmp_path / "x.parquet",
    )

    # Per symbol: n_days - 5 rows have a forward return (last 5 are NaN).
    expected_labeled = (n_days - 5) * n_symbols
    actual_labeled = int(df["fwd_quintile_5d"].notna().sum())
    # qcut needs at least n_classes=5 distinct values per date — with only 4
    # symbols per date we won't get any quintile labels. So the count will be 0.
    # Re-run with horizon=2 and n_classes=2 to assert the relationship cleanly.
    assert actual_labeled == 0  # documents the n_classes vs n_symbols constraint

    df2 = build_training_dataset(
        "SP500",
        date(2022, 1, 3),
        date(2022, 1, 3) + timedelta(days=n_days - 1),
        horizon_days=5,
        n_classes=2,
        duckdb_path=db,
        output_path=tmp_path / "x2.parquet",
    )
    expected_labeled = (n_days - 5) * n_symbols
    actual2 = int(df2["fwd_quintile_5d"].notna().sum())
    assert actual2 == expected_labeled


def test_build_training_dataset_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_fake_universe(db, n_symbols=4, n_days=280)

    a = build_training_dataset(
        "SP500", date(2022, 1, 3), date(2022, 1, 3) + timedelta(days=279),
        horizon_days=5, n_classes=2, duckdb_path=db,
        output_path=tmp_path / "a.parquet",
    )
    b = build_training_dataset(
        "SP500", date(2022, 1, 3), date(2022, 1, 3) + timedelta(days=279),
        horizon_days=5, n_classes=2, duckdb_path=db,
        output_path=tmp_path / "b.parquet",
    )

    label_cols = ["symbol", "bar_date", "fwd_return_5d", "fwd_quintile_5d", "in_universe"]
    pd.testing.assert_frame_equal(
        a[label_cols].sort_values(["symbol", "bar_date"]).reset_index(drop=True),
        b[label_cols].sort_values(["symbol", "bar_date"]).reset_index(drop=True),
    )
