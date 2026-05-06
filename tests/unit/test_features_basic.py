"""Basic sanity tests for the feature-engineering pipeline."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.common.config import settings
from packages.features.cross_sectional import CrossSectionalFeatures
from packages.features.microstructure import MicrostructureFeatures
from packages.features.momentum import MomentumFeatures
from packages.features.pipeline import build_feature_matrix
from packages.features.price import PriceFeatures
from packages.features.regime import RegimeFeatures
from packages.features.trend import TrendFeatures
from packages.features.volatility import VolatilityFeatures
from packages.features.volume import VolumeFeatures
from packages.ingestion.storage import get_conn, upsert_membership, upsert_ohlcv

# Expected per-feature counts from the prompt's catalog.
EXPECTED_PER_GROUP = {
    "price": 8,
    "trend": 10,
    "momentum": 8,
    "volatility": 6,
    "volume": 5,
    "microstructure": 5,
    "cross_sectional": 6,
    "regime": 5,
}


def _synthetic_ohlcv(n_days: int = 400, *, seed: int = 0, start_close: float = 100.0) -> pd.DataFrame:
    """A single symbol's deterministic-ish synthetic OHLCV."""
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(loc=0.0005, scale=0.012, size=n_days)
    close = start_close * np.exp(np.cumsum(daily_ret))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n_days))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n_days)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n_days)))
    volume = rng.integers(1_000_000, 5_000_000, size=n_days)
    bar_dates = [date(2022, 1, 3) + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame(
        {
            "bar_date": bar_dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ---------------------------------------------------------------------------
# Per-group feature counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group,expected",
    [
        (PriceFeatures(), EXPECTED_PER_GROUP["price"]),
        (TrendFeatures(), EXPECTED_PER_GROUP["trend"]),
        (MomentumFeatures(), EXPECTED_PER_GROUP["momentum"]),
        (VolatilityFeatures(), EXPECTED_PER_GROUP["volatility"]),
        (VolumeFeatures(), EXPECTED_PER_GROUP["volume"]),
        (MicrostructureFeatures(), EXPECTED_PER_GROUP["microstructure"]),
    ],
)
def test_per_symbol_feature_count_and_naming(group, expected) -> None:
    df = _synthetic_ohlcv(400)
    out = group.compute(df)
    assert out.shape[1] == expected, (
        f"{group.name}: expected {expected} cols, got {out.shape[1]}: {list(out.columns)}"
    )
    # Column-naming convention.
    assert all(c.startswith(f"{group.name}__") for c in out.columns)


def test_input_validation_missing_columns() -> None:
    bad = pd.DataFrame({"bar_date": [date(2024, 1, 1)], "close": [100.0]})
    with pytest.raises(ValueError, match="missing"):
        PriceFeatures().compute(bad)


def test_insufficient_data_yields_nan_not_errors() -> None:
    """A short series should produce NaNs in long-window features, not raise."""
    df = _synthetic_ohlcv(20)  # < 200 → sma_ratio_50_200 etc. all NaN
    out = TrendFeatures().compute(df)
    # The 200-day rolling features should be entirely NaN
    assert out["trend__price_to_sma200_ratio"].isna().all()
    # but ma_ratio_5_20 should have non-NaN tail
    assert out["trend__sma_ratio_5_20"].iloc[-1] == out["trend__sma_ratio_5_20"].iloc[-1]  # not NaN


# ---------------------------------------------------------------------------
# Panel-feature semantics
# ---------------------------------------------------------------------------


def _build_long_panel(n_symbols: int = 4, n_days: int = 250) -> pd.DataFrame:
    """Build a long-format panel with per-symbol features already computed."""
    pieces: list[pd.DataFrame] = []
    for i in range(n_symbols):
        ohlcv = _synthetic_ohlcv(n_days, seed=i, start_close=100.0 + 10.0 * i)
        feat_pieces = [
            PriceFeatures().compute(ohlcv).reset_index(drop=True),
            MomentumFeatures().compute(ohlcv).reset_index(drop=True),
            VolumeFeatures().compute(ohlcv).reset_index(drop=True),
        ]
        feat = pd.concat(feat_pieces, axis=1)
        feat.insert(0, "bar_date", ohlcv["bar_date"].values)
        feat.insert(0, "symbol", f"SYM{i}")
        # Also keep raw OHLCV columns (panel groups read no raw, but the API expects them).
        feat["open"] = ohlcv["open"].values
        feat["high"] = ohlcv["high"].values
        feat["low"] = ohlcv["low"].values
        feat["close"] = ohlcv["close"].values
        feat["volume"] = ohlcv["volume"].values
        pieces.append(feat)
    return pd.concat(pieces, axis=0, ignore_index=True)


def test_cross_sectional_ranks_in_unit_interval() -> None:
    panel = _build_long_panel(n_symbols=4, n_days=120)
    out = CrossSectionalFeatures().compute(panel)
    rank_cols = [c for c in out.columns if c.startswith("cross_sectional__xs_rank_")]
    assert len(rank_cols) == 5
    for c in rank_cols:
        vals = out[c].dropna().astype(float)
        assert ((vals >= 0.0) & (vals <= 1.0)).all(), f"{c} out of [0,1]"


def test_regime_features_constant_across_symbols_on_each_date() -> None:
    panel = _build_long_panel(n_symbols=4, n_days=120)
    out = RegimeFeatures().compute(panel)
    cols = [c for c in out.columns if c.startswith("regime__")]
    assert len(cols) == EXPECTED_PER_GROUP["regime"]
    # On each bar_date, the regime values should be identical across symbols.
    for c in cols:
        # Drop NaN warmup rows; among non-NaN rows, std per date == 0.
        non_na = out.dropna(subset=[c])
        per_date_std = non_na.groupby("bar_date")[c].std().dropna()
        assert (per_date_std < 1e-9).all(), f"{c} not constant per date"


# ---------------------------------------------------------------------------
# Pipeline end-to-end (against a tiny in-memory DuckDB)
# ---------------------------------------------------------------------------


def _seed_pipeline_db(path: str, n_symbols: int = 3, n_days: int = 280) -> None:
    members_rows: list[dict] = []
    bars_rows: list[dict] = []
    for i in range(n_symbols):
        sym = f"SYM{i}"
        members_rows.append(
            {
                "universe": "SP500",
                "symbol": sym,
                "exchange": "NASDAQ",
                "start_date": date(2022, 1, 1),
                "end_date": None,
                "company_name": sym,
            }
        )
        ohlcv = _synthetic_ohlcv(n_days, seed=i)
        for r in ohlcv.itertuples(index=False):
            bars_rows.append(
                {
                    "symbol": sym,
                    "exchange": "NASDAQ",
                    "bar_date": r.bar_date,
                    "open": float(r.open),
                    "high": float(r.high),
                    "low": float(r.low),
                    "close": float(r.close),
                    "volume": int(r.volume),
                    "close_unadj": None,
                    "source": "test",
                    "ingested_at": datetime(2024, 1, 1),
                }
            )

    with get_conn(path) as conn:
        upsert_membership(pd.DataFrame(members_rows), conn=conn)
        upsert_ohlcv(pd.DataFrame(bars_rows), conn=conn)


def test_build_feature_matrix_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_pipeline_db(db, n_symbols=3, n_days=280)

    panel = build_feature_matrix(
        "SP500",
        date(2022, 1, 3),
        date(2022, 1, 3) + timedelta(days=279),
        duckdb_path=db,
    )

    assert not panel.empty
    n_feature_cols = sum(1 for c in panel.columns if c not in ("symbol", "bar_date"))
    # Total feature count ≈ 53 (53 = price 8 + trend 10 + momentum 8 + vol 6 + volume 5 + micro 5 + xs 6 + regime 5).
    # We also retain raw OHLCV passthroughs from the per-symbol concat (5 cols).
    # So just assert >= the 53 advertised features.
    assert n_feature_cols >= 53, n_feature_cols
    # Symbols and dates as expected
    assert set(panel["symbol"].unique()) == {"SYM0", "SYM1", "SYM2"}
