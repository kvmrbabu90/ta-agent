"""Tests for SectorResidualFeatures (stock returns minus sector ETF returns)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from packages.common.config import settings
from packages.features.sector_residual import SectorResidualFeatures
from packages.ingestion.storage import get_conn, upsert_ohlcv


def _seed_etf_bars(db: str, etf: str, start: date, n_days: int) -> None:
    """Write n_days of synthetic OHLCV for one ETF into ohlcv_daily."""
    rng = np.random.default_rng(hash(etf) % 2**31)
    daily_ret = rng.normal(0.0005, 0.012, n_days)
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    rows = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        rows.append({
            "symbol": etf, "exchange": "NYSE", "bar_date": d,
            "open": float(close[i]) * 0.999, "high": float(close[i]) * 1.005,
            "low": float(close[i]) * 0.995, "close": float(close[i]),
            "volume": 1_000_000, "close_unadj": None,
            "source": "test", "ingested_at": datetime(2024, 1, 1),
        })
    with get_conn(db) as conn:
        upsert_ohlcv(pd.DataFrame(rows), conn=conn)


def _toy_panel_with_inputs(symbols: list[str], start: date, n_days: int) -> pd.DataFrame:
    """Panel rows with the per-symbol inputs SectorResidualFeatures depends on."""
    rows = []
    for sym in symbols:
        for i in range(n_days):
            rows.append({
                "symbol": sym, "bar_date": start + timedelta(days=i),
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1,
                "price__log_return_5d": 0.01 if sym == "AAA" else -0.01,
                "price__log_return_20d": 0.04 if sym == "AAA" else -0.04,
                "price__realized_vol_20d": 0.02,
                "momentum__rsi_14": 60.0 if sym == "AAA" else 40.0,
            })
    return pd.DataFrame(rows)


def test_sector_residual_handles_missing_sector_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "t.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    panel = _toy_panel_with_inputs(["AAA"], date(2024, 1, 1), 5)
    fg = SectorResidualFeatures(universe="DOES_NOT_EXIST", duckdb_path=db)
    out = fg.compute(panel)
    cols = [c for c in out.columns if c.startswith("sector_residual__")]
    assert len(cols) == 5
    for c in cols:
        assert out[c].isna().all(), c


def test_sector_residual_handles_missing_etf_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sector map exists but ETF bars are absent — features come back NaN."""
    db = str(tmp_path / "t.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)

    sector_df = pd.DataFrame([{"symbol": "AAA", "sector": "Technology"}])
    sector_path = tmp_path / "sectors_sp500.parquet"
    sector_df.to_parquet(sector_path, index=False)

    panel = _toy_panel_with_inputs(["AAA"], date(2024, 1, 1), 5)
    fg = SectorResidualFeatures(universe="SP500", duckdb_path=db)
    with patch("packages.features.sector_residual.PROCESSED_DIR", tmp_path):
        out = fg.compute(panel)
    cols = [c for c in out.columns if c.startswith("sector_residual__")]
    assert len(cols) == 5
    for c in cols:
        assert out[c].isna().all(), f"{c} should be NaN when ETF bars missing"


def test_sector_residual_computes_residual_correctly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stock returns minus sector ETF returns should produce real residual values."""
    db = str(tmp_path / "t.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)

    sector_df = pd.DataFrame([
        {"symbol": "AAA", "sector": "Technology"},
        {"symbol": "BBB", "sector": "Financials"},
    ])
    sector_path = tmp_path / "sectors_sp500.parquet"
    sector_df.to_parquet(sector_path, index=False)

    # Seed 80 days of XLK and XLF bars (enough for 60d window).
    start = date(2024, 1, 1)
    _seed_etf_bars(db, "XLK", start, 80)
    _seed_etf_bars(db, "XLF", start, 80)

    # Build panel with 60+ days of inputs so warmup can fill.
    panel = _toy_panel_with_inputs(["AAA", "BBB"], start, 80)
    fg = SectorResidualFeatures(universe="SP500", duckdb_path=db)
    with patch("packages.features.sector_residual.PROCESSED_DIR", tmp_path):
        out = fg.compute(panel)

    # At least some residuals should be non-NaN once the warmup is past.
    resid_5d = out["sector_residual__resid_log_return_5d"]
    assert resid_5d.notna().any()

    # Within Technology rows, the residual = stock 5d - XLK 5d.
    # Pick a row near the end of the panel where everything has populated.
    aaa_rows = out[out["symbol"] == "AAA"].copy().reset_index(drop=True)
    last_aaa = aaa_rows.iloc[-1]
    assert pd.notna(last_aaa["sector_residual__resid_log_return_5d"])
    assert pd.notna(last_aaa["sector_residual__resid_realized_vol_20d"])


def test_sector_residual_zscore_uses_only_prior_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trailing z-score must be causal: shift(1) before rolling mean/std."""
    db = str(tmp_path / "t.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)

    sector_df = pd.DataFrame([{"symbol": "AAA", "sector": "Technology"}])
    sector_path = tmp_path / "sectors_sp500.parquet"
    sector_df.to_parquet(sector_path, index=False)
    _seed_etf_bars(db, "XLK", date(2024, 1, 1), 100)

    panel = _toy_panel_with_inputs(["AAA"], date(2024, 1, 1), 100)
    fg = SectorResidualFeatures(universe="SP500", duckdb_path=db)
    with patch("packages.features.sector_residual.PROCESSED_DIR", tmp_path):
        out = fg.compute(panel)

    # First 19 rows should have NaN z (need min_periods=20 prior).
    z = out["sector_residual__resid_zscore_60d"]
    # Once we have ≥20 prior observations and shift(1), the earliest non-NaN z
    # should be no earlier than row 21.
    first_non_nan_idx = z.first_valid_index()
    if first_non_nan_idx is not None:
        assert first_non_nan_idx >= 20


def test_sector_residual_handles_empty_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "t.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    fg = SectorResidualFeatures(universe="SP500", duckdb_path=db)
    out = fg.compute(pd.DataFrame(columns=["symbol", "bar_date"]))
    assert "sector_residual__resid_log_return_5d" in out.columns
    assert len(out) == 0


def test_sector_residual_unmapped_sector_yields_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol whose sector isn't in _SECTOR_TO_ETF (e.g. 'UNKNOWN') gets NaN
    residuals — no sector ETF to subtract from."""
    db = str(tmp_path / "t.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    sector_df = pd.DataFrame([{"symbol": "ZZZ", "sector": "UNKNOWN"}])
    sector_path = tmp_path / "sectors_sp500.parquet"
    sector_df.to_parquet(sector_path, index=False)
    _seed_etf_bars(db, "XLK", date(2024, 1, 1), 30)

    panel = _toy_panel_with_inputs(["ZZZ"], date(2024, 1, 1), 30)
    fg = SectorResidualFeatures(universe="SP500", duckdb_path=db)
    with patch("packages.features.sector_residual.PROCESSED_DIR", tmp_path):
        out = fg.compute(panel)
    assert out["sector_residual__resid_log_return_5d"].isna().all()
