"""Mandatory causality (no-look-ahead) test for the feature pipeline.

Strategy: build a feature matrix, then corrupt the OHLCV in DuckDB to NaN
for all dates after a chosen cutoff, rebuild the matrix, and assert that
features for dates ≤ cutoff are byte-identical between the two runs.

If any feature changes when only-future data is corrupted, that feature has
a look-ahead bug.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.common.config import settings
from packages.features.pipeline import build_feature_matrix
from packages.ingestion.storage import get_conn, upsert_membership, upsert_ohlcv

_N_SYMBOLS = 3
_N_DAYS = 320  # ≥ 252 so 52w-high features have data on the post-cutoff portion
_CUTOFF_OFFSET = 240  # cutoff index inside the synthetic series
_START_DATE = date(2022, 1, 3)


def _synthetic_ohlcv(symbol_idx: int) -> pd.DataFrame:
    rng = np.random.default_rng(symbol_idx)
    daily_ret = rng.normal(0.0005, 0.012, size=_N_DAYS)
    close = (100.0 + 10.0 * symbol_idx) * np.exp(np.cumsum(daily_ret))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=_N_DAYS))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=_N_DAYS)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=_N_DAYS)))
    volume = rng.integers(1_000_000, 5_000_000, size=_N_DAYS)
    bar_dates = [_START_DATE + timedelta(days=i) for i in range(_N_DAYS)]
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


def _seed(path: str, *, corrupt_after: date | None = None) -> None:
    """Populate the DB with synthetic data; if corrupt_after is set, replace
    OHLC for dates STRICTLY GREATER THAN it with extreme bogus values."""
    members: list[dict] = []
    bars: list[dict] = []
    for i in range(_N_SYMBOLS):
        sym = f"SYM{i}"
        members.append(
            {
                "universe": "SP500",
                "symbol": sym,
                "exchange": "NASDAQ",
                "start_date": _START_DATE,
                "end_date": None,
                "company_name": sym,
            }
        )
        ohlcv = _synthetic_ohlcv(i)
        for r in ohlcv.itertuples(index=False):
            o, h, lo, c, v = r.open, r.high, r.low, r.close, r.volume
            if corrupt_after is not None and r.bar_date > corrupt_after:
                # Replace with extreme values that would catastrophically
                # contaminate any rolling stat that mistakenly looks ahead.
                o = h = lo = c = 1.0e6
                v = 99
            bars.append(
                {
                    "symbol": sym, "exchange": "NASDAQ", "bar_date": r.bar_date,
                    "open": float(o), "high": float(h), "low": float(lo),
                    "close": float(c), "volume": int(v),
                    "close_unadj": None,
                    "source": "test",
                    "ingested_at": datetime(2024, 1, 1),
                }
            )

    with get_conn(path) as conn:
        upsert_membership(pd.DataFrame(members), conn=conn)
        upsert_ohlcv(pd.DataFrame(bars), conn=conn)


def test_no_lookahead_in_any_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutoff = _START_DATE + timedelta(days=_CUTOFF_OFFSET)

    db_clean = str(tmp_path / "clean.duckdb")
    db_corrupt = str(tmp_path / "corrupt.duckdb")

    monkeypatch.setattr(settings, "duckdb_path", db_clean)
    _seed(db_clean)
    panel_clean = build_feature_matrix(
        "SP500",
        _START_DATE,
        _START_DATE + timedelta(days=_N_DAYS - 1),
        duckdb_path=db_clean,
    )

    monkeypatch.setattr(settings, "duckdb_path", db_corrupt)
    _seed(db_corrupt, corrupt_after=cutoff)
    panel_corrupt = build_feature_matrix(
        "SP500",
        _START_DATE,
        _START_DATE + timedelta(days=_N_DAYS - 1),
        duckdb_path=db_corrupt,
    )

    # Subset both to dates <= cutoff and align rows.
    def _pre_cutoff(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
        return (
            df[df["bar_date"] <= cutoff]
            .sort_values(["symbol", "bar_date"])
            .reset_index(drop=True)
        )

    pre_clean = _pre_cutoff(panel_clean)
    pre_corrupt = _pre_cutoff(panel_corrupt)

    assert len(pre_clean) == len(pre_corrupt) > 0

    # Per-feature equality check; any column that disagrees is a leakage bug.
    feature_cols = [c for c in pre_clean.columns if c not in ("symbol", "bar_date")]
    leaking: list[tuple[str, int]] = []
    for col in feature_cols:
        a = pd.to_numeric(pre_clean[col], errors="coerce")
        b = pd.to_numeric(pre_corrupt[col], errors="coerce")
        # Treat NaN==NaN as equal
        both_nan = a.isna() & b.isna()
        eq = np.isclose(a.fillna(0.0), b.fillna(0.0), atol=1e-9, rtol=1e-9) | both_nan
        if not eq.all():
            n_diff = int((~eq).sum())
            leaking.append((col, n_diff))

    assert not leaking, (
        "Feature(s) changed when only post-cutoff OHLCV was corrupted — "
        f"look-ahead bug suspected: {leaking[:10]}"
    )
