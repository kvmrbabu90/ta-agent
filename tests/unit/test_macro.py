"""Tests for macro ingestion + the conditionally-registered macro feature group."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.common.config import settings
from packages.features import pipeline as pipeline_mod
from packages.features.macro import MacroFeatures
from packages.features.pipeline import build_feature_matrix
from packages.ingestion import macro as macro_mod
from packages.ingestion.macro import (
    SERIES_TICKERS,
    backfill_macro,
    has_macro_data,
    load_macro_series,
    upsert_macro,
)
from packages.ingestion.storage import get_conn, upsert_membership, upsert_ohlcv

# ---------------------------------------------------------------------------
# Macro adapter
# ---------------------------------------------------------------------------


def _fake_yf_frame(closes: list[float], start: date) -> pd.DataFrame:
    idx = pd.to_datetime([start + timedelta(days=i) for i in range(len(closes))])
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [0] * len(closes),
        },
        index=idx,
    )


def test_backfill_macro_writes_both_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)

    def _fake_download(ticker, start, end):
        if ticker == "^VIX":
            return _fake_yf_frame([15.0, 16.0, 14.0, 17.0], start)
        if ticker == "INR=X":
            return _fake_yf_frame([83.0, 83.1, 82.9, 83.2], start)
        return pd.DataFrame()

    monkeypatch.setattr(macro_mod, "_download", _fake_download)

    result = backfill_macro(date(2024, 1, 2), date(2024, 1, 5))
    assert result["rows_written"] == 8  # 4 days * 2 series
    assert sorted(result["series"]) == ["usd_inr", "vix"]
    assert has_macro_data() is True

    vix = load_macro_series("vix", date(2024, 1, 2), date(2024, 1, 5))
    assert len(vix) == 4
    assert vix.iloc[3] == pytest.approx(17.0)


def test_upsert_macro_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)

    df = pd.DataFrame(
        [
            {"macro_date": date(2024, 1, 2), "series": "vix", "value": 15.0,
             "source": "test", "ingested_at": datetime(2024, 1, 3)},
        ]
    )
    n1 = upsert_macro(df)
    n2 = upsert_macro(df)
    assert n1 == 1 and n2 == 1
    out = load_macro_series("vix", date(2024, 1, 1), date(2024, 1, 5))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Macro features
# ---------------------------------------------------------------------------


def test_macro_features_broadcast_per_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)

    # Seed enough VIX history that vix_chg_5d has a value on the panel dates.
    n = 30
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(0)
    vix_values = 15.0 + rng.normal(0, 1, size=n)
    fx_values = 83.0 + rng.normal(0, 0.05, size=n)
    rows = []
    for d, v, f in zip(dates, vix_values, fx_values, strict=True):
        rows.append(
            {"macro_date": d, "series": "vix", "value": float(v),
             "source": "test", "ingested_at": datetime(2024, 1, 1)}
        )
        rows.append(
            {"macro_date": d, "series": "usd_inr", "value": float(f),
             "source": "test", "ingested_at": datetime(2024, 1, 1)}
        )
    upsert_macro(pd.DataFrame(rows))

    # Build a panel of (symbol, bar_date) for two symbols on the last 10 dates.
    panel_dates = dates[-10:]
    panel = pd.DataFrame(
        [
            {"symbol": s, "bar_date": d, "open": 1.0, "high": 1.0, "low": 1.0,
             "close": 1.0, "volume": 1}
            for s in ("A", "B")
            for d in panel_dates
        ]
    )

    out = MacroFeatures(duckdb_path=db).compute(panel)
    cols = [c for c in out.columns if c.startswith("macro__")]
    assert len(cols) == 3
    # Within each bar_date, the macro values must be identical across symbols.
    for c in cols:
        per_date_std = out.groupby("bar_date")[c].std().dropna()
        assert (per_date_std < 1e-9).all(), f"{c} not constant per date"


# ---------------------------------------------------------------------------
# Pipeline conditional registration
# ---------------------------------------------------------------------------


def _seed_pipeline_universe(path: str, n_symbols: int = 3, n_days: int = 80) -> None:
    members: list[dict] = []
    bars: list[dict] = []
    for i in range(n_symbols):
        sym = f"SYM{i}"
        members.append(
            {"universe": "SP500", "symbol": sym, "exchange": "NASDAQ",
             "start_date": date(2024, 1, 1), "end_date": None, "company_name": sym}
        )
        rng = np.random.default_rng(i)
        ret = rng.normal(0.0, 0.012, size=n_days)
        close = 100.0 * np.exp(np.cumsum(ret))
        for j in range(n_days):
            bars.append(
                {
                    "symbol": sym, "exchange": "NASDAQ",
                    "bar_date": date(2024, 1, 1) + timedelta(days=j),
                    "open": float(close[j]) * 0.999,
                    "high": float(close[j]) * 1.005,
                    "low": float(close[j]) * 0.995,
                    "close": float(close[j]),
                    "volume": 1_000_000,
                    "close_unadj": None, "source": "test",
                    "ingested_at": datetime(2024, 1, 1),
                }
            )
    with get_conn(path) as conn:
        upsert_membership(pd.DataFrame(members), conn=conn)
        upsert_ohlcv(pd.DataFrame(bars), conn=conn)


def test_pipeline_skips_macro_when_no_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_pipeline_universe(db)
    # No macro_daily rows seeded.
    panel = build_feature_matrix(
        "SP500", date(2024, 1, 1), date(2024, 1, 1) + timedelta(days=79), duckdb_path=db
    )
    macro_cols = [c for c in panel.columns if c.startswith("macro__")]
    assert macro_cols == []


def test_pipeline_includes_macro_when_data_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_pipeline_universe(db)

    # Seed minimal macro data covering the panel range.
    rows = []
    for i in range(80):
        d = date(2024, 1, 1) + timedelta(days=i)
        rows.append({"macro_date": d, "series": "vix", "value": 15.0 + (i % 5),
                     "source": "test", "ingested_at": datetime(2024, 1, 1)})
        rows.append({"macro_date": d, "series": "usd_inr", "value": 83.0 + (i % 3) * 0.1,
                     "source": "test", "ingested_at": datetime(2024, 1, 1)})
    upsert_macro(pd.DataFrame(rows))

    panel = build_feature_matrix(
        "SP500", date(2024, 1, 1), date(2024, 1, 1) + timedelta(days=79), duckdb_path=db
    )
    macro_cols = [c for c in panel.columns if c.startswith("macro__")]
    assert sorted(macro_cols) == [
        "macro__fx_ret_5d",
        "macro__vix_chg_5d",
        "macro__vix_level_z_252",
    ]


def test_known_series_tickers_match() -> None:
    assert SERIES_TICKERS["vix"] == "^VIX"
    assert SERIES_TICKERS["usd_inr"] == "INR=X"


def test_pipeline_resolve_panel_groups_handles_macro_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If has_macro_data raises (e.g. table missing), pipeline must still run."""
    monkeypatch.setattr(
        pipeline_mod, "has_macro_data",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    groups = pipeline_mod._resolve_panel_groups(duckdb_path=None)
    # Two base groups present, no macro group appended.
    assert len(groups) == 2
