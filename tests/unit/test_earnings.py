"""Unit tests for earnings adapter + earnings feature group."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from packages.common.config import settings
from packages.features.earnings import EarningsFeatures
from packages.ingestion.earnings import (
    has_earnings_data,
    load_earnings,
    upsert_earnings,
)


def _seed_earnings(db: str, rows: list[dict]) -> None:
    """Helper: write rows to a fresh DuckDB, then verify."""
    import duckdb

    from packages.ingestion.earnings import _EARNINGS_DDL

    df = pd.DataFrame(rows)
    df["source"] = "test"
    df["ingested_at"] = datetime(2024, 1, 1)
    conn = duckdb.connect(db)
    conn.execute(_EARNINGS_DDL)
    upsert_earnings(df, conn=conn)
    conn.close()


# ---------------------------------------------------------------------------
# Adapter — upsert + load + has_earnings_data
# ---------------------------------------------------------------------------


def test_upsert_earnings_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "earn.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)

    rows = [
        {"symbol": "AAPL", "announce_date": date(2024, 7, 30), "period_end": None,
         "eps_estimate": 1.34, "eps_actual": 1.40, "surprise_pct": 4.30},
    ]
    _seed_earnings(db, rows)
    _seed_earnings(db, rows)  # second call should overwrite, not duplicate

    out = load_earnings(["AAPL"], duckdb_path=db)
    assert len(out) == 1
    assert out.iloc[0]["surprise_pct"] == pytest.approx(4.30)


def test_has_earnings_data_reflects_table_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "earn.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    assert has_earnings_data(duckdb_path=db) is False

    _seed_earnings(
        db,
        [{"symbol": "AAPL", "announce_date": date(2024, 7, 30), "period_end": None,
          "eps_estimate": 1.34, "eps_actual": 1.40, "surprise_pct": 4.30}],
    )
    assert has_earnings_data(duckdb_path=db) is True


# ---------------------------------------------------------------------------
# Feature group — causality
# ---------------------------------------------------------------------------


def _toy_panel_for_aapl() -> pd.DataFrame:
    """5 trading days around a hypothetical July 30, 2024 earnings."""
    rows = []
    for d in (date(2024, 7, 25), date(2024, 7, 26), date(2024, 7, 29),
              date(2024, 7, 30), date(2024, 7, 31)):
        rows.append({"symbol": "AAPL", "bar_date": d, "open": 1.0, "high": 1.0,
                     "low": 1.0, "close": 1.0, "volume": 1})
    return pd.DataFrame(rows)


def test_earnings_features_causal_days_since_and_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before earnings, days_since must be NaN (no past report);
    after earnings, days_to must be NaN until next one is announced."""
    db = str(tmp_path / "earn.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    # ONE earnings report on 2024-07-30. Before that = no past earnings.
    _seed_earnings(
        db,
        [{"symbol": "AAPL", "announce_date": date(2024, 7, 30), "period_end": None,
          "eps_estimate": 1.34, "eps_actual": 1.40, "surprise_pct": 4.30}],
    )

    panel = _toy_panel_for_aapl()
    out = EarningsFeatures(duckdb_path=db).compute(panel)
    out = out.set_index("bar_date")

    # Pre-event: no past earnings, no future earnings (single record IS the future one for early dates)
    pre = out.loc[date(2024, 7, 25)]
    assert pd.isna(pre["earnings__days_since_last_earnings"])
    assert pre["earnings__days_to_next_earnings"] == 5  # 2024-07-30 - 2024-07-25 = 5

    # On the event day: days_since = 0, days_to = NaN (no later record)
    on = out.loc[date(2024, 7, 30)]
    assert on["earnings__days_since_last_earnings"] == 0
    assert pd.isna(on["earnings__days_to_next_earnings"])
    assert on["earnings__last_surprise_pct"] == pytest.approx(4.30)

    # Day after: days_since = 1
    post = out.loc[date(2024, 7, 31)]
    assert post["earnings__days_since_last_earnings"] == 1
    assert pd.isna(post["earnings__days_to_next_earnings"])


def test_earnings_features_no_data_for_symbol_returns_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol with no earnings rows in the table gets all-NaN features."""
    db = str(tmp_path / "earn.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    # Seed AAPL only.
    _seed_earnings(
        db,
        [{"symbol": "AAPL", "announce_date": date(2024, 7, 30), "period_end": None,
          "eps_estimate": 1.34, "eps_actual": 1.40, "surprise_pct": 4.30}],
    )

    panel = pd.DataFrame(
        [
            {"symbol": "MSFT", "bar_date": date(2024, 7, 30),
             "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
        ]
    )
    out = EarningsFeatures(duckdb_path=db).compute(panel)
    row = out.iloc[0]
    for c in ("days_since_last_earnings", "days_to_next_earnings",
              "last_surprise_pct", "last_surprise_z_252"):
        assert pd.isna(row[f"earnings__{c}"]), c


def test_earnings_features_days_since_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the most recent earnings is more than 90 days old, days_since = NaN
    (the post-earnings drift window has expired)."""
    db = str(tmp_path / "earn.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_earnings(
        db,
        [{"symbol": "AAPL", "announce_date": date(2024, 1, 1), "period_end": None,
          "eps_estimate": 1.0, "eps_actual": 1.0, "surprise_pct": 0.0}],
    )
    # Bar date 100 days later → past the 90-day window.
    panel = pd.DataFrame(
        [
            {"symbol": "AAPL", "bar_date": date(2024, 4, 11),
             "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
        ]
    )
    out = EarningsFeatures(duckdb_path=db).compute(panel)
    assert pd.isna(out.iloc[0]["earnings__days_since_last_earnings"])


def test_earnings_features_surprise_z_score_uses_only_prior_quarters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The z-score at quarter T must use only quarters before T (causal).
    With 5 prior quarters of [+1, +1, +1, +1, +1] surprises, then a +5 surprise:
    on the +5 day, z = (5 - 1) / 0 = undefined (std=0). Need at least 4 prior with variance."""
    db = str(tmp_path / "earn.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    rows = [
        {"symbol": "X", "announce_date": date(2022, 1, 1), "period_end": None,
         "eps_estimate": 1.0, "eps_actual": 1.0, "surprise_pct": 1.0},
        {"symbol": "X", "announce_date": date(2022, 4, 1), "period_end": None,
         "eps_estimate": 1.0, "eps_actual": 1.0, "surprise_pct": 2.0},
        {"symbol": "X", "announce_date": date(2022, 7, 1), "period_end": None,
         "eps_estimate": 1.0, "eps_actual": 1.0, "surprise_pct": 3.0},
        {"symbol": "X", "announce_date": date(2022, 10, 1), "period_end": None,
         "eps_estimate": 1.0, "eps_actual": 1.0, "surprise_pct": 4.0},
        {"symbol": "X", "announce_date": date(2023, 1, 1), "period_end": None,
         "eps_estimate": 1.0, "eps_actual": 1.0, "surprise_pct": 10.0},
    ]
    _seed_earnings(db, rows)
    panel = pd.DataFrame(
        [
            {"symbol": "X", "bar_date": date(2023, 1, 5),  # right after the +10 surprise
             "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
        ]
    )
    out = EarningsFeatures(duckdb_path=db).compute(panel)
    row = out.iloc[0]
    # last_surprise should be 10
    assert row["earnings__last_surprise_pct"] == pytest.approx(10.0)
    # z-score uses prior 4 surprises {1,2,3,4}: mean=2.5, std=~1.29
    # so z = (10 - 2.5) / 1.29 ≈ 5.81
    assert row["earnings__last_surprise_z_252"] == pytest.approx(5.81, abs=0.1)


def test_earnings_features_handles_empty_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "earn.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    out = EarningsFeatures(duckdb_path=db).compute(pd.DataFrame(columns=["symbol", "bar_date"]))
    assert "earnings__days_since_last_earnings" in out.columns
    assert len(out) == 0


# ---------------------------------------------------------------------------
# yfinance fetch helper — uses mocking, no network
# ---------------------------------------------------------------------------


def test_yf_fetch_earnings_handles_yfinance_returning_none() -> None:
    """If yf.Ticker.earnings_dates returns None, we get an empty frame, not a crash."""
    from packages.ingestion.earnings import _yf_fetch_earnings

    class _FakeTicker:
        earnings_dates = None

    with patch("packages.ingestion.earnings.yf.Ticker", return_value=_FakeTicker()):
        out = _yf_fetch_earnings("AAPL", "SP500")
    assert out.empty


def test_yf_fetch_earnings_normalizes_yfinance_response() -> None:
    """A typical yfinance response with mixed NaN/values should round-trip correctly."""
    from packages.ingestion.earnings import _yf_fetch_earnings

    yf_df = pd.DataFrame(
        {
            "EPS Estimate": [1.5, 1.6, np.nan],
            "Reported EPS": [1.7, np.nan, 1.4],
            "Surprise(%)": [13.3, np.nan, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-04-30 16:00", tz="America/New_York"),
                pd.Timestamp("2024-07-30 16:00", tz="America/New_York"),
                pd.Timestamp("2024-10-31 16:00", tz="America/New_York"),
            ]
        ),
    )

    class _FakeTicker:
        def __init__(self) -> None:
            self.earnings_dates = yf_df

    with patch("packages.ingestion.earnings.yf.Ticker", return_value=_FakeTicker()):
        out = _yf_fetch_earnings("AAPL", "SP500")
    assert len(out) == 3
    # Date conversion preserves the calendar day.
    assert out.iloc[0]["announce_date"] == date(2024, 4, 30)
    assert out.iloc[0]["surprise_pct"] == pytest.approx(13.3)
    # NaN preserved (None at write time, pd.NaN/np.nan at read time after pandas coerces).
    assert pd.isna(out.iloc[1]["surprise_pct"])
