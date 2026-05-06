"""Unit tests for the yfinance adapter."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from packages.common.config import settings
from packages.common.schemas import OHLCV_COLUMNS
from packages.ingestion.adapters import yfinance_adapter as yfa
from packages.ingestion.storage import get_conn, get_ohlcv, upsert_membership

# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------


def test_to_yfinance_symbol_sp500_dot_to_dash() -> None:
    assert yfa.to_yfinance_symbol("BRK.B", "SP500") == "BRK-B"
    assert yfa.to_yfinance_symbol("BF.B", "SP500") == "BF-B"


def test_to_yfinance_symbol_sp500_passthrough() -> None:
    assert yfa.to_yfinance_symbol("AAPL", "SP500") == "AAPL"


def test_to_yfinance_symbol_nifty_appends_ns() -> None:
    assert yfa.to_yfinance_symbol("RELIANCE", "NIFTY100") == "RELIANCE.NS"
    assert yfa.to_yfinance_symbol("TCS", "NIFTY100") == "TCS.NS"


# ---------------------------------------------------------------------------
# fetch_daily_bars
# ---------------------------------------------------------------------------


def _fake_yf_frame(
    closes_adj: list[float], closes_unadj: list[float] | None = None
) -> pd.DataFrame:
    """Build a yfinance-shaped DataFrame for two days."""
    dates = pd.to_datetime([date(2024, 1, 2), date(2024, 1, 3)])
    closes = closes_unadj if closes_unadj is not None else closes_adj
    return pd.DataFrame(
        {
            "Open": [187.0, 187.5],
            "High": [188.5, 189.0],
            "Low": [186.5, 187.0],
            "Close": closes_adj if closes_unadj is None else closes,
            "Volume": [50_000_000, 60_000_000],
        },
        index=dates,
    )


def test_fetch_daily_bars_returns_canonical_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    adj = _fake_yf_frame([186.0, 187.0])
    unadj = _fake_yf_frame([186.0, 187.0], closes_unadj=[200.0, 201.0])

    calls: list[bool] = []

    def _fake_download(symbol, start, end, auto_adjust):
        calls.append(auto_adjust)
        return adj if auto_adjust else unadj

    monkeypatch.setattr(yfa, "_download", _fake_download)
    monkeypatch.setattr(yfa, "_sleep", lambda s: None)

    df = yfa.fetch_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 3), universe="SP500")

    # Both auto_adjust=True and =False were called (two-call pattern).
    assert sorted(calls) == [False, True]

    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 2
    assert (df["source"] == "yfinance").all()
    assert (df["symbol"] == "AAPL").all()
    # close from adjusted, close_unadj from unadjusted
    assert df["close"].iloc[0] == pytest.approx(186.0)
    assert df["close_unadj"].iloc[0] == pytest.approx(200.0)


def test_fetch_daily_bars_unadj_failure_falls_back_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adj = _fake_yf_frame([186.0, 187.0])

    def _fake_download(symbol, start, end, auto_adjust):
        if not auto_adjust:
            raise RuntimeError("yahoo flaky")
        return adj

    monkeypatch.setattr(yfa, "_download", _fake_download)
    monkeypatch.setattr(yfa, "_sleep", lambda s: None)

    df = yfa.fetch_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 3), universe="SP500")

    assert len(df) == 2
    assert df["close_unadj"].isna().all()


def test_fetch_daily_bars_validates_date_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yfa, "_download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(yfa, "_sleep", lambda s: None)
    with pytest.raises(ValueError, match="must be <="):
        yfa.fetch_daily_bars("AAPL", date(2024, 1, 5), date(2024, 1, 1), universe="SP500")


def test_fetch_daily_bars_empty_yf_returns_empty_df(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yfa, "_download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(yfa, "_sleep", lambda s: None)
    df = yfa.fetch_daily_bars("ZZZZ", date(2024, 1, 2), date(2024, 1, 3), universe="SP500")
    assert df.empty
    assert list(df.columns) == OHLCV_COLUMNS


# ---------------------------------------------------------------------------
# backfill_universe (storage roundtrip)
# ---------------------------------------------------------------------------


def test_backfill_universe_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "test.duckdb")
    members = pd.DataFrame(
        [
            {"universe": "SP500", "symbol": "AAPL", "exchange": "NASDAQ",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Apple"},
        ]
    )
    with get_conn(db) as conn:
        upsert_membership(members, conn=conn)

    monkeypatch.setattr(settings, "duckdb_path", db)

    adj = _fake_yf_frame([186.0, 187.0])
    unadj = _fake_yf_frame([186.0, 187.0], closes_unadj=[200.0, 201.0])

    def _fake_download(symbol, start, end, auto_adjust):
        return adj if auto_adjust else unadj

    monkeypatch.setattr(yfa, "_download", _fake_download)
    monkeypatch.setattr(yfa, "_sleep", lambda s: None)

    result = yfa.backfill_universe(
        "SP500", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=db
    )
    assert result["requested"] == 1
    assert result["succeeded"] == 1
    assert result["rows_written"] == 2

    with get_conn(db) as conn:
        out = get_ohlcv("AAPL", conn=conn, source="yfinance")
    assert len(out) == 2
