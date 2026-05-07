"""Unit tests for the Kite Connect adapter.

Mocks ``KiteConnect`` entirely. For a live integration test, see TestKiteLive
at the bottom; run with ``pytest -m integration tests/unit/test_kite_adapter.py``.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from kiteconnect.exceptions import TokenException

from packages.common.schemas import OHLCV_COLUMNS
from packages.ingestion.adapters import kite_adapter
from packages.ingestion.storage import get_conn, get_ohlcv, upsert_membership

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INSTRUMENT_TOKENS = {
    "RELIANCE": 738561,
    "TCS": 2953217,
    "INFY": 408065,
}


def _fake_instrument_list() -> list[dict]:
    """Mirrors the shape of kite.instruments(exchange='NSE')."""
    rows: list[dict] = []
    for sym, tok in _INSTRUMENT_TOKENS.items():
        rows.append(
            {
                "tradingsymbol": sym,
                "instrument_token": tok,
                "segment": "NSE",
                "instrument_type": "EQ",
                "exchange": "NSE",
            }
        )
    # A futures contract that should be filtered out.
    rows.append(
        {
            "tradingsymbol": "RELIANCE25JANFUT",
            "instrument_token": 99999,
            "segment": "NFO-FUT",
            "instrument_type": "FUT",
            "exchange": "NFO",
        }
    )
    return rows


def _fake_kite_bar(
    d: date, o: float, h: float, lo: float, c: float, v: int
) -> dict:
    """Mirror of one element returned by kite.historical_data(..., 'day')."""
    return {
        "date": datetime.combine(d, datetime.min.time(), tzinfo=UTC),
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "volume": v,
    }


def _two_fake_bars() -> list[dict]:
    return [
        _fake_kite_bar(date(2024, 1, 2), 2900.0, 2920.5, 2890.0, 2910.0, 1_500_000),
        _fake_kite_bar(date(2024, 1, 3), 2912.0, 2930.0, 2900.0, 2925.0, 1_700_000),
    ]


@pytest.fixture
def patched_kite(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """A MagicMock KiteConnect with sensible defaults; no real network."""
    monkeypatch.setattr(kite_adapter.settings, "kite_access_token", "fake-token-xyz")
    fake = MagicMock(name="FakeKite")
    fake.instruments.return_value = _fake_instrument_list()
    fake.historical_data.return_value = _two_fake_bars()
    monkeypatch.setattr(kite_adapter, "_create_kite", lambda: fake)
    monkeypatch.setattr(kite_adapter, "_throttle_sleep", lambda s: None)
    return fake


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    path = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(kite_adapter.settings, "duckdb_path", path)
    return path


def _seed_membership(path: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    with get_conn(path) as conn:
        upsert_membership(df, conn=conn)


# ---------------------------------------------------------------------------
# load_instrument_map
# ---------------------------------------------------------------------------


def test_load_instrument_map_filters_to_nse_equities(patched_kite: MagicMock) -> None:
    m = kite_adapter.load_instrument_map(patched_kite)
    assert m == _INSTRUMENT_TOKENS  # FUT row filtered out


def test_load_instrument_map_restricts_to_requested_symbols(
    patched_kite: MagicMock,
) -> None:
    m = kite_adapter.load_instrument_map(patched_kite, symbols=["RELIANCE", "UNKNOWN"])
    assert m == {"RELIANCE": _INSTRUMENT_TOKENS["RELIANCE"]}


# ---------------------------------------------------------------------------
# fetch_daily_bars
# ---------------------------------------------------------------------------


def test_fetch_daily_bars_returns_canonical_schema(patched_kite: MagicMock) -> None:
    df = kite_adapter.fetch_daily_bars(
        "RELIANCE", date(2024, 1, 2), date(2024, 1, 3)
    )

    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 2
    assert (df["symbol"] == "RELIANCE").all()
    assert (df["exchange"] == "NSE").all()
    assert (df["source"] == "kite").all()
    # close_unadj == close (no separate unadjusted series from Kite)
    assert (df["close_unadj"] == df["close"]).all()
    assert df["bar_date"].iloc[0] == date(2024, 1, 2)
    assert df["close"].iloc[1] == pytest.approx(2925.0)


def test_fetch_daily_bars_validates_date_range(patched_kite: MagicMock) -> None:
    with pytest.raises(ValueError, match="must be <="):
        kite_adapter.fetch_daily_bars(
            "RELIANCE", date(2024, 1, 5), date(2024, 1, 1)
        )


def test_fetch_daily_bars_unknown_symbol_returns_empty(
    patched_kite: MagicMock,
) -> None:
    df = kite_adapter.fetch_daily_bars(
        "NOPE", date(2024, 1, 2), date(2024, 1, 3)
    )
    assert df.empty
    assert list(df.columns) == OHLCV_COLUMNS


def test_fetch_daily_bars_chunks_long_windows(patched_kite: MagicMock) -> None:
    """Ranges > _MAX_DAYS_PER_CALL should split into multiple historical_data calls."""
    start = date(2014, 1, 1)
    end = date(2024, 1, 1)  # ~3653 days → expect 2 chunks at 2000-day cap
    kite_adapter.fetch_daily_bars(
        "RELIANCE",
        start,
        end,
        kite=patched_kite,
        instrument_token=_INSTRUMENT_TOKENS["RELIANCE"],
    )
    assert patched_kite.historical_data.call_count == 2
    chunks = [
        (call.args[1], call.args[2]) for call in patched_kite.historical_data.call_args_list
    ]
    # Chunks contiguous and each within cap
    assert chunks[0][0] == start
    assert (chunks[0][1] - chunks[0][0]).days < kite_adapter._MAX_DAYS_PER_CALL
    assert chunks[1][0] == chunks[0][1] + timedelta(days=1)
    assert chunks[1][1] == end


# ---------------------------------------------------------------------------
# backfill_universe
# ---------------------------------------------------------------------------


def test_backfill_universe_aggregates_succ_and_fail(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_membership(
        tmp_db,
        [
            {"universe": "NIFTY100", "symbol": "RELIANCE", "exchange": "NSE",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Reliance Industries"},
            {"universe": "NIFTY100", "symbol": "TCS", "exchange": "NSE",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Tata Consultancy"},
        ],
    )
    monkeypatch.setattr(kite_adapter.settings, "kite_access_token", "fake-token-xyz")

    fake = MagicMock()
    fake.instruments.return_value = _fake_instrument_list()

    def _hist(token, *args, **kwargs):
        if token == _INSTRUMENT_TOKENS["RELIANCE"]:
            return _two_fake_bars()
        raise RuntimeError("simulated kite error")

    fake.historical_data.side_effect = _hist
    monkeypatch.setattr(kite_adapter, "_create_kite", lambda: fake)
    monkeypatch.setattr(kite_adapter, "_throttle_sleep", lambda s: None)

    result = kite_adapter.backfill_universe(
        "NIFTY100", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )

    assert result["requested"] == 2
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["rows_written"] == 2
    assert len(result["failures"]) == 1
    assert result["failures"][0]["symbol"] == "TCS"


def test_backfill_universe_skips_unknown_symbol(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_membership(
        tmp_db,
        [
            {"universe": "NIFTY100", "symbol": "RELIANCE", "exchange": "NSE",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Reliance Industries"},
            {"universe": "NIFTY100", "symbol": "ZZZNONEXIST", "exchange": "NSE",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Phantom"},
        ],
    )
    monkeypatch.setattr(kite_adapter.settings, "kite_access_token", "fake-token-xyz")

    fake = MagicMock()
    fake.instruments.return_value = _fake_instrument_list()
    fake.historical_data.return_value = _two_fake_bars()
    monkeypatch.setattr(kite_adapter, "_create_kite", lambda: fake)
    monkeypatch.setattr(kite_adapter, "_throttle_sleep", lambda s: None)

    result = kite_adapter.backfill_universe(
        "NIFTY100", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )

    assert result["requested"] == 2
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["failures"][0]["symbol"] == "ZZZNONEXIST"
    assert "instrument map" in result["failures"][0]["error"]


def test_backfill_universe_idempotent(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_membership(
        tmp_db,
        [
            {"universe": "NIFTY100", "symbol": "RELIANCE", "exchange": "NSE",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Reliance"},
        ],
    )
    monkeypatch.setattr(kite_adapter.settings, "kite_access_token", "fake-token-xyz")

    fake = MagicMock()
    fake.instruments.return_value = _fake_instrument_list()
    fake.historical_data.return_value = _two_fake_bars()
    monkeypatch.setattr(kite_adapter, "_create_kite", lambda: fake)
    monkeypatch.setattr(kite_adapter, "_throttle_sleep", lambda s: None)

    r1 = kite_adapter.backfill_universe(
        "NIFTY100", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )
    r2 = kite_adapter.backfill_universe(
        "NIFTY100", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )

    assert r1["rows_written"] == 2
    assert r2["rows_written"] == 2

    with get_conn(tmp_db) as conn:
        out = get_ohlcv("RELIANCE", conn=conn)
    assert len(out) == 2
    bar_dates = {pd.Timestamp(d).date() for d in out["bar_date"]}
    assert bar_dates == {date(2024, 1, 2), date(2024, 1, 3)}


def test_backfill_universe_token_expired_at_load_aborts(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the instrument-map load itself raises TokenException, abort cleanly."""
    _seed_membership(
        tmp_db,
        [
            {"universe": "NIFTY100", "symbol": "RELIANCE", "exchange": "NSE",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Reliance"},
        ],
    )
    monkeypatch.setattr(kite_adapter.settings, "kite_access_token", "fake-token-xyz")

    fake = MagicMock()
    fake.instruments.side_effect = TokenException("Token is invalid or has expired.")
    monkeypatch.setattr(kite_adapter, "_create_kite", lambda: fake)
    monkeypatch.setattr(kite_adapter, "_throttle_sleep", lambda s: None)

    result = kite_adapter.backfill_universe(
        "NIFTY100", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )

    # Aborts cleanly with empty counts; no historical_data call attempted.
    assert result == {
        "requested": 1,
        "succeeded": 0,
        "failed": 0,
        "rows_written": 0,
        "failures": [],
    }
    fake.historical_data.assert_not_called()


def test_backfill_universe_token_expired_mid_run_aborts(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If token expires while iterating, abort: don't continue to remaining symbols."""
    _seed_membership(
        tmp_db,
        [
            {"universe": "NIFTY100", "symbol": "RELIANCE", "exchange": "NSE",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Reliance"},
            {"universe": "NIFTY100", "symbol": "TCS", "exchange": "NSE",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Tata Consultancy"},
        ],
    )
    monkeypatch.setattr(kite_adapter.settings, "kite_access_token", "fake-token-xyz")

    fake = MagicMock()
    fake.instruments.return_value = _fake_instrument_list()
    fake.historical_data.side_effect = TokenException("expired mid-run")
    monkeypatch.setattr(kite_adapter, "_create_kite", lambda: fake)
    monkeypatch.setattr(kite_adapter, "_throttle_sleep", lambda s: None)

    result = kite_adapter.backfill_universe(
        "NIFTY100", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )

    # First symbol fails with token expiry, run aborts before second symbol.
    assert result["requested"] == 2
    assert result["succeeded"] == 0
    assert result["failed"] == 1
    assert "token expired" in result["failures"][0]["error"]
    # Only one historical_data call attempted (the failing one).
    assert fake.historical_data.call_count == 1


def test_kite_connection_raises_when_token_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env-var token AND no fallback session file → connection raises."""
    monkeypatch.setattr(kite_adapter.settings, "kite_access_token", "")
    # Point the fallback path at a directory we know is empty.
    monkeypatch.setattr(
        kite_adapter.settings,
        "kite_session_path",
        str(tmp_path / "no-such-kite_session.json"),
    )
    with (
        pytest.raises(RuntimeError, match="No Kite access token"),
        kite_adapter._kite_connection(),
    ):
        pass


# ---------------------------------------------------------------------------
# Live integration (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestKiteLive:
    """Live Kite tests. Require KITE_API_KEY and a fresh KITE_ACCESS_TOKEN.

    Run with: pytest -m integration tests/unit/test_kite_adapter.py
    """

    def test_fetch_real_reliance_recent(self) -> None:
        if not (os.getenv("KITE_API_KEY") and os.getenv("KITE_ACCESS_TOKEN")):
            pytest.skip("KITE_API_KEY / KITE_ACCESS_TOKEN not set")
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=10)
        df = kite_adapter.fetch_daily_bars("RELIANCE", start, end)
        assert not df.empty
        assert (df["close"] > 0).all()
        assert df["bar_date"].is_monotonic_increasing
