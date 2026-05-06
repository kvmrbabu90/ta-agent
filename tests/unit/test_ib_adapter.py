"""Unit tests for the Interactive Brokers adapter.

Mocks ib_insync entirely — these tests never touch a real TWS/Gateway.
For a live integration test, see TestIBLive at the bottom; run with
``pytest -m integration tests/unit/test_ib_adapter.py``.
"""

from __future__ import annotations

import types
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from packages.common.schemas import OHLCV_COLUMNS
from packages.ingestion.adapters import ib_adapter
from packages.ingestion.storage import get_conn, get_ohlcv, upsert_membership

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _fake_bar(d: date, o: float, h: float, lo: float, c: float, v: int) -> object:
    """Build a SimpleNamespace that quacks like ib_insync.BarData for util.df."""
    return types.SimpleNamespace(
        date=d, open=o, high=h, low=lo, close=c, volume=v, average=0.0, barCount=0,
    )


def _two_fake_bars() -> list:
    return [
        _fake_bar(date(2024, 1, 2), 187.0, 188.5, 186.5, 187.4, 50_000_000),
        _fake_bar(date(2024, 1, 3), 187.5, 189.0, 187.0, 188.0, 60_000_000),
    ]


@pytest.fixture
def patched_ib(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """A MagicMock IB instance returned by every call to ib_adapter._create_ib."""
    fake_ib = MagicMock(name="FakeIB")
    fake_ib.reqHistoricalData.return_value = _two_fake_bars()
    monkeypatch.setattr(ib_adapter, "_create_ib", lambda: fake_ib)
    monkeypatch.setattr(ib_adapter, "_throttle_sleep", lambda s: None)
    return fake_ib


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A throwaway DuckDB path, also wired into settings.duckdb_path so any
    code path that bypasses an explicit conn (e.g. members_on) finds it."""
    path = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(ib_adapter.settings, "duckdb_path", path)
    return path


def _seed_membership(path: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    with get_conn(path) as conn:
        upsert_membership(df, conn=conn)


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------


def test_wiki_to_ib_symbol_dot_to_space() -> None:
    assert ib_adapter.wiki_to_ib_symbol("BRK.B") == "BRK B"
    assert ib_adapter.wiki_to_ib_symbol("BF.B") == "BF B"


def test_wiki_to_ib_symbol_passthrough() -> None:
    assert ib_adapter.wiki_to_ib_symbol("AAPL") == "AAPL"
    assert ib_adapter.wiki_to_ib_symbol("MSFT") == "MSFT"


# ---------------------------------------------------------------------------
# fetch_daily_bars
# ---------------------------------------------------------------------------


def test_fetch_daily_bars_returns_canonical_schema(patched_ib: MagicMock) -> None:
    df = ib_adapter.fetch_daily_bars(
        "AAPL", date(2024, 1, 2), date(2024, 1, 3), exchange="NASDAQ"
    )

    # Schema and column order
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 2

    # Constants applied uniformly
    assert (df["symbol"] == "AAPL").all()
    assert (df["exchange"] == "NASDAQ").all()
    assert (df["source"] == "ib").all()
    assert df["close_unadj"].isna().all()

    # Data preserved
    assert df["bar_date"].iloc[0] == date(2024, 1, 2)
    assert df["close"].iloc[0] == pytest.approx(187.4)
    assert df["volume"].iloc[1] == 60_000_000


def test_fetch_daily_bars_validates_date_range(patched_ib: MagicMock) -> None:
    with pytest.raises(ValueError, match="must be <="):
        ib_adapter.fetch_daily_bars(
            "AAPL", date(2024, 1, 5), date(2024, 1, 1), exchange="NASDAQ"
        )


def test_fetch_daily_bars_empty_result_returns_empty_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ib = MagicMock()
    fake_ib.reqHistoricalData.return_value = []
    monkeypatch.setattr(ib_adapter, "_create_ib", lambda: fake_ib)
    monkeypatch.setattr(ib_adapter, "_throttle_sleep", lambda s: None)

    df = ib_adapter.fetch_daily_bars(
        "ZZZZ", date(2024, 1, 2), date(2024, 1, 3), exchange="NASDAQ"
    )
    assert df.empty
    assert list(df.columns) == OHLCV_COLUMNS


def test_fetch_daily_bars_maps_brk_b(patched_ib: MagicMock) -> None:
    """Wikipedia symbol BRK.B should reach IB as ticker 'BRK B'."""
    ib_adapter.fetch_daily_bars(
        "BRK.B", date(2024, 1, 2), date(2024, 1, 3), exchange="NYSE"
    )
    contract = patched_ib.reqHistoricalData.call_args.args[0]
    assert contract.symbol == "BRK B"


# ---------------------------------------------------------------------------
# backfill_universe
# ---------------------------------------------------------------------------


def test_backfill_universe_aggregates_succ_and_fail(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_membership(
        tmp_db,
        [
            {"universe": "SP500", "symbol": "AAPL", "exchange": "NASDAQ",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Apple"},
            {"universe": "SP500", "symbol": "BADCO", "exchange": "NASDAQ",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Bad Co"},
        ],
    )

    fake_ib = MagicMock()

    def _req(contract, **kwargs):
        if contract.symbol == "AAPL":
            return _two_fake_bars()
        raise RuntimeError("simulated IB error")

    fake_ib.reqHistoricalData.side_effect = _req
    monkeypatch.setattr(ib_adapter, "_create_ib", lambda: fake_ib)
    monkeypatch.setattr(ib_adapter, "_throttle_sleep", lambda s: None)

    result = ib_adapter.backfill_universe(
        "SP500", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )

    assert result["requested"] == 2
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["rows_written"] == 2
    assert len(result["failures"]) == 1
    assert result["failures"][0]["symbol"] == "BADCO"
    assert "simulated IB error" in result["failures"][0]["error"]


def test_backfill_universe_idempotent(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_membership(
        tmp_db,
        [
            {"universe": "SP500", "symbol": "AAPL", "exchange": "NASDAQ",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Apple"},
        ],
    )

    fake_ib = MagicMock()
    fake_ib.reqHistoricalData.return_value = _two_fake_bars()
    monkeypatch.setattr(ib_adapter, "_create_ib", lambda: fake_ib)
    monkeypatch.setattr(ib_adapter, "_throttle_sleep", lambda s: None)

    r1 = ib_adapter.backfill_universe(
        "SP500", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )
    r2 = ib_adapter.backfill_universe(
        "SP500", date(2024, 1, 2), date(2024, 1, 3), duckdb_path=tmp_db
    )

    assert r1["rows_written"] == 2
    assert r2["rows_written"] == 2

    with get_conn(tmp_db) as conn:
        out = get_ohlcv("AAPL", conn=conn)
    # Two unique (symbol, bar_date, source) rows — second run upserted, didn't duplicate.
    assert len(out) == 2
    bar_dates = {pd.Timestamp(d).date() for d in out["bar_date"]}
    assert bar_dates == {date(2024, 1, 2), date(2024, 1, 3)}


def test_backfill_universe_validates_date_range(tmp_db: str) -> None:
    with pytest.raises(ValueError):
        ib_adapter.backfill_universe(
            "SP500", date(2024, 1, 5), date(2024, 1, 1), duckdb_path=tmp_db
        )


def test_backfill_universe_with_explicit_symbols_skips_membership(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller-provided symbol list should bypass the membership table entirely."""
    fake_ib = MagicMock()
    fake_ib.reqHistoricalData.return_value = _two_fake_bars()
    monkeypatch.setattr(ib_adapter, "_create_ib", lambda: fake_ib)
    monkeypatch.setattr(ib_adapter, "_throttle_sleep", lambda s: None)

    result = ib_adapter.backfill_universe(
        "SP500",
        date(2024, 1, 2),
        date(2024, 1, 3),
        symbols=["NVDA"],
        duckdb_path=tmp_db,
    )
    assert result["requested"] == 1
    assert result["succeeded"] == 1
    assert result["rows_written"] == 2


# ---------------------------------------------------------------------------
# daily_update
# ---------------------------------------------------------------------------


def test_daily_update_starts_from_day_after_last_bar(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """daily_update must request bars starting the day AFTER the latest stored bar."""
    _seed_membership(
        tmp_db,
        [
            {"universe": "SP500", "symbol": "AAPL", "exchange": "NASDAQ",
             "start_date": date(2010, 1, 1), "end_date": None,
             "company_name": "Apple"},
        ],
    )

    last_stored = date.today() - timedelta(days=3)
    seed = pd.DataFrame(
        [
            {
                "symbol": "AAPL", "exchange": "NASDAQ", "bar_date": last_stored,
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                "volume": 1_000_000, "close_unadj": None,
                "source": "ib", "ingested_at": datetime.utcnow(),
            }
        ]
    )
    from packages.ingestion.storage import upsert_ohlcv as _upsert
    with get_conn(tmp_db) as conn:
        _upsert(seed, conn=conn)

    fake_ib = MagicMock()
    captured: dict = {}

    def _req(contract, **kwargs):
        captured["durationStr"] = kwargs.get("durationStr")
        return _two_fake_bars()

    fake_ib.reqHistoricalData.side_effect = _req
    monkeypatch.setattr(ib_adapter, "_create_ib", lambda: fake_ib)
    monkeypatch.setattr(ib_adapter, "_throttle_sleep", lambda s: None)

    result = ib_adapter.daily_update("SP500", duckdb_path=tmp_db)
    assert result["requested"] == 1
    assert result["succeeded"] == 1
    # Window is (today - last_stored) days
    expected_days = (date.today() - last_stored).days
    assert captured["durationStr"] == f"{expected_days} D"


# ---------------------------------------------------------------------------
# Live integration (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIBLive:
    """Smoke tests against a real TWS / IB Gateway.

    To run:
        1. Start TWS or IB Gateway in paper mode (port 7497 by default).
        2. Enable "ActiveX and Socket Clients" in Global Configuration → API → Settings.
        3. Add the host as a trusted IP if needed.
        4. pytest -m integration tests/unit/test_ib_adapter.py -v
    """

    def test_fetch_real_aapl_recent(self) -> None:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=10)
        df = ib_adapter.fetch_daily_bars("AAPL", start, end, exchange="NASDAQ")
        assert not df.empty
        assert (df["close"] > 0).all()
        assert df["bar_date"].is_monotonic_increasing
