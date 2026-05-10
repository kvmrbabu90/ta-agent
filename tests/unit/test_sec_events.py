"""Tests for SEC EDGAR filings adapter + 8-K event features."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from packages.common.config import settings
from packages.features.sec_events import (
    SecEventsFeatures,
    _filter_8k_by_item,
)
from packages.ingestion.sec_filings import (
    has_filings_data,
    load_filings,
    upsert_filings,
)


def _seed_filings(db: str, rows: list[dict]) -> None:
    import duckdb

    from packages.ingestion.sec_filings import _EDGAR_DDL

    df = pd.DataFrame(rows)
    df["source"] = "test"
    df["ingested_at"] = datetime(2024, 1, 1)
    conn = duckdb.connect(db)
    conn.execute(_EDGAR_DDL)
    upsert_filings(df, conn=conn)
    conn.close()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def test_upsert_filings_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "edgar.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    rows = [{
        "cik": "0000320193", "symbol": "AAPL",
        "accession": "0001-25-001", "filing_date": date(2024, 7, 30),
        "form_type": "8-K", "items": "2.02,9.01", "primary_doc": "doc.htm",
    }]
    _seed_filings(db, rows)
    _seed_filings(db, rows)  # second call should overwrite, not duplicate
    out = load_filings(["AAPL"], duckdb_path=db)
    assert len(out) == 1
    assert out.iloc[0]["accession"] == "0001-25-001"


def test_has_filings_data_reflects_table_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "edgar.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    assert has_filings_data(duckdb_path=db) is False
    _seed_filings(db, [{
        "cik": "0", "symbol": "X", "accession": "A1",
        "filing_date": date(2024, 1, 1), "form_type": "8-K",
        "items": "2.02", "primary_doc": "x.htm",
    }])
    assert has_filings_data(duckdb_path=db) is True


def test_load_filings_filters_by_form_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "edgar.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_filings(db, [
        {"cik": "0", "symbol": "X", "accession": "A1", "filing_date": date(2024, 1, 1),
         "form_type": "8-K", "items": "2.02", "primary_doc": "x.htm"},
        {"cik": "0", "symbol": "X", "accession": "A2", "filing_date": date(2024, 2, 1),
         "form_type": "10-Q", "items": None, "primary_doc": "y.htm"},
        {"cik": "0", "symbol": "X", "accession": "A3", "filing_date": date(2024, 3, 1),
         "form_type": "10-K", "items": None, "primary_doc": "z.htm"},
    ])
    out = load_filings(["X"], form_types=["8-K"], duckdb_path=db)
    assert len(out) == 1 and out.iloc[0]["form_type"] == "8-K"
    out = load_filings(["X"], form_types=["10-Q", "10-K"], duckdb_path=db)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Item-code filter
# ---------------------------------------------------------------------------


def test_filter_8k_by_item_handles_csv_and_partial_match() -> None:
    """The items column is a comma-separated list like '2.02,9.01'.
    Filtering by '2.02' should match that row but NOT match '12.02' (no false-positive)."""
    df = pd.DataFrame([
        {"form_type": "8-K", "items": "2.02,9.01"},
        {"form_type": "8-K", "items": "5.02"},
        {"form_type": "8-K", "items": "1.01,2.02"},
        {"form_type": "8-K", "items": "12.02"},        # red herring (substring match would falsely include)
        {"form_type": "8-K", "items": None},
        {"form_type": "10-Q", "items": "2.02"},        # 10-Q won't pass the form filter
    ])
    out = _filter_8k_by_item(df, "2.02")
    assert len(out) == 2  # only the two real 2.02 8-K rows
    assert out["items"].tolist() == ["2.02,9.01", "1.01,2.02"]


# ---------------------------------------------------------------------------
# Feature group
# ---------------------------------------------------------------------------


def test_sec_events_features_causal_days_since(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Days-since must use only filings with date <= bar_date (causal)."""
    db = str(tmp_path / "edgar.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_filings(db, [{
        "cik": "0", "symbol": "AAPL", "accession": "A1",
        "filing_date": date(2024, 7, 30), "form_type": "8-K",
        "items": "2.02,9.01", "primary_doc": "x.htm",
    }])
    panel = pd.DataFrame([
        {"symbol": "AAPL", "bar_date": date(2024, 7, 25),
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"symbol": "AAPL", "bar_date": date(2024, 7, 30),
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"symbol": "AAPL", "bar_date": date(2024, 7, 31),
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ])
    out = SecEventsFeatures(duckdb_path=db).compute(panel).set_index("bar_date")
    # Pre-event: no past 8-K
    assert pd.isna(out.loc[date(2024, 7, 25), "sec_events__days_since_last_8k"])
    # On the event day: days_since = 0
    assert out.loc[date(2024, 7, 30), "sec_events__days_since_last_8k"] == 0
    # Day after: days_since = 1; 2.02 fired (it was an item 2.02 8-K)
    assert out.loc[date(2024, 7, 31), "sec_events__days_since_last_8k"] == 1
    assert out.loc[date(2024, 7, 31), "sec_events__days_since_last_8k_2_02"] == 1
    # 5.02 wasn't in the items list — should be NaN
    assert pd.isna(out.loc[date(2024, 7, 31), "sec_events__days_since_last_8k_5_02"])


def test_sec_events_count_8k_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """count_8k_last_30d counts filings in (T-30, T]."""
    db = str(tmp_path / "edgar.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_filings(db, [
        {"cik": "0", "symbol": "X", "accession": "A1", "filing_date": date(2024, 6, 1),
         "form_type": "8-K", "items": "2.02", "primary_doc": "x.htm"},
        {"cik": "0", "symbol": "X", "accession": "A2", "filing_date": date(2024, 6, 15),
         "form_type": "8-K", "items": "5.02", "primary_doc": "x.htm"},
        {"cik": "0", "symbol": "X", "accession": "A3", "filing_date": date(2024, 6, 28),
         "form_type": "8-K", "items": "8.01", "primary_doc": "x.htm"},
    ])
    panel = pd.DataFrame([
        {"symbol": "X", "bar_date": date(2024, 6, 30),
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"symbol": "X", "bar_date": date(2024, 7, 5),
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"symbol": "X", "bar_date": date(2024, 8, 1),  # all 3 filings are >30d ago by now
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ])
    out = SecEventsFeatures(duckdb_path=db).compute(panel).set_index("bar_date")
    # Bar 2024-06-30: window (2024-05-31, 2024-06-30] -> all 3 fall in.
    assert out.loc[date(2024, 6, 30), "sec_events__count_8k_last_30d"] == 3
    # Bar 2024-07-05: window (2024-06-05, 2024-07-05] -> 06-01 falls OUT, 06-15 and 06-28 fall in.
    assert out.loc[date(2024, 7, 5),  "sec_events__count_8k_last_30d"] == 2
    # Bar 2024-08-01: window (2024-07-02, 2024-08-01] -> all 3 too old.
    assert out.loc[date(2024, 8, 1),  "sec_events__count_8k_last_30d"] == 0


def test_sec_events_no_data_for_symbol_returns_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "edgar.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_filings(db, [{
        "cik": "0", "symbol": "AAPL", "accession": "A1",
        "filing_date": date(2024, 7, 30), "form_type": "8-K",
        "items": "2.02", "primary_doc": "x.htm",
    }])
    panel = pd.DataFrame([{
        "symbol": "MSFT", "bar_date": date(2024, 7, 30),
        "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
    }])
    out = SecEventsFeatures(duckdb_path=db).compute(panel).iloc[0]
    for c in (
        "days_since_last_8k", "days_since_last_8k_2_02", "days_since_last_8k_5_02",
        "days_since_last_10q", "days_since_last_10k",
    ):
        assert pd.isna(out[f"sec_events__{c}"])
    # count_8k_last_30d returns 0 (not NaN) for symbols with no filings.
    assert out["sec_events__count_8k_last_30d"] == 0


def test_sec_events_handles_empty_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "edgar.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    out = SecEventsFeatures(duckdb_path=db).compute(pd.DataFrame(columns=["symbol", "bar_date"]))
    assert "sec_events__days_since_last_8k" in out.columns
    assert len(out) == 0


def test_sec_events_days_since_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When most recent 8-K is older than _DAYS_CAP_8K (90), days_since = NaN."""
    db = str(tmp_path / "edgar.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_filings(db, [{
        "cik": "0", "symbol": "X", "accession": "A1",
        "filing_date": date(2024, 1, 1), "form_type": "8-K",
        "items": "2.02", "primary_doc": "x.htm",
    }])
    panel = pd.DataFrame([{
        "symbol": "X", "bar_date": date(2024, 5, 1),  # 121 days later
        "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
    }])
    out = SecEventsFeatures(duckdb_path=db).compute(panel)
    assert pd.isna(out.iloc[0]["sec_events__days_since_last_8k"])
