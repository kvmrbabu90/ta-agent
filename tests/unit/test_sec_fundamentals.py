"""Tests for SEC EDGAR fundamentals adapter + features."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from packages.common.config import settings
from packages.features.sec_fundamentals import SecFundamentalsFeatures
from packages.ingestion.sec_fundamentals import (
    _extract_concept,
    has_fundamentals_data,
    load_fundamentals,
    upsert_fundamentals,
)


def _seed_fundamentals(db: str, rows: list[dict]) -> None:
    import duckdb

    from packages.ingestion.sec_fundamentals import _FUNDAMENTALS_DDL

    df = pd.DataFrame(rows)
    df["source"] = "test"
    df["ingested_at"] = datetime(2024, 1, 1)
    conn = duckdb.connect(db)
    conn.execute(_FUNDAMENTALS_DDL)
    upsert_fundamentals(df, conn=conn)
    conn.close()


def _fund_row(**kwargs) -> dict:
    base = {
        "cik": "0000000123", "symbol": "AAA", "concept": "NetIncomeLoss",
        "period_end": date(2023, 12, 31), "fp": "FY", "fy": 2023,
        "value": 1_000_000.0, "form": "10-K",
        "accession": "ACC-001", "filed_date": date(2024, 2, 15),
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Fallback chain + dedup
# ---------------------------------------------------------------------------


def test_extract_concept_dedups_by_period_with_later_fallback_winning() -> None:
    """When fallbacks have overlapping (period_end, fp), the later fallback wins."""
    facts = {
        "us-gaap": {
            "Revenues": {"units": {"USD": [
                {"val": 100, "end": "2018-12-31", "fp": "FY", "filed": "2019-02-15",
                 "form": "10-K", "accn": "OLD"},
            ]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                # SAME period_end + fp as above; should overwrite
                {"val": 105, "end": "2018-12-31", "fp": "FY", "filed": "2019-02-15",
                 "form": "10-K", "accn": "NEW"},
                # Different period — both kept
                {"val": 110, "end": "2019-12-31", "fp": "FY", "filed": "2020-02-15",
                 "form": "10-K", "accn": "NEW2"},
            ]}},
        }
    }
    rows = _extract_concept(
        facts, "Revenues",
        fallbacks=["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    )
    by_pe = {(r["period_end"], r["fp"]): r["value"] for r in rows}
    assert by_pe[("2018-12-31", "FY")] == 105      # later fallback won
    assert by_pe[("2019-12-31", "FY")] == 110
    assert len(rows) == 2


def test_extract_concept_skips_observations_with_missing_required_fields() -> None:
    """Observations with no val, end, or filed are dropped silently."""
    facts = {
        "us-gaap": {
            "NetIncomeLoss": {"units": {"USD": [
                {"val": 100, "end": "2024-12-31", "fp": "FY", "filed": "2025-02-15",
                 "form": "10-K", "accn": "A"},
                {"val": None, "end": "2023-12-31", "fp": "FY", "filed": "2024-02-15",
                 "form": "10-K", "accn": "B"},   # no value
                {"val": 200, "end": None, "fp": "FY", "filed": "2025-02-15",
                 "form": "10-K", "accn": "C"},   # no period_end
                {"val": 300, "end": "2022-12-31", "fp": "FY", "filed": None,
                 "form": "10-K", "accn": "D"},   # no filed_date
            ]}},
        }
    }
    rows = _extract_concept(facts, "NetIncomeLoss", fallbacks=["NetIncomeLoss"])
    assert len(rows) == 1
    assert rows[0]["value"] == 100


def test_extract_concept_coerces_explicit_none_fp_to_question_mark() -> None:
    """obs.get('fp', '?') doesn't fire when fp is present-but-None — we must
    coerce explicitly. Otherwise the NOT NULL DDL constraint fires."""
    facts = {
        "us-gaap": {
            "NetIncomeLoss": {"units": {"USD": [
                {"val": 100, "end": "2024-12-31", "fp": None, "filed": "2025-02-15",
                 "form": "10-K", "accn": "A"},
            ]}},
        }
    }
    rows = _extract_concept(facts, "NetIncomeLoss", fallbacks=["NetIncomeLoss"])
    assert rows[0]["fp"] == "?"


# ---------------------------------------------------------------------------
# Adapter — upsert + load
# ---------------------------------------------------------------------------


def test_upsert_fundamentals_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "f.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    rows = [_fund_row()]
    _seed_fundamentals(db, rows)
    _seed_fundamentals(db, rows)
    out = load_fundamentals(["AAA"], duckdb_path=db)
    assert len(out) == 1


def test_has_fundamentals_data_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "f.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    assert has_fundamentals_data(duckdb_path=db) is False
    _seed_fundamentals(db, [_fund_row()])
    assert has_fundamentals_data(duckdb_path=db) is True


def test_load_fundamentals_filters_by_fp_and_concept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "f.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_fundamentals(db, [
        _fund_row(concept="NetIncomeLoss",      fp="FY", accession="A1"),
        _fund_row(concept="NetIncomeLoss",      fp="Q1", accession="A2"),
        _fund_row(concept="StockholdersEquity", fp="FY", accession="A3"),
    ])
    out_fy = load_fundamentals(["AAA"], fp=["FY"], duckdb_path=db)
    assert len(out_fy) == 2
    out_concept = load_fundamentals(["AAA"], concepts=["NetIncomeLoss"], duckdb_path=db)
    assert len(out_concept) == 2
    out_both = load_fundamentals(
        ["AAA"], concepts=["NetIncomeLoss"], fp=["FY"], duckdb_path=db,
    )
    assert len(out_both) == 1


# ---------------------------------------------------------------------------
# Feature group
# ---------------------------------------------------------------------------


def _seed_complete_company(db: str, symbol: str = "AAA") -> None:
    """Seed a company with all 6 required FY concepts so all 5 features can compute."""
    rows = [
        _fund_row(symbol=symbol, concept="NetIncomeLoss",      value=1_000.0,
                  filed_date=date(2024, 2, 1), accession="A_NI"),
        _fund_row(symbol=symbol, concept="StockholdersEquity", value=10_000.0,
                  filed_date=date(2024, 2, 1), accession="A_SE"),
        _fund_row(symbol=symbol, concept="Assets",             value=20_000.0,
                  filed_date=date(2024, 2, 1), accession="A_AS"),
        _fund_row(symbol=symbol, concept="LongTermDebt",       value=5_000.0,
                  filed_date=date(2024, 2, 1), accession="A_LD"),
        _fund_row(symbol=symbol, concept="GrossProfit",        value=4_000.0,
                  filed_date=date(2024, 2, 1), accession="A_GP"),
        _fund_row(symbol=symbol, concept="Revenues",           value=10_000.0,
                  filed_date=date(2024, 2, 1), accession="A_RV"),
    ]
    _seed_fundamentals(db, rows)


def test_features_use_only_filings_filed_before_bar_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Causality: a bar_date BEFORE the filed_date must NOT see the data."""
    db = str(tmp_path / "f.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_complete_company(db)

    panel = pd.DataFrame([
        {"symbol": "AAA", "bar_date": date(2024, 1, 15),  # BEFORE filed_date 2024-02-01
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"symbol": "AAA", "bar_date": date(2024, 2, 5),   # AFTER filed_date
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ])
    out = SecFundamentalsFeatures(duckdb_path=db).compute(panel).set_index("bar_date")
    # Pre-filing: all features NaN
    pre = out.loc[date(2024, 1, 15)]
    for c in ("roe", "debt_to_equity", "gross_margin", "net_margin", "asset_turnover"):
        assert pd.isna(pre[f"sec_fundamentals__{c}"])
    # Post-filing: all features computed
    post = out.loc[date(2024, 2, 5)]
    assert post["sec_fundamentals__roe"] == pytest.approx(1_000 / 10_000)
    assert post["sec_fundamentals__debt_to_equity"] == pytest.approx(5_000 / 10_000)
    assert post["sec_fundamentals__gross_margin"] == pytest.approx(4_000 / 10_000)
    assert post["sec_fundamentals__net_margin"] == pytest.approx(1_000 / 10_000)
    assert post["sec_fundamentals__asset_turnover"] == pytest.approx(10_000 / 20_000)


def test_features_handle_missing_concepts_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a company is missing some concepts (e.g. no GrossProfit), the
    affected features come back NaN — but unaffected features still compute."""
    db = str(tmp_path / "f.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    # Only seed NetIncomeLoss + StockholdersEquity — roe should compute, others NaN.
    _seed_fundamentals(db, [
        _fund_row(symbol="X", concept="NetIncomeLoss",      value=500.0,
                  filed_date=date(2024, 1, 1), accession="X1"),
        _fund_row(symbol="X", concept="StockholdersEquity", value=5_000.0,
                  filed_date=date(2024, 1, 1), accession="X2"),
    ])
    panel = pd.DataFrame([{
        "symbol": "X", "bar_date": date(2024, 6, 1),
        "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
    }])
    out = SecFundamentalsFeatures(duckdb_path=db).compute(panel).iloc[0]
    assert out["sec_fundamentals__roe"] == pytest.approx(500 / 5_000)
    assert pd.isna(out["sec_fundamentals__gross_margin"])
    assert pd.isna(out["sec_fundamentals__asset_turnover"])


def test_features_no_data_for_symbol_returns_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(tmp_path / "f.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    _seed_complete_company(db, symbol="AAA")
    panel = pd.DataFrame([{
        "symbol": "ZZZ", "bar_date": date(2024, 6, 1),
        "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
    }])
    out = SecFundamentalsFeatures(duckdb_path=db).compute(panel).iloc[0]
    for c in ("roe", "debt_to_equity", "gross_margin", "net_margin", "asset_turnover"):
        assert pd.isna(out[f"sec_fundamentals__{c}"]), c


def test_features_use_most_recent_filing_when_multiple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When two FY 10-Ks have been filed (2023 then 2024), bar_date should
    use the most recent."""
    db = str(tmp_path / "f.duckdb")
    monkeypatch.setattr(settings, "duckdb_path", db)
    rows = [
        _fund_row(symbol="X", concept="NetIncomeLoss",      value=100.0,
                  period_end=date(2022, 12, 31), filed_date=date(2023, 2, 15),
                  fy=2022, accession="X_NI_22"),
        _fund_row(symbol="X", concept="NetIncomeLoss",      value=300.0,
                  period_end=date(2023, 12, 31), filed_date=date(2024, 2, 15),
                  fy=2023, accession="X_NI_23"),
        _fund_row(symbol="X", concept="StockholdersEquity", value=1_000.0,
                  period_end=date(2023, 12, 31), filed_date=date(2024, 2, 15),
                  fy=2023, accession="X_SE_23"),
    ]
    _seed_fundamentals(db, rows)
    # Bar date AFTER second filing — should see the 2023 (val=300) data, not 2022.
    panel = pd.DataFrame([{
        "symbol": "X", "bar_date": date(2024, 5, 1),
        "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
    }])
    out = SecFundamentalsFeatures(duckdb_path=db).compute(panel).iloc[0]
    assert out["sec_fundamentals__roe"] == pytest.approx(300 / 1_000)
