"""Unit tests for the FastAPI backend routes."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from packages.common.config import settings
from packages.inference.db import init_predictions_db
from packages.inference.tracking import log_predictions
from packages.ingestion.storage import (
    get_conn as duck_conn,
)
from packages.ingestion.storage import (
    upsert_membership,
    upsert_ohlcv,
)
from services.api.main import app

# ---------------------------------------------------------------------------
# Fixtures: seed both DBs into tmp_path and patch settings to point at them.
# ---------------------------------------------------------------------------


def _seed_membership(duck_path: str, rows: list[dict]) -> None:
    with duck_conn(duck_path) as conn:
        upsert_membership(pd.DataFrame(rows), conn=conn)


def _seed_ohlcv(duck_path: str, rows: list[dict]) -> None:
    with duck_conn(duck_path) as conn:
        upsert_ohlcv(pd.DataFrame(rows), conn=conn)


def _ohlcv_row(symbol: str, d: date, close: float) -> dict:
    return {
        "symbol": symbol,
        "exchange": "NASDAQ",
        "bar_date": d,
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": 1_000_000,
        "close_unadj": None,
        "source": "test",
        "ingested_at": datetime(2024, 1, 1),
    }


def _membership_row(universe: str, symbol: str, company: str) -> dict:
    return {
        "universe": universe,
        "symbol": symbol,
        "exchange": "NASDAQ",
        "start_date": date(2010, 1, 1),
        "end_date": None,
        "company_name": company,
    }


def _prediction_row(
    symbol: str,
    as_of: date,
    universe: str = "SP500",
    predicted_return: float = 0.01,
    predicted_quintile: int = 4,
    top_proba: float = 0.55,
    bot_proba: float = 0.05,
    realized_return: float | None = None,
    realized_quintile: int | None = None,
) -> dict:
    return {
        "universe": universe,
        "symbol": symbol,
        "as_of": as_of,
        "horizon_days": 5,
        "predicted_return": predicted_return,
        "predicted_quintile": predicted_quintile,
        "top_quintile_proba": top_proba,
        "bottom_quintile_proba": bot_proba,
        "model_version_regression": "regv1",
        "model_version_classification": "clsv1",
        "realized_return": realized_return,
        "realized_quintile": realized_quintile,
    }


@pytest.fixture
def seeded_dbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    duck_path = str(tmp_path / "market.duckdb")
    sqlite_path = str(tmp_path / "predictions.sqlite")
    monkeypatch.setattr(settings, "duckdb_path", duck_path)
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)

    _seed_membership(
        duck_path,
        [
            _membership_row("SP500", "AAA", "Alpha Co"),
            _membership_row("SP500", "BBB", "Beta Co"),
            _membership_row("SP500", "CCC", "Gamma Co"),
            _membership_row("NIFTY100", "RELIANCE", "Reliance Industries"),
        ],
    )

    today = date.today()
    bars = []
    for sym, base in [("AAA", 100.0), ("BBB", 200.0), ("CCC", 50.0)]:
        for i in range(10):
            d = today - timedelta(days=10 - i)
            bars.append(_ohlcv_row(sym, d, base + i * 0.5))
    _seed_ohlcv(duck_path, bars)

    init_predictions_db(sqlite_path)

    pred_date = today - timedelta(days=8)
    log_predictions(
        pd.DataFrame(
            [
                _prediction_row("AAA", pred_date, predicted_return=0.05,
                                predicted_quintile=4, top_proba=0.7, bot_proba=0.05),
                _prediction_row("BBB", pred_date, predicted_return=0.01,
                                predicted_quintile=2, top_proba=0.3, bot_proba=0.2),
                _prediction_row("CCC", pred_date, predicted_return=-0.03,
                                predicted_quintile=0, top_proba=0.05, bot_proba=0.6),
            ]
        )
    )

    # Manually populate realized fields (mimicking what settle_predictions
    # would do; we don't have enough OHLCV in the fixture to drive settlement
    # naturally, and that's not the focus of these tests).
    import sqlite3
    conn = sqlite3.connect(sqlite_path)
    realized = [(0.04, 4, "AAA"), (0.02, 3, "BBB"), (-0.02, 0, "CCC")]
    for ret, q, sym in realized:
        conn.execute(
            "UPDATE predictions_log SET realized_return = ?, realized_quintile = ?, settled_at = ? WHERE symbol = ?",
            [ret, q, datetime.utcnow(), sym],
        )
    conn.commit()
    conn.close()

    return duck_path, sqlite_path


@pytest.fixture
def client(seeded_dbs: tuple[str, str]) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /universes
# ---------------------------------------------------------------------------


def test_universes_lists_seeded_universes(client: TestClient) -> None:
    r = client.get("/universes")
    assert r.status_code == 200
    body = r.json()
    by_name = {u["name"]: u for u in body}
    assert "SP500" in by_name and "NIFTY100" in by_name
    assert by_name["SP500"]["n_members"] == 3
    assert by_name["NIFTY100"]["n_members"] == 1


def test_universes_members_default_today(client: TestClient) -> None:
    r = client.get("/universes/SP500/members")
    assert r.status_code == 200
    body = r.json()
    symbols = {m["symbol"] for m in body}
    assert symbols == {"AAA", "BBB", "CCC"}
    company_by_symbol = {m["symbol"]: m["company_name"] for m in body}
    assert company_by_symbol["AAA"] == "Alpha Co"


# ---------------------------------------------------------------------------
# /predictions/top
# ---------------------------------------------------------------------------


def test_top_long_picks_sorted_descending(client: TestClient) -> None:
    r = client.get("/predictions/top?universe=SP500&direction=long&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["universe"] == "SP500"
    assert body["direction"] == "long"
    returns = [p["predicted_return_5d"] for p in body["picks"]]
    assert returns == sorted(returns, reverse=True)
    # Top long should be AAA (return 0.05).
    assert body["picks"][0]["symbol"] == "AAA"
    assert body["picks"][0]["company_name"] == "Alpha Co"


def test_top_short_picks_sorted_ascending(client: TestClient) -> None:
    r = client.get("/predictions/top?universe=SP500&direction=short&limit=10")
    assert r.status_code == 200
    body = r.json()
    returns = [p["predicted_return_5d"] for p in body["picks"]]
    assert returns == sorted(returns)
    assert body["picks"][0]["symbol"] == "CCC"


def test_top_picks_invalid_direction_rejected(client: TestClient) -> None:
    r = client.get("/predictions/top?universe=SP500&direction=sideways")
    assert r.status_code == 422  # FastAPI pattern validation


def test_top_picks_invalid_limit_rejected(client: TestClient) -> None:
    r = client.get("/predictions/top?universe=SP500&direction=long&limit=0")
    assert r.status_code == 422
    r = client.get("/predictions/top?universe=SP500&direction=long&limit=10000")
    assert r.status_code == 422


def test_top_picks_no_data_returns_empty_list_not_500(client: TestClient) -> None:
    r = client.get("/predictions/top?universe=NIFTY100&direction=long")
    assert r.status_code == 200
    body = r.json()
    assert body["picks"] == []


# ---------------------------------------------------------------------------
# /predictions/{universe}/{symbol}
# ---------------------------------------------------------------------------


def test_stock_history_returns_data(client: TestClient) -> None:
    r = client.get("/predictions/SP500/AAA")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "AAA"
    assert len(body["history"]) >= 1
    pt = body["history"][0]
    assert pt["realized_return_5d"] is not None
    assert pt["realized_quintile"] is not None


def test_stock_history_404_for_unknown_symbol(client: TestClient) -> None:
    r = client.get("/predictions/SP500/UNKNOWN_TICKER")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /stocks/{symbol}/ohlcv
# ---------------------------------------------------------------------------


def test_stock_ohlcv_returns_bars(client: TestClient) -> None:
    today = date.today()
    start = today - timedelta(days=15)
    r = client.get(
        f"/stocks/AAA/ohlcv?start={start.isoformat()}&end={today.isoformat()}"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "AAA"
    assert len(body["bars"]) > 0
    bar = body["bars"][0]
    assert {"bar_date", "open", "high", "low", "close", "volume"}.issubset(bar)


# ---------------------------------------------------------------------------
# /performance/{universe}
# ---------------------------------------------------------------------------


def test_performance_returns_valid_schema(client: TestClient) -> None:
    r = client.get("/performance/SP500?lookback_days=30")
    assert r.status_code == 200
    body = r.json()
    assert body["universe"] == "SP500"
    assert body["lookback_days"] == 30
    # All seeded predictions are settled, so n_settled should equal n_predictions.
    assert body["n_predictions"] == body["n_settled"] == 3
    # The lists are well-formed (may be empty if too few rows for stats).
    assert isinstance(body["calibration"], list)
    assert isinstance(body["ic_timeseries"], list)


def test_performance_empty_for_unseen_universe(client: TestClient) -> None:
    r = client.get("/performance/NIFTY100?lookback_days=30")
    assert r.status_code == 200
    body = r.json()
    assert body["n_predictions"] == 0
    assert body["calibration"] == []
    assert body["ic_timeseries"] == []
