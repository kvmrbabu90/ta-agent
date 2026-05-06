"""End-to-end smoke test for the FastAPI backend.

Boots the app via TestClient against the same seeded DBs the unit tests use
and walks every endpoint to confirm response shapes line up. Marked
integration so it's deselected by default; run with::

    pytest -m integration tests/integration/test_api_e2e.py
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from packages.common.config import settings
from packages.inference.db import init_predictions_db
from packages.inference.tracking import log_predictions
from packages.ingestion.storage import get_conn as duck_conn
from packages.ingestion.storage import upsert_membership, upsert_ohlcv
from services.api.main import app


def _seed(tmp_path: Path) -> tuple[str, str]:
    duck_path = str(tmp_path / "market.duckdb")
    sqlite_path = str(tmp_path / "predictions.sqlite")

    members = [
        {
            "universe": "SP500", "symbol": s, "exchange": "NASDAQ",
            "start_date": date(2010, 1, 1), "end_date": None,
            "company_name": f"{s} Co",
        }
        for s in ("AAA", "BBB", "CCC")
    ]
    today = date.today()
    bars = []
    for sym, base in [("AAA", 100.0), ("BBB", 200.0), ("CCC", 50.0)]:
        for i in range(15):
            bars.append(
                {
                    "symbol": sym, "exchange": "NASDAQ",
                    "bar_date": today - timedelta(days=15 - i),
                    "open": base + i * 0.5, "high": base + i * 0.5 + 1,
                    "low": base + i * 0.5 - 1, "close": base + i * 0.5 + 0.2,
                    "volume": 1_000_000, "close_unadj": None,
                    "source": "test", "ingested_at": datetime(2024, 1, 1),
                }
            )

    with duck_conn(duck_path) as conn:
        upsert_membership(pd.DataFrame(members), conn=conn)
        upsert_ohlcv(pd.DataFrame(bars), conn=conn)

    init_predictions_db(sqlite_path)
    pred_date = today - timedelta(days=10)
    rows = []
    for sym, ret in [("AAA", 0.04), ("BBB", 0.005), ("CCC", -0.03)]:
        rows.append(
            {
                "universe": "SP500", "symbol": sym, "as_of": pred_date,
                "horizon_days": 5, "predicted_return": ret,
                "predicted_quintile": 4 if ret > 0.02 else (0 if ret < 0 else 2),
                "top_quintile_proba": 0.7 if ret > 0.02 else 0.2,
                "bottom_quintile_proba": 0.6 if ret < 0 else 0.05,
                "model_version_regression": "regv1",
                "model_version_classification": "clsv1",
            }
        )
    log_predictions(pd.DataFrame(rows), sqlite_path=sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    for sym, ret, q in [("AAA", 0.05, 4), ("BBB", 0.01, 2), ("CCC", -0.04, 0)]:
        conn.execute(
            "UPDATE predictions_log SET realized_return=?, realized_quintile=?, settled_at=? WHERE symbol=?",
            [ret, q, datetime.utcnow(), sym],
        )
    conn.commit()
    conn.close()

    return duck_path, sqlite_path


@pytest.mark.integration
class TestApiE2E:
    def test_walk_every_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        duck_path, sqlite_path = _seed(tmp_path)
        monkeypatch.setattr(settings, "duckdb_path", duck_path)
        monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)

        with TestClient(app) as client:
            # /health
            assert client.get("/health").status_code == 200

            # /universes
            r = client.get("/universes")
            assert r.status_code == 200
            assert any(u["name"] == "SP500" for u in r.json())

            # /universes/{u}/members
            r = client.get("/universes/SP500/members")
            assert r.status_code == 200
            assert {m["symbol"] for m in r.json()} == {"AAA", "BBB", "CCC"}

            # /predictions/top long + short
            for direction in ("long", "short"):
                r = client.get(
                    f"/predictions/top?universe=SP500&direction={direction}&limit=10"
                )
                assert r.status_code == 200
                assert r.json()["direction"] == direction

            # /predictions/{u}/{sym}
            r = client.get("/predictions/SP500/AAA")
            assert r.status_code == 200

            # /stocks/{sym}/ohlcv
            r = client.get("/stocks/AAA/ohlcv")
            assert r.status_code == 200
            assert len(r.json()["bars"]) > 0

            # /performance/{u}
            r = client.get("/performance/SP500?lookback_days=30")
            assert r.status_code == 200
            body = r.json()
            assert body["n_predictions"] == 3 and body["n_settled"] == 3

            # /openapi.json
            r = client.get("/openapi.json")
            assert r.status_code == 200
            assert "paths" in r.json()
