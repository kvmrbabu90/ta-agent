"""Unit tests for log_predictions / settle_predictions / performance_summary."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from packages.common.config import settings
from packages.inference.db import get_sqlite_conn
from packages.inference.tracking import (
    log_predictions,
    performance_summary,
    settle_predictions,
)
from packages.ingestion.storage import get_conn as get_duck_conn
from packages.ingestion.storage import upsert_ohlcv


def _prediction_row(
    symbol: str = "AAA",
    as_of: date = date(2024, 1, 5),
    universe: str = "SP500",
    horizon_days: int = 5,
    predicted_return: float = 0.01,
    predicted_quintile: int = 4,
    top_proba: float = 0.55,
    bot_proba: float = 0.05,
    reg_v: str = "v1",
    cls_v: str = "v1",
) -> dict:
    return {
        "universe": universe,
        "symbol": symbol,
        "as_of": as_of,
        "horizon_days": horizon_days,
        "predicted_return": predicted_return,
        "predicted_quintile": predicted_quintile,
        "top_quintile_proba": top_proba,
        "bottom_quintile_proba": bot_proba,
        "model_version_regression": reg_v,
        "model_version_classification": cls_v,
    }


# ---------------------------------------------------------------------------
# log_predictions idempotency
# ---------------------------------------------------------------------------


def test_log_predictions_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_path = str(tmp_path / "predictions.sqlite")
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)

    df = pd.DataFrame([_prediction_row()])
    log_predictions(df)
    log_predictions(df)
    log_predictions(df)

    with get_sqlite_conn() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM predictions_log WHERE symbol = ?", ["AAA"]
        ).fetchone()
    assert rows[0] == 1


def test_log_predictions_updates_predicted_columns_on_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = str(tmp_path / "predictions.sqlite")
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)

    log_predictions(pd.DataFrame([_prediction_row(predicted_return=0.01, reg_v="v1")]))
    log_predictions(pd.DataFrame([_prediction_row(predicted_return=0.05, reg_v="v2")]))

    with get_sqlite_conn() as conn:
        row = conn.execute(
            "SELECT predicted_return, model_version_regression FROM predictions_log WHERE symbol = ?",
            ["AAA"],
        ).fetchone()
    assert row[0] == pytest.approx(0.05)
    assert row[1] == "v2"


def test_log_predictions_preserves_realized_columns_on_re_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = str(tmp_path / "predictions.sqlite")
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)

    log_predictions(pd.DataFrame([_prediction_row()]))

    # Manually fill in realized fields as if settled.
    with get_sqlite_conn() as conn:
        conn.execute(
            "UPDATE predictions_log SET realized_return = ?, realized_quintile = ?, settled_at = ? WHERE symbol = ?",
            [0.02, 3, datetime(2024, 1, 12), "AAA"],
        )
        conn.commit()

    log_predictions(pd.DataFrame([_prediction_row(predicted_return=0.99)]))

    with get_sqlite_conn() as conn:
        row = conn.execute(
            "SELECT predicted_return, realized_return, realized_quintile FROM predictions_log WHERE symbol = ?",
            ["AAA"],
        ).fetchone()
    assert row[0] == pytest.approx(0.99)  # prediction overwritten
    assert row[1] == pytest.approx(0.02)  # realized preserved
    assert row[2] == 3                    # quintile preserved


# ---------------------------------------------------------------------------
# settle_predictions
# ---------------------------------------------------------------------------


def _seed_ohlcv(duckdb_path: str, symbol: str, prices: list[tuple[date, float]]) -> None:
    rows = [
        {
            "symbol": symbol, "exchange": "NASDAQ",
            "bar_date": d, "open": p, "high": p * 1.01,
            "low": p * 0.99, "close": p, "volume": 1_000_000,
            "close_unadj": None, "source": "test",
            "ingested_at": datetime(2024, 1, 1),
        }
        for d, p in prices
    ]
    with get_duck_conn(duckdb_path) as conn:
        upsert_ohlcv(pd.DataFrame(rows), conn=conn)


def test_settle_predictions_fills_realized_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = str(tmp_path / "predictions.sqlite")
    duckdb_path = str(tmp_path / "market.duckdb")
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)
    monkeypatch.setattr(settings, "duckdb_path", duckdb_path)

    pred_date = date(2024, 1, 8)  # Mon
    # 6 bars of trading data (need 5 future bars after pred_date).
    bars = [
        (date(2024, 1, 8), 100.0),
        (date(2024, 1, 9), 101.0),
        (date(2024, 1, 10), 102.0),
        (date(2024, 1, 11), 103.0),
        (date(2024, 1, 12), 104.0),
        (date(2024, 1, 16), 110.0),  # 5th bar after pred_date — Mon 1/15 was MLK day
    ]
    _seed_ohlcv(duckdb_path, "AAA", bars)
    log_predictions(pd.DataFrame([_prediction_row(symbol="AAA", as_of=pred_date)]))

    n = settle_predictions(as_of=date(2024, 1, 16), horizon_days=5)
    assert n == 1

    import math
    expected = math.log(110.0 / 100.0)
    with get_sqlite_conn() as conn:
        row = conn.execute(
            "SELECT realized_return, settled_at FROM predictions_log WHERE symbol = ?",
            ["AAA"],
        ).fetchone()
    assert row[0] == pytest.approx(expected, rel=1e-6)
    assert row[1] is not None


def test_settle_predictions_skips_when_horizon_not_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = str(tmp_path / "predictions.sqlite")
    duckdb_path = str(tmp_path / "market.duckdb")
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)
    monkeypatch.setattr(settings, "duckdb_path", duckdb_path)

    pred_date = date(2024, 1, 8)
    # Only 3 future bars — not enough to settle at horizon=5.
    _seed_ohlcv(
        duckdb_path,
        "BBB",
        [
            (date(2024, 1, 8), 100.0),
            (date(2024, 1, 9), 101.0),
            (date(2024, 1, 10), 102.0),
        ],
    )
    log_predictions(pd.DataFrame([_prediction_row(symbol="BBB", as_of=pred_date)]))

    # Even though we pass a future settle date, ohlcv has insufficient bars.
    n = settle_predictions(as_of=date(2024, 1, 30), horizon_days=5)
    assert n == 0
    with get_sqlite_conn() as conn:
        row = conn.execute(
            "SELECT realized_return FROM predictions_log WHERE symbol = ?", ["BBB"]
        ).fetchone()
    assert row[0] is None


def test_settle_assigns_realized_quintiles_cross_sectionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = str(tmp_path / "predictions.sqlite")
    duckdb_path = str(tmp_path / "market.duckdb")
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)
    monkeypatch.setattr(settings, "duckdb_path", duckdb_path)

    pred_date = date(2024, 1, 8)
    realized_close_date = date(2024, 1, 15)
    pred_rows = []
    for i, sym in enumerate(["A", "B", "C", "D", "E"]):
        # Different terminal prices -> different realized returns.
        bars = [
            (pred_date, 100.0),
            (date(2024, 1, 9), 100.0),
            (date(2024, 1, 10), 100.0),
            (date(2024, 1, 11), 100.0),
            (date(2024, 1, 12), 100.0),
            (realized_close_date, 100.0 + (i + 1) * 1.0),
        ]
        _seed_ohlcv(duckdb_path, sym, bars)
        pred_rows.append(_prediction_row(symbol=sym, as_of=pred_date))
    log_predictions(pd.DataFrame(pred_rows))

    settle_predictions(as_of=realized_close_date + timedelta(days=1), horizon_days=5)

    with get_sqlite_conn() as conn:
        df = pd.read_sql_query(
            "SELECT symbol, realized_return, realized_quintile FROM predictions_log WHERE as_of = ? ORDER BY realized_return",
            conn,
            params=[pred_date.isoformat()],
        )
    # All 5 should have a realized_quintile assigned spanning {0..4}.
    quints = sorted(df["realized_quintile"].dropna().unique().tolist())
    assert quints == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# performance_summary
# ---------------------------------------------------------------------------


def test_performance_summary_reports_n_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = str(tmp_path / "predictions.sqlite")
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)

    today = date.today()
    rows = []
    for i, sym in enumerate(["A", "B", "C", "D", "E", "F"]):
        rows.append(
            _prediction_row(
                symbol=sym,
                as_of=today - timedelta(days=10),
                predicted_return=0.01 * (i - 3),
            )
        )
    log_predictions(pd.DataFrame(rows))

    # Manually populate realized_return + realized_quintile.
    with get_sqlite_conn() as conn:
        for i, sym in enumerate(["A", "B", "C", "D", "E", "F"]):
            conn.execute(
                "UPDATE predictions_log SET realized_return = ?, realized_quintile = ?, settled_at = ? WHERE symbol = ?",
                [0.005 * (i - 3), i % 5, datetime.utcnow(), sym],
            )
        conn.commit()

    summary = performance_summary("SP500", lookback_days=30)
    assert summary["n_settled"] == 6
    assert summary["universe"] == "SP500"
