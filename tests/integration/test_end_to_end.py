"""Full-pipeline smoke test: synthetic OHLCV → features → labels → train →
predict → log → settle.

This is the safety net for refactors. Slow (~30s); deselected by default.
Run with::

    pytest -m integration tests/integration/test_end_to_end.py -v
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.common.config import settings
from packages.inference.db import init_predictions_db
from packages.inference.predict import _UniverseModels, predict_with_features
from packages.inference.tracking import log_predictions, settle_predictions
from packages.ingestion.storage import get_conn, upsert_membership, upsert_ohlcv
from packages.labels.dataset import build_training_dataset
from packages.modeling.registry import ModelMetadata, save_model
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import (
    TrainConfig,
    train_final_model,
    train_with_cv,
)

_SP500_BASE_DATE = date(2022, 1, 3)
_N_SYMBOLS = 12
_N_DAYS = 600


def _seed_synthetic_universe(duck_path: str) -> None:
    members: list[dict] = []
    bars: list[dict] = []
    for i in range(_N_SYMBOLS):
        sym = f"SYN{i:02d}"
        members.append(
            {
                "universe": "SP500",
                "symbol": sym,
                "exchange": "NASDAQ",
                "start_date": _SP500_BASE_DATE,
                "end_date": None,
                "company_name": f"Synthetic {i}",
            }
        )
        rng = np.random.default_rng(i)
        # Drift varies slightly per symbol so cross-sectional ranks have signal.
        drift = rng.normal(0.0003, 0.0002)
        ret = rng.normal(drift, 0.012, size=_N_DAYS)
        close = (100.0 + 5.0 * i) * np.exp(np.cumsum(ret))
        open_ = close * (1.0 + rng.normal(0.0, 0.003, size=_N_DAYS))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=_N_DAYS)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=_N_DAYS)))
        volume = rng.integers(500_000, 5_000_000, size=_N_DAYS)
        for j in range(_N_DAYS):
            bars.append(
                {
                    "symbol": sym, "exchange": "NASDAQ",
                    "bar_date": _SP500_BASE_DATE + timedelta(days=j),
                    "open": float(open_[j]), "high": float(high[j]),
                    "low": float(low[j]), "close": float(close[j]),
                    "volume": int(volume[j]),
                    "close_unadj": None,
                    "source": "test",
                    "ingested_at": datetime(2024, 1, 1),
                }
            )
    with get_conn(duck_path) as conn:
        upsert_membership(pd.DataFrame(members), conn=conn)
        upsert_ohlcv(pd.DataFrame(bars), conn=conn)


@pytest.mark.integration
def test_end_to_end_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    duck_path = str(tmp_path / "market.duckdb")
    sqlite_path = str(tmp_path / "predictions.sqlite")
    monkeypatch.setattr(settings, "duckdb_path", duck_path)
    monkeypatch.setattr(settings, "predictions_sqlite_path", sqlite_path)

    # 1. Synthetic OHLCV + membership.
    _seed_synthetic_universe(duck_path)

    # 2. Build the training dataset (features + labels + in_universe).
    end_d = _SP500_BASE_DATE + timedelta(days=_N_DAYS - 1)
    df = build_training_dataset(
        "SP500",
        _SP500_BASE_DATE,
        end_d,
        horizon_days=5,
        n_classes=4,  # only 12 symbols/day — quintiles need >= 5 distinct
        duckdb_path=duck_path,
        output_path=tmp_path / "training_sp500.parquet",
    )
    assert not df.empty
    feature_cols = [
        c for c in df.columns
        if c not in ("symbol", "bar_date", "in_universe", "fwd_return_5d", "fwd_quintile_5d")
    ]
    assert feature_cols, "no feature columns produced"

    # 3. Train regression model with a tiny CV config so the test stays fast.
    splitter = PurgedWalkForwardSplit(
        n_folds=2, horizon_days=5, embargo_days=5, min_train_size_days=200
    )
    cfg = TrainConfig(
        objective="regression",
        num_boost_round=40,
        early_stopping_rounds=10,
        min_data_in_leaf=20,
        num_leaves=15,
    )
    cv = train_with_cv(df, feature_cols, "fwd_return_5d", splitter, cfg)
    assert len(cv["models"]) == 2

    # 4. Final model + save via the registry (mock ModelMetadata).
    final_train_end = end_d - timedelta(days=10)
    booster, _ = train_final_model(df, feature_cols, "fwd_return_5d", cfg, final_train_end)
    metadata = ModelMetadata(
        universe="SP500",
        target="regression",
        horizon_days=5,
        feature_cols=feature_cols,
        train_start=_SP500_BASE_DATE,
        train_end=final_train_end,
        config=cfg.__dict__,
        cv_metrics=cv["mean_metrics"],
    )
    save_model(booster, metadata, base_dir=tmp_path / "models")

    # 5. Predict for the most recent panel date using the trained booster.
    #    We bypass predict_universe to avoid the big build_inference_features
    #    rebuild — the dataset already has features through end_d.
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
    inference_date = sorted(df["bar_date"].unique())[-15]  # leave runway for settling
    inference_rows = df[
        (df["bar_date"] == inference_date)
        & (df["in_universe"].fillna(False).astype(bool))
    ].copy()
    assert not inference_rows.empty

    # Construct a minimal cls model that just predicts argmax over a synthetic
    # softmax of feature_0 — gives us non-trivial top/bottom quintile probas.
    rng = np.random.default_rng(0)
    n_classes = 4
    fake_proba = rng.dirichlet([1.0] * n_classes, size=len(inference_rows))

    class _FakeClsBooster:
        def predict(self, X):  # noqa: N803, ARG002
            return fake_proba

    fake_models = _UniverseModels(
        universe="SP500",
        horizon_days=5,
        n_classes=n_classes,
        reg_model=booster,
        reg_meta={"feature_cols": feature_cols, "horizon_days": 5},
        reg_version="SP500_regression_test",
        cls_model=_FakeClsBooster(),
        cls_meta={"feature_cols": feature_cols, "config": {"n_classes": n_classes}},
        cls_version="SP500_classification_test",
    )

    inference_features = inference_rows[["symbol", "bar_date", *feature_cols]].copy()
    preds = predict_with_features(inference_features, fake_models)
    assert not preds.empty
    assert {"predicted_return", "predicted_quintile", "top_quintile_proba"}.issubset(preds.columns)

    # 6. Log predictions.
    init_predictions_db(sqlite_path)
    n_logged = log_predictions(preds, sqlite_path=sqlite_path)
    assert n_logged == len(preds)

    # Re-logging is idempotent.
    n_logged_again = log_predictions(preds, sqlite_path=sqlite_path)
    assert n_logged_again == len(preds)
    import sqlite3

    with sqlite3.connect(sqlite_path) as conn:
        n_rows = conn.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0]
    assert n_rows == len(preds)  # no duplicates

    # 7. Settle: realized closes are 5 trading bars after inference_date and
    #    the synthetic OHLCV has them.
    settle_date = inference_date + timedelta(days=10)
    n_settled = settle_predictions(
        as_of=settle_date,
        horizon_days=5,
        sqlite_path=sqlite_path,
        duckdb_path=duck_path,
    )
    assert n_settled == len(preds)

    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.execute(
            "SELECT realized_return, realized_quintile FROM predictions_log "
            "WHERE realized_return IS NOT NULL"
        )
        settled_rows = cur.fetchall()
    assert len(settled_rows) == len(preds)
    realized_returns = [r[0] for r in settled_rows]
    # Sanity: synthetic returns aren't all zero and aren't extreme.
    assert any(abs(r) > 1e-6 for r in realized_returns)
    assert all(abs(r) < 0.5 for r in realized_returns), "implausible realized return"
