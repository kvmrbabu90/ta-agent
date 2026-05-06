"""Daily inference: predict, rank, explain, log, settle, summarize."""

from packages.inference.db import get_sqlite_conn, init_predictions_db
from packages.inference.explain import explain_predictions
from packages.inference.predict import (
    build_inference_features,
    load_models_for_universe,
    predict_universe,
)
from packages.inference.ranker import rank_predictions
from packages.inference.tracking import (
    log_predictions,
    performance_summary,
    settle_predictions,
)

__all__ = [
    "init_predictions_db",
    "get_sqlite_conn",
    "build_inference_features",
    "load_models_for_universe",
    "predict_universe",
    "rank_predictions",
    "explain_predictions",
    "log_predictions",
    "settle_predictions",
    "performance_summary",
]
