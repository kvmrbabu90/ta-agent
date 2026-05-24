"""Honest walk-forward backtest.

For each retrain date in a monthly schedule, train the model on data
ending strictly BEFORE that date (with a 5-day embargo), then predict
forward for every trading day until the next retrain. Write predictions
to a separate SQLite so they don't collide with the look-ahead-biased
`predictions_log` already in place.

This is the single most important test we can run: it answers whether
the +145% backtest number is real or an artifact of using today's model
on historical dates.

Compute budget: ~24 monthly retrains × 1-3 min per train (no Optuna) +
~21 trading days/month × <1s per predict = roughly 60-90 minutes total.
We DELIBERATELY skip Optuna re-tuning — we reuse the production tuned
hyperparameters from the currently-deployed model. Re-tuning at each
date would 10× the compute without changing the conclusion (we're
measuring model + features, not the search).

Output:
  data/processed/walkforward/predictions.sqlite   — honest predictions
  data/processed/walkforward/paper.sqlite         — backtest run
  data/processed/walkforward/report.json          — comparison metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal

from packages.common.config import settings
from packages.common.logging import log
from packages.inference.db import init_predictions_db
from packages.inference.predict import (
    build_inference_features,
    load_models_for_universe,
    predict_with_features,
)
from packages.inference.tracking import log_predictions
from packages.labels.dataset import build_training_dataset
from packages.modeling.calibrate import calibrate_classifier
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_final_model
from packages.modeling.tune import tune_hyperparameters

_DEFAULT_HORIZON = 5
_EMBARGO_DAYS = 5  # match horizon — labels needing future >5d are excluded
_DEFAULT_LOOKBACK_YEARS = 5
_TRAIN_LOOKBACK_DAYS = 365 * _DEFAULT_LOOKBACK_YEARS

# Honest walk-forward outputs.
_WF_DIR = Path("data/processed/walkforward")


# Per-universe trading calendar. Indian indices follow NSE which has
# different holidays from NYSE (Diwali, Holi, Independence Day, etc.) —
# using the wrong calendar would miss real trading days and add fake ones.
_UNIVERSE_CALENDAR = {
    "SP500": "NYSE",
    "NIFTY100": "XNSE",
    "NIFTY50": "XNSE",
}


def _calendar_for(universe: str) -> str:
    return _UNIVERSE_CALENDAR.get(universe, "NYSE")


def _trading_days(start: date, end: date, *, calendar: str = "NYSE") -> list[date]:
    cal = mcal.get_calendar(calendar)
    sched = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    if sched.empty:
        return []
    return [d.date() for d in sched.index.to_pydatetime()]


def _retrain_dates(start: date, end: date, *, calendar: str = "NYSE") -> list[date]:
    """First trading day of each calendar month between start and end."""
    days = _trading_days(start, end, calendar=calendar)
    seen_months: set[tuple[int, int]] = set()
    out: list[date] = []
    for d in days:
        key = (d.year, d.month)
        if key in seen_months:
            continue
        seen_months.add(key)
        out.append(d)
    return out


def _load_production_configs(universe: str) -> tuple[TrainConfig, TrainConfig]:
    """Pull the tuned hyperparameters from the currently-deployed model.

    These are the result of Optuna search done at production-train time;
    we reuse them at every walk-forward retrain rather than re-tuning,
    on the grounds that the walk-forward test is about the model+features
    architecture, not the search.
    """
    models = load_models_for_universe(universe)
    reg_cfg = _config_from_meta(models.reg_meta.get("config", {}), "regression")
    cls_cfg = _config_from_meta(models.cls_meta.get("config", {}), "classification")
    return reg_cfg, cls_cfg


def _load_configs_for(universe: str, *, hyperparams_from: str | None) -> tuple[TrainConfig, TrainConfig]:
    """Load hyperparameter configs, optionally borrowing them from a different
    universe. Useful when target-universe production models were trained on
    suspect data (e.g. NIFTY100 May-7 models trained pre-HAL-fix) — borrow
    SP500's tuned params as a sane proxy."""
    return _load_production_configs(hyperparams_from or universe)


def _config_from_meta(meta_cfg: dict, target: str) -> TrainConfig:
    """Coerce a saved-model config dict back into a TrainConfig."""
    base = TrainConfig(objective=target)
    fields = {k: v for k, v in meta_cfg.items() if k in asdict(base)}
    fields["objective"] = target
    return TrainConfig(**fields)


def _retry_on_lock(fn, *, attempts: int = 6, delay_s: float = 5.0):
    """Retry a callable on DuckDB IO/lock errors with linear backoff.

    Walk-forward runs for >1h alongside a live API + scheduler. Any one
    of those can briefly hold market.duckdb in a way that blocks our
    open. A single retry-with-backoff loop is enough to survive the
    typical 1-5 second contention window.
    """
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Cannot open file" not in msg and "IO Error" not in msg:
                raise  # not a lock error, surface it
            last_exc = exc
            log.warning(
                f"_retry_on_lock: attempt {i+1}/{attempts} failed ({msg[:120]}); "
                f"sleeping {delay_s:.1f}s"
            )
            time.sleep(delay_s)
    raise last_exc  # pragma: no cover


def _maybe_tune_at_retrain(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    base_cfg: TrainConfig,
    *,
    enabled: bool,
    n_trials: int,
    n_folds: int,
    horizon_days: int,
    embargo_days: int,
    optuna_n_jobs: int,
    study_name: str,
    cv_min_train_days: int = 504,
) -> TrainConfig:
    """If per-retrain Optuna is enabled, run a tune restricted to ``df``
    (which only contains data up to the retrain's train_end). The returned
    config is used to fit the final model for THIS retrain only.

    The TPE sampler is reseeded per retrain so optimization is deterministic
    given the data; otherwise the legacy behavior of borrowing production
    hyperparameters is preserved.
    """
    if not enabled:
        return base_cfg

    splitter = PurgedWalkForwardSplit(
        n_folds=n_folds,
        horizon_days=horizon_days,
        embargo_days=embargo_days,
        min_train_size_days=cv_min_train_days,
    )
    best_cfg, _study = tune_hyperparameters(
        df, feature_cols, label_col, splitter, base_cfg,
        n_trials=n_trials,
        timeout_seconds=None,  # let n_trials drive — don't truncate
        study_name=study_name,
        constraints=None,  # constraint thresholds vary by dataset size
        seeds=None,
        n_jobs=optuna_n_jobs,
    )
    return best_cfg


def _train_models_through(
    universe: str,
    final_train_end: date,
    *,
    horizon_days: int,
    reg_cfg: TrainConfig,
    cls_cfg: TrainConfig,
    per_retrain_optuna: bool = False,
    optuna_trials: int = 20,
    optuna_n_jobs: int = 1,
    cv_min_train_days: int = 504,
) -> tuple[Any, Any, list[str], TrainConfig, TrainConfig]:
    """Train regression + (calibrated) classification models on data
    through `final_train_end`.

    When ``per_retrain_optuna=True``, hyperparameters are re-tuned via
    Optuna on the training slice BEFORE the final model is fit — this
    eliminates the hyperparameter-look-ahead bias that comes from
    borrowing production-model params (which were tuned on the full
    2010-2026 dataset). The cost is a 5-30× slowdown per retrain depending
    on dataset size and trial count.

    Returns (reg_model, cls_model, feature_cols, reg_cfg_used, cls_cfg_used).
    """
    train_start = final_train_end - timedelta(days=_TRAIN_LOOKBACK_DAYS)
    df = build_training_dataset(
        universe, train_start, final_train_end, horizon_days=horizon_days,
    )
    if df.empty:
        raise RuntimeError(
            f"training dataset empty for {universe} [{train_start} .. {final_train_end}]"
        )
    non_feature = {
        "symbol", "bar_date", "in_universe",
        f"fwd_return_{horizon_days}d", f"fwd_quintile_{horizon_days}d",
    }
    feature_cols = [c for c in df.columns if c not in non_feature]

    # Per-retrain Optuna tune (look-ahead-free): restricts the search to
    # data already in `df`, which is bounded by `final_train_end`.
    reg_cfg_used = _maybe_tune_at_retrain(
        df, feature_cols, f"fwd_return_{horizon_days}d", reg_cfg,
        enabled=per_retrain_optuna,
        n_trials=optuna_trials,
        n_folds=5,
        horizon_days=horizon_days,
        embargo_days=_EMBARGO_DAYS,
        optuna_n_jobs=optuna_n_jobs,
        study_name=f"wf_{universe}_reg_{final_train_end.isoformat()}",
        cv_min_train_days=cv_min_train_days,
    )
    cls_cfg_used = _maybe_tune_at_retrain(
        df, feature_cols, f"fwd_quintile_{horizon_days}d", cls_cfg,
        enabled=per_retrain_optuna,
        n_trials=optuna_trials,
        n_folds=5,
        horizon_days=horizon_days,
        embargo_days=_EMBARGO_DAYS,
        optuna_n_jobs=optuna_n_jobs,
        study_name=f"wf_{universe}_cls_{final_train_end.isoformat()}",
        cv_min_train_days=cv_min_train_days,
    )

    reg_booster, _ = train_final_model(
        df, feature_cols,
        f"fwd_return_{horizon_days}d", reg_cfg_used, final_train_end,
    )
    cls_booster, _ = train_final_model(
        df, feature_cols,
        f"fwd_quintile_{horizon_days}d", cls_cfg_used, final_train_end,
    )
    # Calibrate classifier on the most recent labeled slice — matches the
    # monthly_retrain.py logic so probabilities are comparable.
    df_local = df.copy()
    df_local["bar_date"] = pd.to_datetime(df_local["bar_date"]).dt.date
    es_cutoff = final_train_end - timedelta(days=60)
    cal_start = es_cutoff - timedelta(days=30)
    cal_mask = (
        (df_local["bar_date"] > cal_start)
        & (df_local["bar_date"] <= es_cutoff)
        & (df_local["in_universe"].fillna(False).astype(bool))
        & (df_local[f"fwd_quintile_{horizon_days}d"].notna())
    )
    if cal_mask.sum() > 0:
        X_cal = df_local.loc[cal_mask, feature_cols]
        y_cal = df_local.loc[cal_mask, f"fwd_quintile_{horizon_days}d"].astype(int)
        cls_model = calibrate_classifier(cls_booster, X_cal, y_cal)
    else:
        cls_model = cls_booster
    return reg_booster, cls_model, feature_cols, reg_cfg_used, cls_cfg_used


def _predict_window(
    universe: str,
    days: list[date],
    feature_cols: list[str],
    reg_model: Any,
    cls_model: Any,
    horizon_days: int,
    *,
    reg_version: str,
    cls_version: str,
    sqlite_path: str,
) -> int:
    """Predict for each trading day in `days`. Returns # rows logged.

    Performance: builds the feature panel ONCE for [days[0] - 400d, days[-1]]
    instead of once per day. The original `build_inference_features` is
    convenient for daily live use but recomputes the rolling 400-day
    feature panel for every call (~50s/day), which would push a 24-month
    walk-forward to 7+ hours. Batched, it's ~5 minutes per retrain window.
    """
    from packages.features.pipeline import build_feature_matrix
    from packages.inference.predict import (
        _UniverseModels,
        _align_features,
        predict_with_features as _pwf,
    )
    from packages.ingestion.universe.membership import members_on

    if not days:
        return 0
    bundle = _UniverseModels(
        universe=universe, horizon_days=horizon_days, n_classes=5,
        reg_model=reg_model,
        reg_meta={"feature_cols": feature_cols, "horizon_days": horizon_days},
        reg_version=reg_version,
        cls_model=cls_model,
        cls_meta={"feature_cols": feature_cols, "config": {"n_classes": 5}},
        cls_version=cls_version,
    )

    # ONE feature build for the whole window.
    panel_start = days[0] - timedelta(days=400)
    panel_end = days[-1]
    panel = build_feature_matrix(universe, panel_start, panel_end)
    if panel.empty:
        log.warning(f"predict_window: empty panel for {universe} [{panel_start}..{panel_end}]")
        return 0
    panel["bar_date"] = pd.to_datetime(panel["bar_date"]).dt.date

    def _members_on_with_retry(uni: str, dt, *, attempts: int = 30, delay_s: float = 5.0):
        """members_on opens its own DuckDB connection. If another WF
        process is holding the DB exclusively (training-dataset assembly
        phase), this read-open will fail with an IOException. Retry for
        up to attempts × delay_s = 2.5 min — long enough to outlast any
        single training-dataset write."""
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return members_on(uni, dt)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "Cannot open file" not in msg and "being used by another process" not in msg \
                        and "IO Error" not in msg:
                    raise  # not a lock collision — let the outer handler log it
                last_exc = exc
                log.warning(
                    f"predict {uni} {dt}: DuckDB locked by another process "
                    f"(attempt {i+1}/{attempts}); sleeping {delay_s:.0f}s"
                )
                time.sleep(delay_s)
        # Out of attempts — re-raise so the per-day handler logs it.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")

    rows_logged = 0
    for d in days:
        try:
            day_rows = panel[panel["bar_date"] == d]
            if day_rows.empty:
                continue
            members = _members_on_with_retry(universe, d)
            if members.empty:
                continue
            day_rows = day_rows[day_rows["symbol"].isin(members["symbol"])]
            if day_rows.empty:
                continue
            aligned = _align_features(day_rows, feature_cols, model_label="inference")
            inf_df = pd.DataFrame({
                "symbol": day_rows["symbol"].values,
                "bar_date": day_rows["bar_date"].values,
            })
            for c in feature_cols:
                inf_df[c] = aligned[c].values
            preds = _pwf(inf_df, bundle)
            if preds.empty:
                continue
            rows_logged += int(log_predictions(preds, sqlite_path=sqlite_path))
        except Exception as exc:  # noqa: BLE001
            log.warning(f"predict {universe} {d}: failed: {exc!r}")
    return rows_logged


# Hide the imports we no longer use to silence the linter.
_ = (build_inference_features, predict_with_features)


def run_walkforward(
    universe: str,
    start: date,
    end: date,
    *,
    horizon_days: int = _DEFAULT_HORIZON,
    out_dir: Path = _WF_DIR,
    hyperparams_from: str | None = None,
    calendar: str | None = None,
    per_retrain_optuna: bool = False,
    optuna_trials: int = 20,
    optuna_n_jobs: int = 1,
    device: str = "cpu",
    cv_min_train_days: int = 504,
) -> dict:
    """Top-level entry. Returns a summary dict; report.json is written too."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = str(out_dir / "predictions.sqlite")
    paper_path = str(out_dir / "paper.sqlite")
    report_path = out_dir / "report.json"

    cal = calendar or _calendar_for(universe)

    # Predictions DB — DO NOT UNLINK if it already exists. The previous
    # version of this script blindly unlinked at every startup, which
    # destroyed in-flight WF progress when the runner needed to be
    # restarted. init_predictions_db is idempotent (CREATE TABLE IF NOT
    # EXISTS), so calling it on an existing file is safe.
    #
    # If you genuinely want a fresh run, delete the output directory
    # manually before launching:
    #     rm -rf data/processed/walkforward_10yr_strict
    #
    # Pre-startup safety backup: before opening the DB for any writes,
    # snapshot the existing file with a timestamped name. Cheap insurance
    # — copies a 100 MB file in ~1 second.
    if os.path.exists(preds_path) and os.path.getsize(preds_path) > 50_000:
        import shutil as _shutil
        _ts = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(preds_path).parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        safety_path = str(backup_dir / f"predictions_pre_startup_{_ts}.sqlite")
        _shutil.copy2(preds_path, safety_path)
        log.info(
            f"walkforward: pre-startup safety backup → {safety_path} "
            f"({os.path.getsize(safety_path)/1e6:.1f} MB)"
        )
        # Prune old pre-startup safety files: keep last 5.
        startup_backups = sorted(backup_dir.glob("predictions_pre_startup_*.sqlite"))
        for old in startup_backups[:-5]:
            old.unlink(missing_ok=True)
    init_predictions_db(preds_path)

    log.info(
        f"walkforward: universe={universe} window={start}..{end} "
        f"calendar={cal} hyperparams_from={hyperparams_from or universe} "
        f"per_retrain_optuna={per_retrain_optuna} optuna_trials={optuna_trials} "
        f"device={device} out={out_dir}"
    )
    reg_cfg, cls_cfg = _load_configs_for(universe, hyperparams_from=hyperparams_from)
    # Apply GPU / threading override across both targets.
    from dataclasses import replace as _replace
    reg_cfg = _replace(reg_cfg, device=device)
    cls_cfg = _replace(cls_cfg, device=device)
    log.info(
        f"walkforward: reg_lr={reg_cfg.learning_rate:.4f} "
        f"reg_leaves={reg_cfg.num_leaves} cls_lr={cls_cfg.learning_rate:.4f} "
        f"device={reg_cfg.device}"
    )

    retrain_dates = _retrain_dates(start, end, calendar=cal)
    all_trading = _trading_days(start, end, calendar=cal)
    log.info(
        f"walkforward: {len(retrain_dates)} monthly retrains over "
        f"{len(all_trading)} trading days"
    )

    summary: dict = {
        "universe": universe, "start": start.isoformat(), "end": end.isoformat(),
        "horizon_days": horizon_days, "retrains": [], "errors": [],
        "predictions_path": preds_path, "paper_path": paper_path,
    }
    t_start_total = time.monotonic()

    # Iterate retrain windows. Each retrain is responsible for predicting
    # every trading day in [retrain_date, next_retrain_date).
    retrain_dates_padded = retrain_dates + [end + timedelta(days=1)]

    # Resume support: check which retrain windows already have predictions
    # in the output sqlite. Skip those entirely (no train, no predict).
    # Identifies a retrain as "done" when predictions exist for the LAST
    # trading day of its window — guarantees the retrain emitted a full
    # window, not a partial crash. Cheap: one count query at startup.
    already_done: set = set()
    try:
        import sqlite3 as _sqlite3
        if os.path.exists(preds_path):
            _con = _sqlite3.connect(preds_path)
            rows_existing = _con.execute(
                "SELECT DISTINCT as_of FROM predictions_log WHERE universe = ?",
                (universe,),
            ).fetchall()
            _con.close()
            already_done = {str(r[0]) for r in rows_existing}
            log.info(
                f"walkforward: resume — found {len(already_done)} existing "
                f"prediction days in {preds_path}"
            )
    except Exception as exc:  # noqa: BLE001
        log.warning(f"walkforward: resume check failed ({exc!r}); will run from scratch")
        already_done = set()

    for i, rd in enumerate(retrain_dates):
        window_end = retrain_dates_padded[i + 1] - timedelta(days=1)
        # Embargo: model can only see data ending `embargo` days before rd.
        # Since the label is a 5-day forward return, we cut the training
        # end at rd - embargo so we don't leak the labels at the boundary.
        train_end = rd - timedelta(days=_EMBARGO_DAYS + 1)
        days_in_window = [d for d in all_trading if rd <= d <= window_end]
        if not days_in_window:
            continue
        # Skip if the LAST day of this retrain's window already has
        # predictions — i.e. this retrain previously completed cleanly.
        if str(days_in_window[-1]) in already_done:
            log.info(
                f"walkforward: [{i+1}/{len(retrain_dates)}] {rd}: skipping — "
                f"predictions for last day {days_in_window[-1]} already exist"
            )
            continue
        t_retrain = time.monotonic()
        try:
            reg_model, cls_model, feature_cols, reg_cfg_used, cls_cfg_used = _retry_on_lock(
                lambda: _train_models_through(
                    universe, train_end,
                    horizon_days=horizon_days,
                    reg_cfg=reg_cfg, cls_cfg=cls_cfg,
                    per_retrain_optuna=per_retrain_optuna,
                    optuna_trials=optuna_trials,
                    optuna_n_jobs=optuna_n_jobs,
                    cv_min_train_days=cv_min_train_days,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.exception(f"walkforward: train {rd} failed: {exc!r}")
            summary["errors"].append({"retrain_date": rd.isoformat(), "phase": "train", "error": repr(exc)})
            continue
        train_seconds = time.monotonic() - t_retrain

        reg_version = f"wf_{universe}_regression_{rd.isoformat()}"
        cls_version = f"wf_{universe}_classification_{rd.isoformat()}"
        t_predict = time.monotonic()
        try:
            n_rows = _retry_on_lock(
                lambda: _predict_window(
                    universe, days_in_window, feature_cols,
                    reg_model, cls_model, horizon_days,
                    reg_version=reg_version, cls_version=cls_version,
                    sqlite_path=preds_path,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.exception(f"walkforward: predict {rd} failed: {exc!r}")
            summary["errors"].append({"retrain_date": rd.isoformat(), "phase": "predict", "error": repr(exc)})
            continue
        predict_seconds = time.monotonic() - t_predict
        log.info(
            f"  [{i+1}/{len(retrain_dates)}] {rd}: train_end={train_end} "
            f"window={days_in_window[0]}..{days_in_window[-1]} "
            f"({len(days_in_window)}d)  rows={n_rows}  "
            f"train={train_seconds:.1f}s predict={predict_seconds:.1f}s"
        )
        summary["retrains"].append({
            "retrain_date": rd.isoformat(),
            "train_end": train_end.isoformat(),
            "window_start": days_in_window[0].isoformat(),
            "window_end": days_in_window[-1].isoformat(),
            "n_trading_days": len(days_in_window),
            "n_predictions_logged": n_rows,
            "train_seconds": round(train_seconds, 1),
            "predict_seconds": round(predict_seconds, 1),
            "reg_hyperparams": {
                "learning_rate": reg_cfg_used.learning_rate,
                "num_leaves": reg_cfg_used.num_leaves,
                "min_data_in_leaf": reg_cfg_used.min_data_in_leaf,
                "feature_fraction": reg_cfg_used.feature_fraction,
                "bagging_fraction": reg_cfg_used.bagging_fraction,
                "lambda_l1": reg_cfg_used.lambda_l1,
                "lambda_l2": reg_cfg_used.lambda_l2,
            },
            "cls_hyperparams": {
                "learning_rate": cls_cfg_used.learning_rate,
                "num_leaves": cls_cfg_used.num_leaves,
                "min_data_in_leaf": cls_cfg_used.min_data_in_leaf,
                "feature_fraction": cls_cfg_used.feature_fraction,
                "bagging_fraction": cls_cfg_used.bagging_fraction,
                "lambda_l1": cls_cfg_used.lambda_l1,
                "lambda_l2": cls_cfg_used.lambda_l2,
            },
        })

    elapsed = time.monotonic() - t_start_total
    summary["elapsed_s"] = round(elapsed, 1)
    summary["total_predictions"] = _count_predictions(preds_path, universe)
    log.info(
        f"walkforward done: {len(summary['retrains'])} retrains, "
        f"{summary['total_predictions']} predictions, errors={len(summary['errors'])}, "
        f"elapsed={elapsed/60:.1f}min"
    )

    report_path.write_text(json.dumps(summary, indent=2))
    log.info(f"walkforward report: {report_path}")
    return summary


def _count_predictions(sqlite_path: str, universe: str) -> int:
    if not Path(sqlite_path).exists():
        return 0
    conn = sqlite3.connect(sqlite_path)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM predictions_log WHERE universe = ?",
                (universe,),
            ).fetchone()[0]
        )
    finally:
        conn.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="SP500")
    p.add_argument(
        "--start", type=date.fromisoformat, default=None,
        help="First date to backtest. Default: 24 months ago.",
    )
    p.add_argument(
        "--end", type=date.fromisoformat, default=None,
        help="Last date to backtest. Default: yesterday.",
    )
    p.add_argument("--horizon-days", type=int, default=_DEFAULT_HORIZON)
    p.add_argument("--out-dir", type=Path, default=_WF_DIR)
    p.add_argument(
        "--hyperparams-from", default=None,
        help="Universe whose production model hyperparams should be reused "
             "(e.g. SP500 to borrow tuned params for NIFTY100 WF). "
             "Defaults to --universe.",
    )
    p.add_argument(
        "--calendar", default=None,
        help="pandas_market_calendars name (NYSE, XNSE, etc.). "
             "Auto-selected from universe if not set.",
    )
    p.add_argument(
        "--per-retrain-optuna", action="store_true",
        help="Run Optuna at every retrain on data up to that retrain's "
             "train_end. ELIMINATES hyperparameter look-ahead bias (no "
             "params borrowed from a production model tuned on the full "
             "history). Much slower (~5-30x).",
    )
    p.add_argument(
        "--optuna-trials", type=int, default=20,
        help="Number of Optuna trials per retrain per target when "
             "--per-retrain-optuna is set. Default 20.",
    )
    p.add_argument(
        "--optuna-n-jobs", type=int, default=1,
        help="Parallel Optuna trials per retrain. >1 with GPU may contend "
             "for the device; with CPU LightGBM you can usually go 4-8.",
    )
    p.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="LightGBM device. 'gpu' requires a GPU-enabled LightGBM build.",
    )
    p.add_argument(
        "--cv-min-train-days", type=int, default=504,
        help="Minimum training-fold size for purged walk-forward CV. "
             "Default 504 (~2 years). Lower (e.g. 252) when the universe "
             "has shorter history (e.g. NIFTY100 only goes back to 2014).",
    )
    args = p.parse_args()
    end = args.end or (date.today() - timedelta(days=1))
    start = args.start or (end - timedelta(days=365 * 2))
    try:
        run_walkforward(
            args.universe, start, end,
            horizon_days=args.horizon_days, out_dir=args.out_dir,
            hyperparams_from=args.hyperparams_from,
            calendar=args.calendar,
            per_retrain_optuna=args.per_retrain_optuna,
            optuna_trials=args.optuna_trials,
            optuna_n_jobs=args.optuna_n_jobs,
            device=args.device,
            cv_min_train_days=args.cv_min_train_days,
        )
    except Exception:  # noqa: BLE001
        log.error(f"walkforward crashed: {traceback.format_exc()}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Silence unused-import linter for `settings`.
_ = settings
