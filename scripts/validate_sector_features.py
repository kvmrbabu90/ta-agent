"""Phase E validation: GICS sector one-hot features.

Compares baseline (current 118 features) vs +sector (baseline + 12 one-hot
sector indicators) on 3 random seeds × 5-fold purged walk-forward CV.

Decision gate (matches validate_phase_a.py / the project's reject-by-default
convention):
    SHIP  iff  mean rank-IC delta > 0
         AND  worst per-seed rank-IC delta > -0.005
         AND  mean decile spread does not sign-flip negative
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from packages.common.logging import log
from packages.inference.predict import load_models_for_universe
from packages.labels.dataset import build_training_dataset
from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_with_cv

_SECTOR_PREFIX = "sector__"
_NON_FEAT = {"symbol", "bar_date", "in_universe", "fwd_return_5d", "fwd_quintile_5d"}


def _build_cfg(seed: int, hp: dict) -> TrainConfig:
    return TrainConfig(
        objective="regression",
        learning_rate=hp["learning_rate"],
        num_leaves=hp["num_leaves"],
        feature_fraction=hp["feature_fraction"],
        bagging_fraction=hp["bagging_fraction"],
        min_data_in_leaf=hp["min_data_in_leaf"],
        lambda_l1=hp["lambda_l1"],
        lambda_l2=hp["lambda_l2"],
        seed=seed,
    )


def main() -> int:
    # 5-year training window — same as monthly_retrain default.
    end = date(2026, 4, 30)  # leave 5 days of forward labels intact
    start = date(2021, 5, 1)

    log.info(f"validate_sector_features: building dataset {start} → {end}")
    df = build_training_dataset("SP500", start, end, horizon_days=5)
    if df.empty:
        log.error("training dataset empty")
        return 1

    all_feats = [c for c in df.columns if c not in _NON_FEAT]
    sector_feats = [c for c in all_feats if c.startswith(_SECTOR_PREFIX)]
    baseline_feats = [c for c in all_feats if not c.startswith(_SECTOR_PREFIX)]

    print(f"\nrows total:    {len(df):,}")
    print(f"all features:  {len(all_feats)}")
    print(f"baseline:      {len(baseline_feats)}")
    print(f"+sector:       {len(sector_feats)} -> {sector_feats[:6]}{'...' if len(sector_feats)>6 else ''}")

    if not sector_feats:
        log.error("no sector__ features found in panel — sector extension not active?")
        return 1

    # Reuse the production model's tuned hyperparameters so we measure
    # FEATURE lift, not hyperparameter lift.
    models = load_models_for_universe("SP500")
    hp = models.reg_meta.get("config", {})

    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=504,
    )

    configs = {
        "baseline": baseline_feats,
        "+sector":  baseline_feats + sector_feats,
    }
    seeds = [42, 43, 44]

    results: dict[str, dict[int, dict[str, float]]] = {}
    for label, feats in configs.items():
        results[label] = {}
        for seed in seeds:
            print(f"\n=== {label} | seed {seed} | {len(feats)} feats ===")
            out = train_with_cv(df, feats, "fwd_return_5d", splitter, _build_cfg(seed, hp))
            m = out["mean_metrics"]
            print(
                f"  rank_ic={m['mean_daily_rank_ic']:+.5f}  "
                f"ic={m['mean_daily_ic']:+.5f}  "
                f"decile_sp={m['mean_decile_spread']:+.5f}  "
                f"hit={m['hit_rate']:+.4f}"
            )
            results[label][seed] = m

    print("\n=== AGGREGATE (mean of 3 seeds) ===")
    keys = ["mean_daily_ic", "mean_daily_rank_ic", "mean_decile_spread", "hit_rate"]
    print(f"{'label':12s} | " + " | ".join(f"{k:>22s}" for k in keys))
    means: dict[str, dict[str, float]] = {}
    for label in configs:
        means[label] = {}
        row_parts = []
        for k in keys:
            vals = [results[label][s][k] for s in seeds]
            mean = float(sum(vals) / len(vals))
            means[label][k] = mean
            row_parts.append(f"{mean:>+22.5f}")
        print(f"{label:12s} | " + " | ".join(row_parts))

    print("\n=== LIFT vs baseline (mean of 3 seeds) ===")
    b = means["baseline"]
    m = means["+sector"]
    parts = []
    for k in keys:
        delta = m[k] - b[k]
        pct = (delta / abs(b[k]) * 100) if b[k] else float("nan")
        parts.append(f"{k}={delta:+.5f}({pct:+.0f}%)")
    print("  +sector  " + "  ".join(parts))

    print("\n=== PER-SEED REGRESSION CHECK (rank_ic) ===")
    print(f"{'label':12s} | {'seed42':>10s} {'seed43':>10s} {'seed44':>10s} | {'min_seed':>10s}")
    for label in configs:
        per_seed = {s: results[label][s]["mean_daily_rank_ic"] for s in seeds}
        worst = min(per_seed.values())
        print(
            f"{label:12s} | "
            + " ".join(f"{per_seed[s]:>+10.5f}" for s in seeds)
            + f" | {worst:>+10.5f}"
        )

    # === DECISION ===
    print("\n=== SHIP/SKIP DECISION ===")
    mean_lift = means["+sector"]["mean_daily_rank_ic"] - means["baseline"]["mean_daily_rank_ic"]
    worst_per_seed_delta = min(
        results["+sector"][s]["mean_daily_rank_ic"]
        - results["baseline"][s]["mean_daily_rank_ic"]
        for s in seeds
    )
    spread_delta = means["+sector"]["mean_decile_spread"] - means["baseline"]["mean_decile_spread"]
    spread_sign_flip = (
        means["baseline"]["mean_decile_spread"] > 0
        and means["+sector"]["mean_decile_spread"] < 0
    )
    ship = (
        mean_lift > 0
        and worst_per_seed_delta > -0.005
        and not spread_sign_flip
    )
    verdict = "SHIP" if ship else "SKIP"
    print(f"  mean_lift              = {mean_lift:+.5f}  (must be > 0)")
    print(f"  worst_per_seed_delta   = {worst_per_seed_delta:+.5f}  (must be > -0.005)")
    print(f"  decile_spread_delta    = {spread_delta:+.5f}  (sign_flip={spread_sign_flip})")
    print(f"  --> +sector  {verdict}")

    # Save report.
    report = {
        "configs": {k: len(v) for k, v in configs.items()},
        "seeds": seeds,
        "results": results,
        "lift": {k: m[k] - b[k] for k in keys},
        "verdict": verdict,
        "mean_lift_rank_ic": mean_lift,
        "worst_per_seed_delta": worst_per_seed_delta,
        "spread_delta": spread_delta,
        "spread_sign_flipped": spread_sign_flip,
    }
    out_path = Path("data/processed/sector_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport: {out_path}")

    nan_count = sum(
        1 for label in configs for s in seeds for k in keys
        if not math.isfinite(results[label][s].get(k, 0.0))
    )
    if nan_count:
        print(f"\nWARNING: {nan_count} non-finite metric values across all configs/seeds")
    return 0 if ship else 2  # 2 = explicit SKIP, distinguishable from crash


if __name__ == "__main__":
    sys.exit(main())
