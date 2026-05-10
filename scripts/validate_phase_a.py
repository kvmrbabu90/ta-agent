"""Phase A validation: controlled CV across A2 (VIX term structure) +
A3 (sector residuals), confirming on seeds 42/43/44.

Compares:
    baseline: existing 113 features
    +A2: baseline + 5 new VIX term-structure macro features
    +A3: baseline + 5 sector-residualized features
    +A2+A3: baseline + both new groups

Decision rule (per phase):
    Ship if mean rank-IC across 3 seeds > baseline AND no seed regresses
    by more than -0.005 rank-IC.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_with_cv

# Phase A2: 5 new macro features
A2_FEATURES = {
    "macro__vix9d_to_vix_ratio",
    "macro__vix_to_vix3m_ratio",
    "macro__vvix_z_252",
    "macro__skew_level",
    "macro__skew_chg_5d",
}

# Phase A3: 5 sector residual features
A3_FEATURES_PREFIX = "sector_residual__"


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


def main() -> None:
    df = pd.read_parquet("data/processed/training_sp500_2010-01-01_2026-05-06.parquet")
    non_feat = {"symbol", "bar_date", "in_universe", "fwd_return_5d", "fwd_quintile_5d"}
    all_feats = [c for c in df.columns if c not in non_feat]

    a2_in = [c for c in all_feats if c in A2_FEATURES]
    a3_in = [c for c in all_feats if c.startswith(A3_FEATURES_PREFIX)]
    baseline = [c for c in all_feats if c not in A2_FEATURES and not c.startswith(A3_FEATURES_PREFIX)]

    print(f"all features:    {len(all_feats)}")
    print(f"baseline:        {len(baseline)}")
    print(f"+A2 (vix-term):  {len(a2_in)}  -> {a2_in}")
    print(f"+A3 (residuals): {len(a3_in)}  -> {a3_in}")

    with open("data/models/SP500_regression_20260508_213151/metadata.json") as fh:
        prev = json.loads(fh.read())
    hp = prev["config"]

    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=504
    )

    configs = {
        "baseline": baseline,
        "+A2":      baseline + a2_in,
        "+A3":      baseline + a3_in,
        "+A2+A3":   baseline + a2_in + a3_in,
    }

    seeds = [42, 43, 44]
    results: dict[str, dict[int, dict[str, float]]] = {}
    for label, feats in configs.items():
        results[label] = {}
        for seed in seeds:
            print(f"\n=== {label} | seed {seed} | {len(feats)} feats ===")
            out = train_with_cv(df, feats, "fwd_return_5d", splitter, _build_cfg(seed, hp))
            m = out["mean_metrics"]
            print(f"  rank_ic={m['mean_daily_rank_ic']:+.5f}  ic={m['mean_daily_ic']:+.5f}  "
                  f"decile_sp={m['mean_decile_spread']:+.5f}  hit={m['hit_rate']:+.4f}")
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
    for label in ("+A2", "+A3", "+A2+A3"):
        m = means[label]
        parts = []
        for k in keys:
            delta = m[k] - b[k]
            pct = (delta / abs(b[k]) * 100) if b[k] else float("nan")
            parts.append(f"{k}={delta:+.5f}({pct:+.0f}%)")
        print(f"  {label:12s}  " + "  ".join(parts))

    print("\n=== PER-SEED REGRESSION CHECK (rank_ic) ===")
    print(f"{'label':12s} | {'seed42':>10s} {'seed43':>10s} {'seed44':>10s} | {'min_seed':>10s}")
    for label in configs:
        per_seed = {s: results[label][s]["mean_daily_rank_ic"] for s in seeds}
        worst = min(per_seed.values())
        print(f"{label:12s} | " + " ".join(f"{per_seed[s]:>+10.5f}" for s in seeds) + f" | {worst:>+10.5f}")

    # Decision: ship if mean lift > 0 AND no seed worse than baseline by > 0.005
    print("\n=== SHIP/SKIP DECISION ===")
    for label in ("+A2", "+A3", "+A2+A3"):
        mean_lift = means[label]["mean_daily_rank_ic"] - means["baseline"]["mean_daily_rank_ic"]
        worst_per_seed_delta = min(
            results[label][s]["mean_daily_rank_ic"] - results["baseline"][s]["mean_daily_rank_ic"]
            for s in seeds
        )
        verdict = "SHIP" if (mean_lift > 0 and worst_per_seed_delta > -0.005) else "SKIP"
        print(f"  {label:12s}  mean_lift={mean_lift:+.5f}  worst_per_seed_delta={worst_per_seed_delta:+.5f}  -> {verdict}")

    # Also flag interactions if any non-finite metric showed up.
    nan_count = sum(
        1 for label in configs for s in seeds
        for k in keys if not math.isfinite(results[label][s].get(k, 0.0))
    )
    if nan_count:
        print(f"\nWARNING: {nan_count} non-finite metric values across all configs/seeds")


if __name__ == "__main__":
    main()
