"""Phase D2 validation: TTM aggregation + cross-sectional ranks of fundamentals.

Compares:
    baseline: 118 features (current shipped: 113 + 5 A2)
    +D2:      baseline + 10 sec_fundamentals_v2 features (5 raw TTM + 5 xs-rank)

3-seed cross-validation (42, 43, 44). Strict ship gate as before:
    SHIP if mean rank-IC lift > 0 AND no seed regresses by more than
    -0.005 rank-IC AND mean decile spread doesn't sign-flip negative.
"""

from __future__ import annotations

import json

import pandas as pd

from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_with_cv

D2_PREFIX = "sec_fundamentals_v2__"


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

    d2_in = [c for c in all_feats if c.startswith(D2_PREFIX)]
    baseline = [c for c in all_feats if not c.startswith(D2_PREFIX)]
    print(f"all features: {len(all_feats)}")
    print(f"baseline:     {len(baseline)}")
    print(f"+D2:          {len(d2_in)}  -> {d2_in}")
    if not d2_in:
        print("FATAL: no sec_fundamentals_v2__* in panel; rebuild required.")
        return

    print("\nNaN coverage (D2 features only):")
    for c in d2_in:
        print(f"  {c}: NaN_frac={df[c].isna().mean():.3f}, valid_rows={df[c].notna().sum():,}")

    with open("data/models/SP500_regression_20260508_213151/metadata.json") as fh:
        prev = json.loads(fh.read())
    hp = prev["config"]

    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=504
    )

    configs = {"baseline": baseline, "+D2": baseline + d2_in}
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
    print(f"{'label':10s} | " + " | ".join(f"{k:>22s}" for k in keys))
    means: dict[str, dict[str, float]] = {}
    for label in configs:
        means[label] = {}
        row_parts = []
        for k in keys:
            vals = [results[label][s][k] for s in seeds]
            mean = float(sum(vals) / len(vals))
            means[label][k] = mean
            row_parts.append(f"{mean:>+22.5f}")
        print(f"{label:10s} | " + " | ".join(row_parts))

    print("\n=== LIFT vs baseline (mean of 3 seeds) ===")
    b = means["baseline"]
    m = means["+D2"]
    parts = []
    for k in keys:
        delta = m[k] - b[k]
        pct = (delta / abs(b[k]) * 100) if b[k] else float("nan")
        parts.append(f"{k}={delta:+.5f}({pct:+.0f}%)")
    print("  +D2  " + "  ".join(parts))

    print("\n=== PER-SEED RANK-IC ===")
    for label in configs:
        per_seed = {s: results[label][s]["mean_daily_rank_ic"] for s in seeds}
        worst = min(per_seed.values())
        print(f"{label:10s} | " + " ".join(f"seed {s}: {per_seed[s]:+.5f}" for s in seeds) + f" | min: {worst:+.5f}")

    print("\n=== SHIP/SKIP DECISION ===")
    mean_lift = means["+D2"]["mean_daily_rank_ic"] - means["baseline"]["mean_daily_rank_ic"]
    worst_per_seed_delta = min(
        results["+D2"][s]["mean_daily_rank_ic"] - results["baseline"][s]["mean_daily_rank_ic"]
        for s in seeds
    )
    decile_lift = means["+D2"]["mean_decile_spread"] - means["baseline"]["mean_decile_spread"]
    decile_ok = means["+D2"]["mean_decile_spread"] >= 0
    verdict = (
        "SHIP" if (mean_lift > 0 and worst_per_seed_delta > -0.005 and decile_ok)
        else "SKIP"
    )
    reasons = []
    if not (mean_lift > 0):
        reasons.append("(mean rank-IC regression)")
    if worst_per_seed_delta <= -0.005:
        reasons.append("(severe per-seed regression)")
    if not decile_ok:
        reasons.append("(decile spread sign-flipped negative)")
    print(f"  +D2  mean_lift={mean_lift:+.5f}  worst_per_seed={worst_per_seed_delta:+.5f}  "
          f"decile_lift={decile_lift:+.5f}  -> {verdict} {' '.join(reasons)}")


if __name__ == "__main__":
    main()
