"""Phase B validation: re-test interaction features on the +A2 panel.

Compares:
    baseline: 118 features (current shipped: 113 + 5 A2)
    +interactions: baseline + 8 per-symbol x macro products

Interactions were previously disabled because the result on the
113-feature panel ranged from +122% (seed 42) to -156% (seed 44)
rank-IC across seeds. Phase B retests on the new 118-feature panel
that includes A2 (VIX term-structure macro) — these are inputs to the
interaction features and may stabilize the result.
"""

from __future__ import annotations

import json

import pandas as pd

from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_with_cv

INTERACTIONS_PREFIX = "interactions__"


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

    int_in = [c for c in all_feats if c.startswith(INTERACTIONS_PREFIX)]
    baseline = [c for c in all_feats if not c.startswith(INTERACTIONS_PREFIX)]
    print(f"all features:    {len(all_feats)}")
    print(f"baseline:        {len(baseline)}")
    print(f"+interactions:   {len(int_in)}  -> {int_in}")

    if not int_in:
        print("FATAL: no interactions__* columns in panel; rebuild required.")
        return

    with open("data/models/SP500_regression_20260508_213151/metadata.json") as fh:
        prev = json.loads(fh.read())
    hp = prev["config"]

    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=504
    )

    configs = {"baseline": baseline, "+interactions": baseline + int_in}
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
    print(f"{'label':14s} | " + " | ".join(f"{k:>22s}" for k in keys))
    means: dict[str, dict[str, float]] = {}
    for label in configs:
        means[label] = {}
        row_parts = []
        for k in keys:
            vals = [results[label][s][k] for s in seeds]
            mean = float(sum(vals) / len(vals))
            means[label][k] = mean
            row_parts.append(f"{mean:>+22.5f}")
        print(f"{label:14s} | " + " | ".join(row_parts))

    print("\n=== LIFT vs baseline (mean of 3 seeds) ===")
    b = means["baseline"]
    m = means["+interactions"]
    parts = []
    for k in keys:
        delta = m[k] - b[k]
        pct = (delta / abs(b[k]) * 100) if b[k] else float("nan")
        parts.append(f"{k}={delta:+.5f}({pct:+.0f}%)")
    print("  +interactions  " + "  ".join(parts))

    print("\n=== PER-SEED RANK-IC ===")
    for label in configs:
        per_seed = {s: results[label][s]["mean_daily_rank_ic"] for s in seeds}
        worst = min(per_seed.values())
        print(f"{label:14s} | " + " ".join(f"seed {s}: {per_seed[s]:+.5f}" for s in seeds) + f" | min: {worst:+.5f}")

    print("\n=== SHIP/SKIP DECISION ===")
    mean_lift = means["+interactions"]["mean_daily_rank_ic"] - means["baseline"]["mean_daily_rank_ic"]
    worst_per_seed_delta = min(
        results["+interactions"][s]["mean_daily_rank_ic"] - results["baseline"][s]["mean_daily_rank_ic"]
        for s in seeds
    )
    decile_lift = means["+interactions"]["mean_decile_spread"] - means["baseline"]["mean_decile_spread"]
    verdict = (
        "SHIP" if (mean_lift > 0 and worst_per_seed_delta > -0.005 and decile_lift >= 0)
        else "SKIP"
    )
    print(f"  mean_lift={mean_lift:+.5f}  worst_per_seed_delta={worst_per_seed_delta:+.5f}  "
          f"decile_spread_lift={decile_lift:+.5f}  -> {verdict}")


if __name__ == "__main__":
    main()
