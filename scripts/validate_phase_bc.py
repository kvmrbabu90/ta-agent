"""Phase B + C joint validation: 4-way controlled CV across +A2 panel.

Compares:
    baseline:       113 + 5 A2 = 118 features
    +B:             baseline + 8 interactions
    +C:             baseline + 6 sec_events
    +B+C:           baseline + both (132 features)

Each config × 3 seeds (42, 43, 44). Apply the strict ship gate:
    SHIP if mean lift > 0 across the 3 seeds AND no single seed
    regresses by more than 0.005 rank-IC, AND decile spread doesn't
    sign-flip negative.

Lets us decide B and C independently — and also surfaces destructive
interference between B and C if the +B+C combination is much worse than
either solo.
"""

from __future__ import annotations

import json

import pandas as pd

from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_with_cv

INTERACTIONS_PREFIX = "interactions__"
SEC_EVENTS_PREFIX = "sec_events__"


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

    b_in = [c for c in all_feats if c.startswith(INTERACTIONS_PREFIX)]
    c_in = [c for c in all_feats if c.startswith(SEC_EVENTS_PREFIX)]
    baseline = [
        c for c in all_feats
        if not c.startswith(INTERACTIONS_PREFIX) and not c.startswith(SEC_EVENTS_PREFIX)
    ]
    print(f"all features: {len(all_feats)}")
    print(f"baseline:     {len(baseline)}")
    print(f"+B (inter):   {len(b_in)}  -> {b_in}")
    print(f"+C (sec_ev):  {len(c_in)}  -> {c_in}")
    if not b_in:
        print("WARNING: no interactions__* in panel (Phase B can't validate)")
    if not c_in:
        print("WARNING: no sec_events__* in panel (Phase C can't validate)")

    with open("data/models/SP500_regression_20260508_213151/metadata.json") as fh:
        prev = json.loads(fh.read())
    hp = prev["config"]

    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=504
    )

    configs = {
        "baseline": baseline,
        "+B":       baseline + b_in,
        "+C":       baseline + c_in,
        "+B+C":     baseline + b_in + c_in,
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
    for label in ("+B", "+C", "+B+C"):
        m = means[label]
        parts = []
        for k in keys:
            delta = m[k] - b[k]
            pct = (delta / abs(b[k]) * 100) if b[k] else float("nan")
            parts.append(f"{k}={delta:+.5f}({pct:+.0f}%)")
        print(f"  {label:6s}  " + "  ".join(parts))

    print("\n=== PER-SEED RANK-IC ===")
    for label in configs:
        per_seed = {s: results[label][s]["mean_daily_rank_ic"] for s in seeds}
        worst = min(per_seed.values())
        print(f"{label:10s} | " + " ".join(f"seed {s}: {per_seed[s]:+.5f}" for s in seeds) + f" | min: {worst:+.5f}")

    print("\n=== SHIP/SKIP DECISION ===")
    base = means["baseline"]
    for label in ("+B", "+C", "+B+C"):
        mean_lift = means[label]["mean_daily_rank_ic"] - base["mean_daily_rank_ic"]
        worst_per_seed_delta = min(
            results[label][s]["mean_daily_rank_ic"] - results["baseline"][s]["mean_daily_rank_ic"]
            for s in seeds
        )
        decile_lift = means[label]["mean_decile_spread"] - base["mean_decile_spread"]
        decile_ok = means[label]["mean_decile_spread"] >= 0  # decile spread should be non-negative
        verdict = (
            "SHIP" if (mean_lift > 0 and worst_per_seed_delta > -0.005 and decile_ok)
            else "SKIP"
        )
        reason = ""
        if not (mean_lift > 0):
            reason += " (mean rank-IC regression)"
        if worst_per_seed_delta <= -0.005:
            reason += " (severe per-seed regression)"
        if not decile_ok:
            reason += " (decile spread sign-flipped negative)"
        print(f"  {label:6s}  mean_lift={mean_lift:+.5f}  worst_per_seed={worst_per_seed_delta:+.5f}  "
              f"decile_lift={decile_lift:+.5f}  -> {verdict}{reason}")


if __name__ == "__main__":
    main()
