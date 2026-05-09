"""4-way controlled CV validation for earnings + interactions features.

Runs the same hyperparams (the prior-best from SP500_regression_20260508_213151)
across four feature configs and prints lift vs baseline. Confirms winners on
seed 43 before recommending.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from packages.modeling.splits import PurgedWalkForwardSplit
from packages.modeling.train import TrainConfig, train_with_cv


def main() -> None:
    df = pd.read_parquet(
        "data/processed/training_sp500_2010-01-01_2026-05-06.parquet"
    )
    non_feat = {"symbol", "bar_date", "in_universe", "fwd_return_5d", "fwd_quintile_5d"}
    all_feats = [c for c in df.columns if c not in non_feat]
    earn_feats = [c for c in all_feats if c.startswith("earnings__")]
    int_feats = [c for c in all_feats if c.startswith("interactions__")]
    print(
        f"all features: {len(all_feats)} | "
        f"earnings: {len(earn_feats)} | interactions: {len(int_feats)}"
    )

    with open("data/models/SP500_regression_20260508_213151/metadata.json") as fh:
        prev = json.loads(fh.read())
    hp = prev["config"]

    def cfg(seed: int = 42) -> TrainConfig:
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

    splitter = PurgedWalkForwardSplit(
        n_folds=5, horizon_days=5, embargo_days=5, min_train_size_days=504
    )

    baseline_feats = [
        c for c in all_feats if c not in earn_feats and c not in int_feats
    ]
    print(f"baseline_feats: {len(baseline_feats)}")

    configs = {
        "baseline":               baseline_feats,
        "+earnings":              baseline_feats + earn_feats,
        "+interactions":          baseline_feats + int_feats,
        "+earnings+interactions": baseline_feats + earn_feats + int_feats,
    }

    results: dict[str, dict[str, float]] = {}
    for label, feats in configs.items():
        print(f"\n=== {label} (seed 42, {len(feats)} features) ===")
        out = train_with_cv(df, feats, "fwd_return_5d", splitter, cfg(42))
        m = out["mean_metrics"]
        print(f"  rank_ic={m['mean_daily_rank_ic']:+.5f}  ic={m['mean_daily_ic']:+.5f}  "
              f"decile_sp={m['mean_decile_spread']:+.5f}  hit={m['hit_rate']:+.4f}")
        results[label] = m

    # Confirm best variant on seed 43.
    print("\n=== seed 43 confirmation: baseline + best variant ===")
    out_b43 = train_with_cv(df, configs["baseline"], "fwd_return_5d", splitter, cfg(43))
    m43_b = out_b43["mean_metrics"]
    print(
        f"  baseline (seed 43): rank_ic={m43_b['mean_daily_rank_ic']:+.5f}  "
        f"ic={m43_b['mean_daily_ic']:+.5f}  decile_sp={m43_b['mean_decile_spread']:+.5f}  "
        f"hit={m43_b['hit_rate']:+.4f}"
    )

    best_label = max(
        (k for k in configs if k != "baseline"),
        key=lambda k: results[k].get("mean_daily_rank_ic", -math.inf),
    )
    out_best43 = train_with_cv(
        df, configs[best_label], "fwd_return_5d", splitter, cfg(43)
    )
    m43_best = out_best43["mean_metrics"]
    print(
        f"  {best_label} (seed 43): rank_ic={m43_best['mean_daily_rank_ic']:+.5f}  "
        f"ic={m43_best['mean_daily_ic']:+.5f}  decile_sp={m43_best['mean_decile_spread']:+.5f}  "
        f"hit={m43_best['hit_rate']:+.4f}"
    )

    print("\n=== SUMMARY (seed 42) ===")
    keys = ["mean_daily_ic", "mean_daily_rank_ic", "mean_decile_spread", "hit_rate"]
    print(f"{'label':30s}  " + "  ".join(f"{k:>22s}" for k in keys))
    for label, m in results.items():
        row = "  ".join(f"{m.get(k, float('nan')):>+22.5f}" for k in keys)
        print(f"{label:30s}  {row}")

    print("\n=== LIFT vs baseline (seed 42) ===")
    b = results["baseline"]
    for label, m in results.items():
        if label == "baseline":
            continue
        parts = []
        for k in keys:
            a = b.get(k, float("nan"))
            v = m.get(k, float("nan"))
            delta = v - a
            pct = (delta / abs(a) * 100) if a else float("nan")
            parts.append(f"{k}={delta:+.5f}({pct:+.0f}%)")
        print(f"  {label:25s}  " + "  ".join(parts))

    print(f"\n=== SEED 43 COMPARISON ({best_label}) ===")
    keys_short = ["mean_daily_ic", "mean_daily_rank_ic", "mean_decile_spread", "hit_rate"]
    for k in keys_short:
        b = m43_b.get(k, float("nan"))
        v = m43_best.get(k, float("nan"))
        delta = v - b
        pct = (delta / abs(b) * 100) if b else float("nan")
        print(f"  {k:25s}  base={b:+.5f}  best={v:+.5f}  delta={delta:+.5f} ({pct:+.0f}%)")


if __name__ == "__main__":
    main()
