"""File-based model registry. One directory per saved model:

    {base_dir}/{universe}_{target}_{YYYYMMDD_HHMMSS}/
        model.txt              (LightGBM booster)
        calibrators.pkl        (only for calibrated classification models)
        feature_importance.csv (sorted by gain, descending)
        metadata.json          (universe, target, horizon, feature_cols,
                                config, cv_metrics, created_at, ...)

Models without metadata.json are not loadable. The metadata is the
source of truth for which features the model expects, in what order.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import pandas as pd

from packages.common.config import MODELS_DIR
from packages.modeling.calibrate import CalibratedModel


@dataclass
class ModelMetadata:
    universe: str
    target: Literal["regression", "classification"]
    horizon_days: int
    feature_cols: list[str]
    train_start: date
    train_end: date
    config: dict[str, Any]
    cv_metrics: dict[str, Any]
    git_sha: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["train_start"] = self.train_start.isoformat()
        d["train_end"] = self.train_end.isoformat()
        d["created_at"] = self.created_at.isoformat()
        return d


def _detect_git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
        )
        return sha.decode().strip()
    except Exception:
        return None


def _save_feature_importance(
    booster: lgb.Booster, out_path: Path
) -> None:
    importances = booster.feature_importance(importance_type="gain")
    feat_names = booster.feature_name()
    pd.DataFrame({"feature": feat_names, "gain": importances}).sort_values(
        "gain", ascending=False
    ).to_csv(out_path, index=False)


def save_model(
    model: lgb.Booster | CalibratedModel,
    metadata: ModelMetadata,
    base_dir: Path | str = MODELS_DIR,
) -> Path:
    base = Path(base_dir)
    timestamp = metadata.created_at.strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{metadata.universe}_{metadata.target}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, CalibratedModel):
        model.save(out_dir)
        _save_feature_importance(model.booster, out_dir / "feature_importance.csv")
    elif isinstance(model, lgb.Booster):
        model.save_model(str(out_dir / "model.txt"))
        _save_feature_importance(model, out_dir / "feature_importance.csv")
    else:
        raise TypeError(f"unsupported model type: {type(model)!r}")

    if metadata.git_sha is None:
        metadata.git_sha = _detect_git_sha()

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    return out_dir


def find_latest_model_dir(
    universe: str,
    target: str,
    base_dir: Path | str = MODELS_DIR,
) -> Path:
    """Return the directory of the most recently saved model for (universe, target)."""
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"models base dir does not exist: {base}")
    candidates = sorted(
        d
        for d in base.iterdir()
        if d.is_dir()
        and d.name.startswith(f"{universe}_{target}_")
        and (d / "metadata.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no saved model for universe={universe} target={target} in {base}"
        )
    return candidates[-1]


def load_latest_model(
    universe: str,
    target: str,
    base_dir: Path | str = MODELS_DIR,
) -> tuple[lgb.Booster | CalibratedModel, dict[str, Any]]:
    latest = find_latest_model_dir(universe, target, base_dir)

    with open(latest / "metadata.json") as f:
        metadata = json.load(f)

    if (latest / "calibrators.pkl").exists():
        model: lgb.Booster | CalibratedModel = CalibratedModel.load(latest)
    else:
        model = lgb.Booster(model_file=str(latest / "model.txt"))

    return model, metadata
