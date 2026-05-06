"""Forward-return + classification label generation, plus master dataset assembly."""

from packages.labels.dataset import build_training_dataset
from packages.labels.targets import compute_forward_log_return, compute_quintile_labels

__all__ = [
    "compute_forward_log_return",
    "compute_quintile_labels",
    "build_training_dataset",
]
