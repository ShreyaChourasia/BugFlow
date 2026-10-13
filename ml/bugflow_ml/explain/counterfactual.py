"""US-35: counterfactual guidance — "what would lower the risk", searched
only over **actionable** features (size/churn, files touched, directories
and subsystems touched — all of which shrink together when a commit is
split into smaller ones), never over features a developer can't act on
(author history, whether it's a fix).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

# Splitting a commit shrinks its footprint roughly proportionally across
# these — reducing one without the others wouldn't correspond to any real
# action a developer could take.
ACTIONABLE_FEATURES = [
    "lines_added",
    "lines_deleted",
    "churn",
    "files_changed",
    "directories_touched",
    "subsystems_touched",
    "entropy",
]
REDUCTION_STEPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class CounterfactualResult:
    found: bool
    original_probability: float
    new_probability: float | None
    reduction_fraction: float | None
    text: str


def _scaled_features(
    base_features: dict[str, float], reduction_fraction: float
) -> dict[str, float]:
    scaled = dict(base_features)
    for feature in ACTIONABLE_FEATURES:
        if feature in scaled:
            scaled[feature] = scaled[feature] * (1 - reduction_fraction)
    return scaled


def _predict(model: Any, feature_columns: list[str], features: dict[str, float]) -> float:
    X = np.array([[float(features.get(col, 0.0)) for col in feature_columns]])
    return float(model.predict_proba(X)[0, 1])


def search_counterfactual(
    calibrated_model: Any,
    base_features: dict[str, float],
    feature_columns: list[str],
    threshold: float,
) -> CounterfactualResult:
    """Smallest proportional reduction of the actionable features that
    drops calibrated risk below `threshold`. "No actionable suggestion,
    request extra review" (US-35) when even a 90% reduction isn't enough —
    an honest admission the model has nothing useful to suggest, not a
    forced answer."""
    original_probability = _predict(calibrated_model, feature_columns, base_features)

    for reduction_fraction in REDUCTION_STEPS:
        candidate = _scaled_features(base_features, reduction_fraction)
        new_probability = _predict(calibrated_model, feature_columns, candidate)
        if new_probability < threshold:
            return CounterfactualResult(
                found=True,
                original_probability=original_probability,
                new_probability=new_probability,
                reduction_fraction=reduction_fraction,
                text=(
                    f"Splitting this into smaller commits (cutting its size by "
                    f"roughly {reduction_fraction:.0%}) would lower risk from "
                    f"{original_probability:.0%} to {new_probability:.0%}."
                ),
            )

    return CounterfactualResult(
        found=False,
        original_probability=original_probability,
        new_probability=None,
        reduction_fraction=None,
        text=(
            "No actionable suggestion would reliably lower this below the "
            "threshold — request extra review."
        ),
    )
