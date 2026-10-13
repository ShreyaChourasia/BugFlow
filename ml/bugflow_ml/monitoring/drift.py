"""US-41: PSI (population stability index) drift detection — compares a
recent window of feature values against the distribution the champion was
actually trained on, per feature, and flags any feature that's drifted
past a configurable threshold.
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_PSI_THRESHOLD = 0.2
N_BINS = 10


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    severity: str  # "none" | "moderate" | "severe"


def _severity_for(psi: float, threshold: float) -> str:
    if psi >= threshold * 2:
        return "severe"
    if psi >= threshold:
        return "moderate"
    return "none"


def population_stability_index(
    reference: list[float], current: list[float], n_bins: int = N_BINS
) -> float:
    """§8: PSI per feature. Bin edges come from the *reference* (training)
    distribution's quantiles, so each reference bin starts with ~equal
    population by construction — only `current`'s bin populations can
    actually shift. Degenerate inputs (too few points, a constant
    reference) return 0.0 rather than raising or dividing by zero."""
    reference_arr = np.asarray(reference, dtype=float)
    current_arr = np.asarray(current, dtype=float)
    if len(reference_arr) < n_bins or len(current_arr) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(reference_arr, quantiles))
    if len(edges) < 3:  # reference is ~constant — no meaningful bins
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference_arr, bins=edges)
    cur_counts, _ = np.histogram(current_arr, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    # Floor both at a small epsilon so an empty bin never produces log(0)
    # or a division by zero — a bin truly empty in both contributes ~0 PSI.
    epsilon = 1e-4
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_feature_drift(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
    threshold: float = DEFAULT_PSI_THRESHOLD,
) -> list[FeatureDrift]:
    """One PSI per feature present in both `reference` and `current`,
    sorted worst-first."""
    results = []
    for feature, ref_values in reference.items():
        cur_values = current.get(feature)
        if not cur_values:
            continue
        psi = population_stability_index(ref_values, cur_values)
        results.append(
            FeatureDrift(feature=feature, psi=psi, severity=_severity_for(psi, threshold))
        )
    results.sort(key=lambda d: -d.psi)
    return results
