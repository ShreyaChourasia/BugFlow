"""US-08 (model part): commit-risk training pipeline — chronological split
(C9), LightGBM + logistic-regression baseline with class weighting,
isotonic calibration, effort-aware metrics, and a per-tenure fairness report
(C10). No MLflow or database access here — see the backend's
`training_service` for that; this module is pure, deterministic and
independently testable.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

FEATURE_COLUMNS = [
    "lines_added",
    "lines_deleted",
    "churn",
    "files_changed",
    "directories_touched",
    "subsystems_touched",
    "entropy",
    "author_prior_commits",
    "is_fix",
]

RISK_LOW_MAX = 0.3
RISK_HIGH_MIN = 0.7
MIN_TRAINING_COMMITS = 20


@dataclass
class TrainingCommit:
    id: int
    sha: str
    timestamp: datetime
    features: dict[str, Any]
    is_bug_inducing: bool
    author_tenure_days: float | None = None


@dataclass
class TrainingResult:
    seed: int
    data_version: str
    train_ids: list[int]
    calib_ids: list[int]
    test_ids: list[int]
    raw_model: Any
    calibrated_model: Any
    baseline_model: Any
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    fairness: dict[str, Any]
    feature_columns: list[str]


def compute_data_version(commit_ids: list[int]) -> str:
    """A hash of the exact training snapshot — changes iff the set of
    commit ids used changes, regardless of order."""
    canonical = ",".join(str(i) for i in sorted(commit_ids))
    return hashlib.sha256(canonical.encode()).hexdigest()


def chronological_split(
    commits: list[TrainingCommit], train_frac: float = 0.7, calib_frac: float = 0.15
) -> tuple[list[TrainingCommit], list[TrainingCommit], list[TrainingCommit]]:
    """C9: train on the past, test on the future. Sorts by (timestamp, id) —
    id as a tiebreaker since git commit timestamps only have one-second
    resolution — then slices into contiguous, non-overlapping windows."""
    ordered = sorted(commits, key=lambda c: (c.timestamp, c.id))
    n = len(ordered)
    n_train = int(n * train_frac)
    n_calib = int(n * calib_frac)
    train = ordered[:n_train]
    calib = ordered[n_train : n_train + n_calib]
    test = ordered[n_train + n_calib :]

    # Leakage check (C9): every train timestamp must be no later than every
    # calibration/test timestamp, and likewise calib before test. Contiguous
    # slicing of a sorted list guarantees this, but the whole point of a
    # leakage test is to not just trust that — verify it holds.
    if train and calib:
        assert max(c.timestamp for c in train) <= min(c.timestamp for c in calib)
    if calib and test:
        assert max(c.timestamp for c in calib) <= min(c.timestamp for c in test)
    if train and test:
        assert max(c.timestamp for c in train) <= min(c.timestamp for c in test)

    return train, calib, test


def build_feature_matrix(commits: list[TrainingCommit]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[float(c.features.get(col, 0.0)) for col in FEATURE_COLUMNS] for c in commits],
        dtype=float,
    )
    y = np.array([1 if c.is_bug_inducing else 0 for c in commits], dtype=int)
    return X, y


def _train_models(
    X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray, seed: int
) -> tuple[Any, Any, Any]:
    import lightgbm as lgb
    from sklearn.frozen import FrozenEstimator

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    raw_model = lgb.LGBMClassifier(
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        verbosity=-1,
    )
    raw_model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(raw_model), method="isotonic")
    calibrated_model.fit(X_calib, y_calib)

    baseline_model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    baseline_model.fit(X_train, y_train)

    return raw_model, calibrated_model, baseline_model


def recall_at_k_percent_effort(
    y_true: np.ndarray, y_prob: np.ndarray, effort: np.ndarray, k: float = 0.2
) -> float:
    """Of all truly bug-inducing commits, what fraction are caught if a
    reviewer works through commits ranked by predicted risk until they've
    reviewed k% of the total churn (lines changed)? The effort-aware
    analogue of recall@k that JIT defect prediction literature uses instead
    of plain recall@k-commits."""
    total_positive = y_true.sum()
    total_effort = effort.sum()
    if total_positive == 0 or total_effort == 0:
        return 0.0

    order = np.argsort(-y_prob)
    sorted_true = y_true[order]
    sorted_effort = effort[order]
    cumulative_effort = np.cumsum(sorted_effort)
    cutoff = int(np.searchsorted(cumulative_effort, k * cumulative_effort[-1], side="left")) + 1
    cutoff = min(cutoff, len(sorted_true))
    captured = sorted_true[:cutoff].sum()
    return float(captured / total_positive)


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, churn_test: np.ndarray
) -> dict[str, float]:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics: dict[str, float] = {}
    if len(set(y_test.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
    metrics["brier_score"] = float(brier_score_loss(y_test, y_prob))
    metrics["recall_at_20pct_effort"] = recall_at_k_percent_effort(y_test, y_prob, churn_test, 0.2)
    return metrics


def fairness_report(
    y_prob: np.ndarray, tenure_days: list[float | None], cutoff_days: float = 180
) -> dict[str, Any]:
    """C10: risk disparity by contributor tenure, e.g. < 6 months vs >= 6
    months. Returns None fields when no tenure data is available rather
    than silently omitting the report — the caller always gets the same
    shape back."""
    known = [(p, t) for p, t in zip(y_prob, tenure_days, strict=True) if t is not None]
    if not known:
        return {
            "junior_mean_risk": None,
            "senior_mean_risk": None,
            "junior_count": 0,
            "senior_count": 0,
            "disparity": None,
        }

    junior = [p for p, t in known if t < cutoff_days]
    senior = [p for p, t in known if t >= cutoff_days]
    junior_mean = float(np.mean(junior)) if junior else None
    senior_mean = float(np.mean(senior)) if senior else None
    disparity = None
    if junior_mean is not None and senior_mean is not None:
        disparity = abs(junior_mean - senior_mean)

    return {
        "junior_mean_risk": junior_mean,
        "senior_mean_risk": senior_mean,
        "junior_count": len(junior),
        "senior_count": len(senior),
        "disparity": disparity,
    }


def risk_level_for(probability: float) -> str:
    if probability < RISK_LOW_MAX:
        return "low"
    if probability >= RISK_HIGH_MIN:
        return "high"
    return "medium"


def confidence_for(probability: float) -> float:
    """How far the prediction is from a coin flip, normalised to [0, 1]."""
    return abs(probability - 0.5) * 2


def train_commit_risk(
    commits: list[TrainingCommit], seed: int = 42, train_frac: float = 0.7, calib_frac: float = 0.15
) -> TrainingResult:
    if len(commits) < MIN_TRAINING_COMMITS:
        raise ValueError(
            f"Need at least {MIN_TRAINING_COMMITS} commits to train "
            f"(train/calibration/test all need data); got {len(commits)}."
        )

    train, calib, test = chronological_split(commits, train_frac, calib_frac)
    if not train or not calib or not test:
        raise ValueError("Chronological split produced an empty train/calib/test set.")

    X_train, y_train = build_feature_matrix(train)
    X_calib, y_calib = build_feature_matrix(calib)
    X_test, y_test = build_feature_matrix(test)

    raw_model, calibrated_model, baseline_model = _train_models(
        X_train, y_train, X_calib, y_calib, seed
    )

    churn_test = np.array([c.features.get("churn", 0.0) for c in test], dtype=float)
    metrics = evaluate_model(calibrated_model, X_test, y_test, churn_test)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, churn_test)

    test_probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    fairness = fairness_report(test_probabilities, [c.author_tenure_days for c in test])

    all_ids = [c.id for c in train] + [c.id for c in calib] + [c.id for c in test]

    return TrainingResult(
        seed=seed,
        data_version=compute_data_version(all_ids),
        train_ids=[c.id for c in train],
        calib_ids=[c.id for c in calib],
        test_ids=[c.id for c in test],
        raw_model=raw_model,
        calibrated_model=calibrated_model,
        baseline_model=baseline_model,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        fairness=fairness,
        feature_columns=FEATURE_COLUMNS,
    )
