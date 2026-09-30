"""US-13/US-14: JITLine-style line-level risk. Tokenises each added line,
vectorises with TF-IDF, and trains a logistic regression on a *weak* label —
every added line in a commit inherits that commit's `is_bug_inducing` flag,
since we don't have (and this phase doesn't attempt) finer-grained
ground truth about which exact line introduced a bug. This is the standard
JITLine approach, not a shortcut unique to this project — see docs/ml.md for
what it does and doesn't get right.

Token attribution (US-14) comes directly from the linear model's own
coefficients: for a scored line, `coefficient[token] * tfidf_value[token]`
for each token present *is* the exact contribution of that token to the
line's score — not an approximation, unlike LIME/SHAP on a non-linear model.
No extra explainability dependency needed for a linear model.
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

TOKEN_PATTERN = r"[A-Za-z_]\w*|[0-9]+|[^\sA-Za-z0-9_]"
MIN_TRAINING_LINES = 50


@dataclass
class LineExample:
    text: str
    label: bool
    timestamp: datetime
    commit_sha: str
    file_path: str
    line_no: int


@dataclass
class LineRiskTrainingResult:
    seed: int
    data_version: str
    vectorizer: Any
    model: Any
    metrics: dict[str, float]
    train_size: int
    test_size: int


def compute_data_version(examples: list[LineExample]) -> str:
    canonical = ",".join(
        sorted(f"{e.commit_sha}:{e.file_path}:{e.line_no}" for e in examples)
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def chronological_split(
    examples: list[LineExample], train_frac: float = 0.8
) -> tuple[list[LineExample], list[LineExample]]:
    """C9: same discipline as the commit-risk model — train on the past,
    test on the future. Sorted by (timestamp, commit_sha, file_path, line_no)
    for a fully deterministic order."""
    ordered = sorted(examples, key=lambda e: (e.timestamp, e.commit_sha, e.file_path, e.line_no))
    n_train = int(len(ordered) * train_frac)
    train, test = ordered[:n_train], ordered[n_train:]
    if train and test:
        assert max(e.timestamp for e in train) <= min(e.timestamp for e in test)
    return train, test


def _tokenize(text: str) -> list[str]:
    return re.findall(TOKEN_PATTERN, text)


def train_line_risk(
    examples: list[LineExample], seed: int = 42, train_frac: float = 0.8
) -> LineRiskTrainingResult:
    if len(examples) < MIN_TRAINING_LINES:
        raise ValueError(
            f"Need at least {MIN_TRAINING_LINES} labelled lines to train; got {len(examples)}."
        )

    train, test = chronological_split(examples, train_frac)
    if not train or not test:
        raise ValueError("Chronological split produced an empty train/test set.")

    vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None, lowercase=False)
    X_train = vectorizer.fit_transform([e.text for e in train])
    y_train = np.array([1 if e.label else 0 for e in train], dtype=int)
    X_test = vectorizer.transform([e.text for e in test])
    y_test = np.array([1 if e.label else 0 for e in test], dtype=int)

    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(X_train, y_train)

    metrics = _evaluate(model, X_test, y_test, test)

    return LineRiskTrainingResult(
        seed=seed,
        data_version=compute_data_version(examples),
        vectorizer=vectorizer,
        model=model,
        metrics=metrics,
        train_size=len(train),
        test_size=len(test),
    )


def _evaluate(
    model: Any, X_test: Any, y_test: np.ndarray, test: list[LineExample]
) -> dict[str, float]:
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float] = {}
    if len(set(y_test.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    metrics["recall_at_20pct_lines"] = _recall_at_20pct_lines(y_test, y_prob)
    metrics["top_k_accuracy"] = _top_k_accuracy_per_commit(test, y_prob, k=5)
    return metrics


def _recall_at_20pct_lines(y_test: np.ndarray, y_prob: np.ndarray, k: float = 0.2) -> float:
    total_positive = y_test.sum()
    if total_positive == 0:
        return 0.0
    cutoff = max(1, int(len(y_test) * k))
    order = np.argsort(-y_prob)[:cutoff]
    captured = y_test[order].sum()
    return float(captured / total_positive)


def _top_k_accuracy_per_commit(test: list[LineExample], y_prob: np.ndarray, k: int) -> float:
    """For each test commit that has at least one positive (weakly-labelled)
    line, would showing its top-k highlighted lines have surfaced one? The
    metric that actually matches the feature's use case (US-13), not just a
    global ranking metric."""
    by_commit: dict[str, list[tuple[float, bool]]] = {}
    for example, prob in zip(test, y_prob, strict=True):
        by_commit.setdefault(example.commit_sha, []).append((prob, example.label))

    positive_commits = [lines for lines in by_commit.values() if any(label for _, label in lines)]
    if not positive_commits:
        return 0.0

    hits = 0
    for lines in positive_commits:
        top_k = sorted(lines, key=lambda pair: -pair[0])[:k]
        if any(label for _, label in top_k):
            hits += 1
    return hits / len(positive_commits)


def score_lines(vectorizer: Any, model: Any, texts: list[str]) -> list[float]:
    if not texts:
        return []
    X = vectorizer.transform(texts)
    probs = model.predict_proba(X)[:, 1]
    return [float(p) for p in probs]


def explain_line(vectorizer: Any, model: Any, text: str, top_n: int = 3) -> list[tuple[str, float]]:
    """Exact token attribution for a linear model: coefficient * tfidf value,
    for every token actually present in this line."""
    X = vectorizer.transform([text])
    row = X.tocoo()
    coefficients = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    contributions = [
        (feature_names[col], float(coefficients[col] * value))
        for col, value in zip(row.col, row.data, strict=True)
    ]
    contributions.sort(key=lambda pair: -abs(pair[1]))
    return contributions[:top_n]
