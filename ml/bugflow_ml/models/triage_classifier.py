"""US-21/US-22/US-23: severity/priority classification. TF-IDF + a linear
model per target (§8's stated baseline — "optional DistilBERT later" is not
attempted here), trained on whichever defect reports already carry a human
-decided label. Abstains (US-21 AC2) rather than guessing when there isn't
enough text to go on.
"""

import hashlib
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from bugflow_ml.taxonomy import Priority, Severity

MIN_WORDS_TO_CLASSIFY = 5  # US-21 AC2: abstain below this, ask for more detail
MIN_TRAINING_EXAMPLES = 25
TOKEN_PATTERN = r"[A-Za-z_]\w*|[0-9]+"


@dataclass
class TriageExample:
    text: str
    severity: str
    priority: str
    timestamp: datetime
    defect_id: int


@dataclass
class TriageTrainingResult:
    seed: int
    data_version: str
    vectorizer: Any
    model: Any
    metrics: dict[str, Any]
    train_size: int
    test_size: int


def word_count(text: str) -> int:
    return len(text.split())


def should_abstain(text: str, min_words: int = MIN_WORDS_TO_CLASSIFY) -> bool:
    return word_count(text) < min_words


def compute_data_version(examples: list[TriageExample]) -> str:
    canonical = ",".join(sorted(f"{e.defect_id}:{e.severity}:{e.priority}" for e in examples))
    return hashlib.sha256(canonical.encode()).hexdigest()


def chronological_split(
    examples: list[TriageExample], train_frac: float = 0.8
) -> tuple[list[TriageExample], list[TriageExample]]:
    """C9: train on the past, test on the future, same discipline as every
    other model in this project."""
    ordered = sorted(examples, key=lambda e: (e.timestamp, e.defect_id))
    n_train = int(len(ordered) * train_frac)
    train, test = ordered[:n_train], ordered[n_train:]
    if train and test:
        assert max(e.timestamp for e in train) <= min(e.timestamp for e in test)
    return train, test


def _tokenize(text: str) -> list[str]:
    return re.findall(TOKEN_PATTERN, text.lower())


def _train_one_target(
    examples: list[TriageExample], get_label: Any, seed: int, train_frac: float
) -> tuple[Any, Any, dict[str, Any], int, int]:
    train, test = chronological_split(examples, train_frac)
    if not train or not test:
        raise ValueError("Chronological split produced an empty train/test set.")

    vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None, lowercase=False)
    X_train = vectorizer.fit_transform([e.text for e in train])
    y_train = [get_label(e) for e in train]
    X_test = vectorizer.transform([e.text for e in test])
    y_test = [get_label(e) for e in test]

    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = list(model.classes_)
    metrics = {
        "macro_f1": float(
            f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
    return vectorizer, model, metrics, len(train), len(test)


def _train_target(
    examples: list[TriageExample], get_label: Any, seed: int, train_frac: float
) -> TriageTrainingResult:
    if len(examples) < MIN_TRAINING_EXAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAINING_EXAMPLES} labelled examples to train; "
            f"got {len(examples)}."
        )
    vectorizer, model, metrics, train_size, test_size = _train_one_target(
        examples, get_label, seed, train_frac
    )
    return TriageTrainingResult(
        seed=seed,
        data_version=compute_data_version(examples),
        vectorizer=vectorizer,
        model=model,
        metrics=metrics,
        train_size=train_size,
        test_size=test_size,
    )


def train_severity_classifier(
    examples: list[TriageExample], seed: int = 42, train_frac: float = 0.8
) -> TriageTrainingResult:
    return _train_target(examples, lambda e: e.severity, seed, train_frac)


def train_priority_classifier(
    examples: list[TriageExample], seed: int = 42, train_frac: float = 0.8
) -> TriageTrainingResult:
    return _train_target(examples, lambda e: e.priority, seed, train_frac)


def predict_label(vectorizer: Any, model: Any, text: str) -> tuple[str, float]:
    """Returns (predicted_label, confidence) — confidence is the predicted
    class's own probability, not a separate calibration step (§8 doesn't ask
    for one here, unlike commit-risk)."""
    X = vectorizer.transform([text])
    probabilities = model.predict_proba(X)[0]
    best_idx = probabilities.argmax()
    return str(model.classes_[best_idx]), float(probabilities[best_idx])


def explain_label(
    vectorizer: Any, model: Any, text: str, predicted_label: str, top_n: int = 3
) -> list[tuple[str, float]]:
    """Top words/features (§8) behind this specific prediction: exact
    coefficient × tfidf_value for the predicted class's own row of
    coefficients — the same exact-attribution trick line_risk.py uses, just
    indexed into one row of a multi-class linear model instead of a binary
    one."""
    X = vectorizer.transform([text])
    row = X.tocoo()
    class_idx = list(model.classes_).index(predicted_label)
    coefficients = model.coef_[class_idx]
    feature_names = vectorizer.get_feature_names_out()

    contributions = [
        (feature_names[col], float(coefficients[col] * value))
        for col, value in zip(row.col, row.data, strict=True)
    ]
    contributions.sort(key=lambda pair: -abs(pair[1]))
    return contributions[:top_n]


# --- bootstrap data -----------------------------------------------------
#
# §8: "public Bugzilla datasets such as the Eclipse/Mozilla duplicate bug
# report data" for severity/triage at scale — no verified, licensed copy of
# one is available in this environment (same constraint Phase 6 hit for its
# load test; see docs/decisions/004 and 005). Real human triage decisions
# (DefectReport rows with severity/priority already set) are always
# preferred and used first; this generator only fills the gap so the
# classifier has something to train on before any real decisions exist.

_SEVERITY_KEYWORDS: dict[str, list[str]] = {
    Severity.BLOCKER: ["data loss", "cannot log in", "complete outage", "crashes on startup"],
    Severity.CRITICAL: ["security vulnerability", "payment fails", "crashes frequently"],
    Severity.MAJOR: ["major feature broken", "incorrect results", "frequent errors"],
    Severity.MINOR: ["minor glitch", "occasional error", "small inconsistency"],
    Severity.TRIVIAL: ["cosmetic issue", "typo in label", "minor styling"],
}
_PRIORITY_KEYWORDS: dict[str, list[str]] = {
    Priority.P1: ["fix immediately", "blocking all users", "urgent"],
    Priority.P2: ["fix this sprint", "high impact"],
    Priority.P3: ["fix when convenient", "moderate impact"],
    Priority.P4: ["low impact", "minor annoyance"],
    Priority.P5: ["nice to have", "cosmetic only"],
}
_COMPONENTS = ["login", "checkout", "search", "dashboard", "billing", "api"]


def generate_bootstrap_examples(n: int = 300, seed: int = 42) -> list[TriageExample]:
    rng = random.Random(seed)
    severities = list(_SEVERITY_KEYWORDS)
    priorities = list(_PRIORITY_KEYWORDS)
    base = datetime(2024, 1, 1)

    examples = []
    for i in range(n):
        severity = rng.choice(severities)
        priority = rng.choice(priorities)
        component = rng.choice(_COMPONENTS)
        phrase_s = rng.choice(_SEVERITY_KEYWORDS[severity])
        phrase_p = rng.choice(_PRIORITY_KEYWORDS[priority])
        text = f"The {component} feature has an issue: {phrase_s}. {phrase_p}."
        examples.append(
            TriageExample(
                text=text,
                severity=severity,
                priority=priority,
                timestamp=base + timedelta(minutes=i),
                defect_id=-(i + 1),  # negative ids: never collide with real rows
            )
        )
    return examples
