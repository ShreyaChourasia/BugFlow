import pytest

from bugflow_ml.models.triage_classifier import (
    MIN_TRAINING_EXAMPLES,
    explain_label,
    generate_bootstrap_examples,
    predict_label,
    should_abstain,
    train_priority_classifier,
    train_severity_classifier,
)
from bugflow_ml.taxonomy import Priority, Severity


@pytest.mark.story("US-21")
@pytest.mark.story("US-22")
def test_train_severity_and_priority_classifiers_on_bootstrap_data() -> None:
    examples = generate_bootstrap_examples(300, seed=42)

    severity_result = train_severity_classifier(examples, seed=42)
    priority_result = train_priority_classifier(examples, seed=42)

    assert severity_result.metrics["macro_f1"] > 0.8
    assert priority_result.metrics["macro_f1"] > 0.8
    assert set(severity_result.metrics["labels"]) <= {s.value for s in Severity}
    assert set(priority_result.metrics["labels"]) <= {p.value for p in Priority}


def test_train_rejects_too_few_examples() -> None:
    examples = generate_bootstrap_examples(MIN_TRAINING_EXAMPLES - 1, seed=42)

    with pytest.raises(ValueError, match="at least"):
        train_severity_classifier(examples, seed=42)


@pytest.mark.story("US-21")
def test_predict_label_returns_the_matching_severity() -> None:
    examples = generate_bootstrap_examples(300, seed=42)
    result = train_severity_classifier(examples, seed=42)

    label, confidence = predict_label(
        result.vectorizer, result.model, "The api feature has an issue: data loss."
    )

    assert label == Severity.BLOCKER
    assert 0.0 < confidence <= 1.0


@pytest.mark.story("US-21")
def test_explain_label_surfaces_the_driving_words() -> None:
    examples = generate_bootstrap_examples(300, seed=42)
    result = train_severity_classifier(examples, seed=42)
    text = "The api feature has an issue: data loss."

    label, _ = predict_label(result.vectorizer, result.model, text)
    contributions = explain_label(result.vectorizer, result.model, text, label)

    tokens = [token for token, _ in contributions]
    assert "data" in tokens or "loss" in tokens


@pytest.mark.story("US-21")
def test_should_abstain_below_minimum_word_count() -> None:
    assert should_abstain("too short")
    assert not should_abstain("this description has more than enough words to classify")
