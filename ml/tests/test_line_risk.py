from datetime import UTC, datetime, timedelta

import pytest

from bugflow_ml.models.line_risk import (
    LineExample,
    chronological_split,
    explain_line,
    score_lines,
    train_line_risk,
)

BASE = datetime(2024, 1, 1, tzinfo=UTC)
RISKY = ["except: pass", "x = None", "if x == None:", "TODO fix this", "eval(user_input)"]
SAFE = ["return result", "x = compute(y)", "if x is not None:", "log.info(msg)", "def foo(a, b):"]


def _examples(n: int = 300) -> list[LineExample]:
    examples = []
    for i in range(n):
        commit_sha = f"sha{i // 3}"
        is_bug = (i // 3) % 5 == 0
        text = RISKY[i % len(RISKY)] if is_bug else SAFE[i % len(SAFE)]
        examples.append(
            LineExample(
                text=text,
                label=is_bug,
                timestamp=BASE + timedelta(hours=i),
                commit_sha=commit_sha,
                file_path="a.py",
                line_no=i,
            )
        )
    return examples


@pytest.mark.story("US-13")
def test_train_line_risk_separates_risky_from_safe_snippets() -> None:
    result = train_line_risk(_examples(), seed=42)

    assert result.train_size + result.test_size == 300
    assert result.metrics["roc_auc"] > 0.9

    scores = score_lines(result.vectorizer, result.model, ["except: pass", "return result"])
    assert scores[0] > scores[1]


@pytest.mark.story("US-13")
def test_train_line_risk_rejects_too_few_examples() -> None:
    with pytest.raises(ValueError, match="at least"):
        train_line_risk(_examples(10), seed=42)


@pytest.mark.story("US-14")
def test_explain_line_attributes_the_risky_tokens() -> None:
    result = train_line_risk(_examples(), seed=42)

    explanation = explain_line(result.vectorizer, result.model, "except: pass")

    tokens = [token for token, _ in explanation]
    assert "except" in tokens or "pass" in tokens
    assert all(
        contribution > 0 for token, contribution in explanation if token in ("except", "pass")
    )


def test_chronological_split_no_leakage() -> None:
    train, test = chronological_split(_examples(), train_frac=0.8)

    assert len(train) + len(test) == 300
    assert max(e.timestamp for e in train) <= min(e.timestamp for e in test)
