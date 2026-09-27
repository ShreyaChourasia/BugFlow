import pytest

from bugflow_ml.mining.issue_links import extract_issue_refs, is_fix_commit


@pytest.mark.story("US-03")
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Fixes #123", [123]),
        ("See #1 and #2, closes #3", [1, 2, 3]),
        ("No issue reference here", []),
        ("Duplicate refs #5 ... #5 again", [5]),
    ],
)
def test_extract_issue_refs(message: str, expected: list[int]) -> None:
    assert extract_issue_refs(message) == expected


@pytest.mark.story("US-03")
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Fixes #123: off-by-one", True),
        ("fixed the flaky test", True),
        ("Closes #9", True),
        ("Resolved the merge conflict", True),
        ("Add a new feature", False),
        ("Refactor the widget prefixer", False),  # "fix" must be a whole word
    ],
)
def test_is_fix_commit(message: str, expected: bool) -> None:
    assert is_fix_commit(message) is expected
