"""US-03: match commits to the issues they reference, and flag fixing
commits (the classic JIT defect-prediction "fix" feature)."""

import re

_ISSUE_REF_RE = re.compile(r"#(\d+)")
_FIX_KEYWORDS_RE = re.compile(r"\b(fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b", re.IGNORECASE)


def extract_issue_refs(message: str) -> list[int]:
    """Every issue number referenced in a commit message, e.g. "#123" or
    "fixes #123" — order-preserving, de-duplicated."""
    seen: dict[int, None] = {}
    for match in _ISSUE_REF_RE.finditer(message):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def is_fix_commit(message: str) -> bool:
    """True if the message contains a fixing keyword (fix/fixes/fixed,
    close/closes/closed, resolve/resolves/resolved), independent of whether
    an issue number is present."""
    return bool(_FIX_KEYWORDS_RE.search(message))
