#!/usr/bin/env python3
"""Lists which user stories have at least one automated test.

Scans backend/tests and ml/tests for `@pytest.mark.story("US-xx")`, and
frontend test files for `// US-xx` comments. Usage: `python scripts/list_story_coverage.py`
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PY_STORY_RE = re.compile(r'@pytest\.mark\.story\(\s*["\'](?P<story>[\w-]+)["\']\s*\)')
JS_STORY_RE = re.compile(r'//\s*(?P<story>US-\d+)\b')


def find_python_stories(*roots: Path) -> dict[str, list[Path]]:
    hits: dict[str, list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("test_*.py"):
            text = path.read_text(encoding="utf-8")
            for match in PY_STORY_RE.finditer(text):
                hits.setdefault(match.group("story"), []).append(path)
    return hits


def find_frontend_stories(root: Path) -> dict[str, list[Path]]:
    hits: dict[str, list[Path]] = {}
    if not root.exists():
        return hits
    for path in list(root.rglob("*.test.ts")) + list(root.rglob("*.test.tsx")):
        text = path.read_text(encoding="utf-8")
        for match in JS_STORY_RE.finditer(text):
            hits.setdefault(match.group("story"), []).append(path)
    return hits


def main() -> int:
    hits = find_python_stories(ROOT / "backend" / "tests", ROOT / "ml" / "tests")
    hits_fe = find_frontend_stories(ROOT / "frontend")
    for story, paths in hits_fe.items():
        hits.setdefault(story, []).extend(paths)

    if not hits:
        print("No story-tagged tests found yet.")
        return 0

    print("Story coverage (story -> test files):")
    for story in sorted(hits, key=lambda s: (s.split("-")[0], int(s.split("-")[-1]))):
        rel_paths = [str(p.relative_to(ROOT)) for p in hits[story]]
        print(f"  {story}: {', '.join(rel_paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
