"""Retries a transient failure (e.g. GitHub rate-limiting a clone/fetch)
with exponential backoff."""

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_with_backoff(fn: Callable[[], T], max_attempts: int = 5, base_delay: float = 1.0) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 == max_attempts:
                break
            time.sleep(base_delay * (2**attempt))
    assert last_exc is not None
    raise last_exc
