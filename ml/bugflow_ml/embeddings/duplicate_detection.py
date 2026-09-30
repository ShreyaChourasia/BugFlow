"""US-16/US-17/US-18: sentence-transformer embeddings for defect-report
duplicate detection, plus shared-phrase highlighting between two reports'
text. The embedding model itself is loaded once per process (it's ~90MB and
takes real time to load) and reused for every call.
"""

import re
from typing import Any

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: Any = None


def _get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-encodes many texts in one call — the throughput that makes the
    300k-report load test (C2/US-19) feasible; encoding one at a time would
    pay per-call overhead 300k times over."""
    if not texts:
        return []
    vectors = _get_model().encode(texts, batch_size=256, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def shared_phrases(text_a: str, text_b: str, min_words: int = 2, top_n: int = 5) -> list[str]:
    """US-18: contiguous word runs common to both texts, via `difflib`'s
    matching-blocks algorithm on word-tokenized sequences — stdlib-only,
    and unlike a bag-of-words overlap it finds actual multi-word phrases
    ("null pointer exception", not just "null" and "exception" separately)."""
    from difflib import SequenceMatcher

    words_a = _tokenize(text_a)
    words_b = _tokenize(text_b)
    if not words_a or not words_b:
        return []

    matcher = SequenceMatcher(None, words_a, words_b, autojunk=False)
    phrases = [
        " ".join(words_a[block.a : block.a + block.size])
        for block in matcher.get_matching_blocks()
        if block.size >= min_words
    ]
    # Longest, most distinctive phrases first; de-duplicated but order-stable.
    seen: set[str] = set()
    ranked = []
    for phrase in sorted(phrases, key=len, reverse=True):
        if phrase not in seen:
            seen.add(phrase)
            ranked.append(phrase)
    return ranked[:top_n]
