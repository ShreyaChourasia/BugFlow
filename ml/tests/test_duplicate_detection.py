import pytest

from bugflow_ml.embeddings.duplicate_detection import EMBEDDING_DIM, embed_texts, shared_phrases


@pytest.mark.story("US-16")
def test_embed_texts_returns_one_vector_per_text_with_expected_dim() -> None:
    vectors = embed_texts(["login crashes", "logout works fine"])

    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_embed_texts_empty_input_returns_empty_list() -> None:
    assert embed_texts([]) == []


@pytest.mark.story("US-18")
def test_shared_phrases_finds_the_common_multi_word_run() -> None:
    a = "The app crashes with a null pointer exception when clicking submit"
    b = "Clicking submit causes a null pointer exception and the app crashes"

    phrases = shared_phrases(a, b)

    assert "a null pointer exception" in phrases


def test_shared_phrases_ignores_single_word_matches() -> None:
    phrases = shared_phrases("the button is red", "the sky is blue", min_words=2)

    assert phrases == []


def test_shared_phrases_empty_text_returns_empty_list() -> None:
    assert shared_phrases("", "something") == []
