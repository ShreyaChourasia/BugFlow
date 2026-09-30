import pytest
from sqlalchemy.orm import Session

import app.services.defect_service as defect_service_module
from app.models.defect import DefectReport
from app.models.repository import Repository
from app.schemas.defect import DefectReportCreate
from app.services import defect_service

from .conftest import make_user


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[object, tuple, dict]] = []

    def enqueue(self, func: object, *args: object, **kwargs: object) -> None:
        self.enqueued.append((func, args, kwargs))


@pytest.fixture
def fake_queue(monkeypatch: pytest.MonkeyPatch) -> _FakeQueue:
    queue = _FakeQueue()
    monkeypatch.setattr(defect_service_module, "get_queue", lambda: queue)
    return queue


@pytest.fixture
def repo(db_session: Session) -> Repository:
    repository = Repository(name="demo", url="/tmp/x")
    db_session.add(repository)
    db_session.commit()
    return repository


def _make_report(
    db_session: Session, repo: Repository, title: str, description: str
) -> DefectReport:
    from app.models.enums import Role

    reporter = make_user(
        db_session, f"{title.lower().replace(' ', '-')}@example.com", Role.REPORTER
    )
    report = defect_service.create_defect_report(
        db_session,
        reporter,
        DefectReportCreate(repository_id=repo.id, title=title, description=description),
    )
    return report


@pytest.mark.story("US-17")
def test_create_defect_report_stores_an_embedding(db_session: Session, repo: Repository) -> None:
    report = _make_report(
        db_session, repo, "Login crash", "The app crashes with a null pointer on login"
    )

    assert report.id is not None
    assert report.embedding is not None
    assert len(report.embedding) == 384


@pytest.mark.story("US-17")
@pytest.mark.story("US-18")
def test_get_duplicates_finds_the_similar_report_and_highlights_shared_phrases(
    db_session: Session, repo: Repository
) -> None:
    _make_report(
        db_session, repo, "Login NPE", "Clicking login causes a null pointer exception crash"
    )
    new_report = _make_report(
        db_session,
        repo,
        "Crash on login",
        "The app crashes with a null pointer exception when clicking login",
    )

    duplicates = defect_service.get_duplicates(db_session, new_report)

    assert duplicates.status == "ready"
    assert len(duplicates.results) == 1
    assert "null pointer exception" in " ".join(duplicates.results[0].shared_phrases)


@pytest.mark.story("US-16")
def test_get_suggestions_below_min_word_count_returns_no_results(db_session: Session) -> None:
    result = defect_service.get_suggestions(db_session, "app crashes")

    assert result.status == "ready"
    assert result.results == []


@pytest.mark.story("US-16")
def test_get_suggestions_returns_matches_once_enough_words(
    db_session: Session, repo: Repository
) -> None:
    _make_report(
        db_session, repo, "Timeout error", "Requests to the API time out after thirty seconds"
    )

    result = defect_service.get_suggestions(
        db_session, "My requests to the API keep timing out after about thirty seconds"
    )

    assert result.status == "ready"
    assert any(item.title == "Timeout error" for item in result.results)


def test_index_status_defaults_to_ready(db_session: Session) -> None:
    assert defect_service.get_index_status(db_session) == "ready"


@pytest.mark.story("NFR-US-02")
def test_rebuilding_index_status_short_circuits_search(
    db_session: Session, repo: Repository
) -> None:
    _make_report(db_session, repo, "Slow query", "The dashboard takes forever to load")
    defect_service.set_index_status(db_session, "rebuilding")

    suggestions = defect_service.get_suggestions(db_session, "the dashboard is really slow today")
    assert suggestions.status == "rebuilding"
    assert suggestions.results == []


@pytest.mark.story("US-20")
def test_merge_report_marks_duplicate_and_enqueues_notification(
    db_session: Session, repo: Repository, fake_queue: _FakeQueue
) -> None:
    original = _make_report(db_session, repo, "Original bug", "Something is broken in the UI")
    duplicate = _make_report(db_session, repo, "Duplicate bug", "The UI is also broken here")

    defect_service.merge_report(db_session, duplicate, original)

    db_session.refresh(duplicate)
    assert duplicate.duplicate_of_id == original.id
    assert duplicate.status == "duplicate"
    assert len(fake_queue.enqueued) == 1
    assert fake_queue.enqueued[0][1] == (duplicate.id, original.id)
