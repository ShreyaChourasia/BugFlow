from typing import Literal

from bugflow_ml.embeddings.duplicate_detection import embed_text, shared_phrases
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.queue import get_queue
from app.models.defect import DefectReport
from app.models.system import SystemConfig
from app.models.user import User
from app.schemas.defect import (
    DefectReportCreate,
    DuplicateCandidate,
    DuplicatesRead,
    SuggestionItem,
    SuggestionsRead,
)

INDEX_STATUS_KEY = "defect_index_status"
MIN_SUGGESTION_WORDS = 5  # US-16: only suggest once there's enough text to embed meaningfully
TOP_K = 5


def get_index_status(db: Session) -> Literal["ready", "rebuilding"]:
    row = db.get(SystemConfig, INDEX_STATUS_KEY)
    return "rebuilding" if row is not None and row.value == "rebuilding" else "ready"


def set_index_status(db: Session, status: Literal["ready", "rebuilding"]) -> None:
    row = db.get(SystemConfig, INDEX_STATUS_KEY)
    if row is None:
        db.add(SystemConfig(key=INDEX_STATUS_KEY, value=status))
    else:
        row.value = status
    db.commit()


def _search_similar(
    db: Session, embedding: list[float], *, exclude_id: int | None, limit: int
) -> list[tuple[DefectReport, float]]:
    query = select(DefectReport, DefectReport.embedding.cosine_distance(embedding)).where(
        DefectReport.embedding.isnot(None)
    )
    if exclude_id is not None:
        query = query.where(DefectReport.id != exclude_id)
    query = query.order_by(DefectReport.embedding.cosine_distance(embedding)).limit(limit)

    return [(report, 1.0 - distance) for report, distance in db.execute(query).all()]


def get_suggestions(db: Session, text: str) -> SuggestionsRead:
    """US-16: live suggestions while typing a new report, before it's saved."""
    if get_index_status(db) == "rebuilding":
        return SuggestionsRead(status="rebuilding", results=[])

    if len(text.split()) < MIN_SUGGESTION_WORDS:
        return SuggestionsRead(status="ready", results=[])

    embedding = embed_text(text)
    matches = _search_similar(db, embedding, exclude_id=None, limit=TOP_K)
    return SuggestionsRead(
        status="ready",
        results=[SuggestionItem(id=r.id, title=r.title, status=r.status) for r, _ in matches],
    )


def create_defect_report(db: Session, reporter: User, data: DefectReportCreate) -> DefectReport:
    """US-17: every new report is embedded immediately, so it's itself
    searchable as a future duplicate candidate right away."""
    embedding = embed_text(f"{data.title}\n{data.description}")
    report = DefectReport(
        repository_id=data.repository_id,
        reporter_id=reporter.id,
        title=data.title,
        description=data.description,
        component=data.component,
        embedding=embedding,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_duplicates(db: Session, report: DefectReport) -> DuplicatesRead:
    """US-17/US-18: candidate duplicates for an existing report, with the
    exact shared phrases highlighted against each candidate's own text."""
    if get_index_status(db) == "rebuilding":
        return DuplicatesRead(status="rebuilding", results=[])
    if report.embedding is None:
        return DuplicatesRead(status="ready", results=[])

    matches = _search_similar(db, list(report.embedding), exclude_id=report.id, limit=TOP_K)
    report_text = f"{report.title}\n{report.description}"
    return DuplicatesRead(
        status="ready",
        results=[
            DuplicateCandidate(
                id=candidate.id,
                title=candidate.title,
                status=candidate.status,
                score=score,
                shared_phrases=shared_phrases(
                    report_text, f"{candidate.title}\n{candidate.description}"
                ),
            )
            for candidate, score in matches
        ],
    )


def merge_report(db: Session, duplicate: DefectReport, original: DefectReport) -> None:
    """US-20: marks `duplicate` as merged into `original` and notifies its
    reporter asynchronously — via an RQ job, not inline in this request."""
    from app.workers.jobs.defect import notify_merge_job

    duplicate.duplicate_of_id = original.id
    duplicate.status = "duplicate"
    db.commit()

    get_queue().enqueue(notify_merge_job, duplicate.id, original.id)
