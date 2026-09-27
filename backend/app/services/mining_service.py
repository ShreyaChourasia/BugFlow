from datetime import UTC, datetime

from bugflow_ml.features.commit_features import compute_commit_features
from bugflow_ml.labeling.szz import find_bug_inducing_shas
from bugflow_ml.mining.backoff import retry_with_backoff
from bugflow_ml.mining.git_miner import MinedCommit, mine_commits
from bugflow_ml.mining.issue_links import extract_issue_refs, is_fix_commit
from sqlalchemy import any_, func, select, update
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.commit import Commit
from app.models.developer import Developer
from app.models.repository import MiningRun, Repository

logger = get_logger(__name__)


def _get_or_create_developer(
    db: Session, cache: dict[str, Developer], name: str, email: str, seen_at: datetime
) -> Developer:
    if email in cache:
        return cache[email]

    developer = db.scalar(select(Developer).where(any_(Developer.git_emails) == email))
    if developer is None:
        developer = Developer(name=name, git_emails=[email], joined_at=seen_at)
        db.add(developer)
        db.flush()

    cache[email] = developer
    return developer


def _seed_author_commit_counts(db: Session, repository_id: int) -> dict[int, int]:
    rows = db.execute(
        select(Commit.author_id, func.count())
        .where(Commit.repository_id == repository_id, Commit.author_id.is_not(None))
        .group_by(Commit.author_id)
    ).all()
    return {author_id: count for author_id, count in rows if author_id is not None}


def run_mining(db: Session, mining_run_id: int) -> None:
    """US-02/05/06/07: mines a repository's git history into `Commit` rows,
    with SZZ labelling for fix commits. Resumable (US-05) and safe to re-run
    for newly pushed commits (US-07) via `MiningRun.checkpoint["last_sha"]`.
    """
    mining_run = db.get(MiningRun, mining_run_id)
    if mining_run is None:
        raise ValueError(f"MiningRun {mining_run_id} not found")

    repository = db.get(Repository, mining_run.repository_id)
    if repository is None:
        raise ValueError(f"Repository {mining_run.repository_id} not found")

    mining_run.status = "running"
    mining_run.started_at = datetime.now(UTC)
    mining_run.error = None
    db.commit()

    checkpoint = mining_run.checkpoint or {}
    last_sha: str | None = checkpoint.get("last_sha")
    processed: int = checkpoint.get("processed", 0)

    developer_cache: dict[str, Developer] = {}
    author_commit_counts = _seed_author_commit_counts(db, repository.id)
    fix_shas: list[str] = []

    try:
        # ponytail: materializes the whole history in memory so a transient
        # clone/network failure can retry cleanly from scratch. Fine at the
        # "one or two demo repos" scale this phase targets; move to a
        # streaming retry if mining a repo large enough for that to matter.
        commits: list[MinedCommit] = retry_with_backoff(
            lambda: list(
                mine_commits(repository.url, after_sha=last_sha, branch=repository.default_branch)
            )
        )

        for mined in commits:
            last_sha = mined.sha

            if mined.is_merge:
                processed += 1
                mining_run.checkpoint = {"last_sha": last_sha, "processed": processed}
                db.commit()
                continue

            developer = _get_or_create_developer(
                db, developer_cache, mined.author_name, mined.author_email, mined.timestamp
            )
            prior_commits = author_commit_counts.get(developer.id, 0)
            fix = is_fix_commit(mined.message)
            features = compute_commit_features(
                files=mined.files,
                lines_added=mined.lines_added,
                lines_deleted=mined.lines_deleted,
                is_fix=fix,
                author_prior_commits=prior_commits,
            )

            db.add(
                Commit(
                    sha=mined.sha,
                    repository_id=repository.id,
                    author_id=developer.id,
                    message=mined.message,
                    timestamp=mined.timestamp,
                    lines_added=mined.lines_added,
                    lines_deleted=mined.lines_deleted,
                    files_changed=len(mined.files),
                    features=features,
                    is_fix=fix,
                    linked_issue_refs=extract_issue_refs(mined.message) or None,
                )
            )

            author_commit_counts[developer.id] = prior_commits + 1
            processed += 1
            if fix:
                fix_shas.append(mined.sha)

            mining_run.checkpoint = {"last_sha": last_sha, "processed": processed}
            db.commit()

        # SZZ labelling only for fix commits newly seen this run — a
        # re-mine for incremental ingestion (US-07) doesn't re-scan history
        # that was already labelled.
        for fix_sha in fix_shas:
            inducing_shas = find_bug_inducing_shas(repository.url, fix_sha)
            if inducing_shas:
                db.execute(
                    update(Commit)
                    .where(Commit.repository_id == repository.id, Commit.sha.in_(inducing_shas))
                    .values(is_bug_inducing=True)
                )
        db.commit()

        mining_run.status = "completed"
        mining_run.finished_at = datetime.now(UTC)
        db.commit()
    except Exception as exc:
        db.rollback()
        mining_run.status = "failed"
        mining_run.error = str(exc)
        db.commit()
        raise
