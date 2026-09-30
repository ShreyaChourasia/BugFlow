from bugflow_ml.features.commit_features import compute_commit_features
from bugflow_ml.mining.git_miner import ModifiedFileInfo, mine_specific_commits
from bugflow_ml.mining.issue_links import extract_issue_refs, is_fix_commit
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.integrations.github.client import CommitDetails, GitHubClient, get_github_client
from app.models.commit import Commit
from app.models.pull_request import LineRisk, PullRequest, RiskPrediction
from app.models.repository import Repository
from app.services.developer_service import count_author_commits, get_or_create_developer
from app.services.line_prediction_service import ScoredLine, score_and_explain_lines
from app.services.prediction_service import predict_commit

logger = get_logger(__name__)

NOT_APPLICABLE = "not_applicable"  # US-13 AC2: below the repo's risk threshold
UNAVAILABLE = "unavailable"  # no line-risk model yet, or its diff couldn't be fetched
AVAILABLE = "available"


def _fetch_and_store_commit(
    db: Session, repository: Repository, sha: str, client: GitHubClient
) -> Commit | None:
    """A PR's head commit usually hasn't been mined yet (it's brand new) —
    fetch it live from GitHub and store it exactly like mining would, so
    scoring can reuse Phase 3's prediction_service unchanged."""
    try:
        details: CommitDetails = client.fetch_commit_details(repository, sha)
    except LookupError:
        return None

    if details.is_merge:
        return None

    developer = get_or_create_developer(
        db, {}, details.author_name, details.author_email, details.timestamp
    )
    prior_commits = count_author_commits(db, repository.id, developer.id)
    fix = is_fix_commit(details.message)
    features = compute_commit_features(
        files=details.files,
        lines_added=details.lines_added,
        lines_deleted=details.lines_deleted,
        is_fix=fix,
        author_prior_commits=prior_commits,
    )

    commit = Commit(
        sha=details.sha,
        repository_id=repository.id,
        author_id=developer.id,
        message=details.message,
        timestamp=details.timestamp,
        lines_added=details.lines_added,
        lines_deleted=details.lines_deleted,
        files_changed=len(details.files),
        features=features,
        is_fix=fix,
        linked_issue_refs=extract_issue_refs(details.message) or None,
    )
    db.add(commit)
    db.flush()
    return commit


def _get_modified_files(repository: Repository, commit: Commit) -> list[ModifiedFileInfo]:
    """Neither a mined `Commit` nor one fetched fresh from GitHub keeps added-
    line text (see docs/decisions/003) — re-walk just this one commit via
    PyDriller to recover it.
    ponytail: reclones the repo per call; fine at demo scale, cache the
    checkout (like mining_service's per-run clone) if this gets hot."""
    mined = next(mine_specific_commits(repository.url, [commit.sha]), None)
    return mined.files if mined else []


def _score_line_risk(
    db: Session,
    repository: Repository,
    commit: Commit,
    risk_prediction: RiskPrediction,
    tracking_uri: str | None,
) -> tuple[str, list[LineRisk]]:
    """US-13: only run line-level analysis once a commit clears the repo's
    risk threshold (docs/decisions/003 reuses risk_threshold rather than
    adding a second knob)."""
    if risk_prediction.calibrated_probability < repository.risk_threshold:
        return NOT_APPLICABLE, []

    try:
        files = _get_modified_files(repository, commit)
        scored: list[ScoredLine] = score_and_explain_lines(
            db, files, repository.line_risk_top_n, tracking_uri=tracking_uri
        )
    except LookupError:
        return UNAVAILABLE, []
    except Exception as exc:
        # A missing/unclonable repo must not turn an otherwise-successful
        # commit-risk assessment into a full "unavailable" failure.
        logger.warning("line_risk_scoring_failed", commit_id=commit.id, error=str(exc))
        return UNAVAILABLE, []

    if not scored:
        return UNAVAILABLE, []

    lines = [
        LineRisk(
            prediction_id=risk_prediction.id,
            file_path=line.file_path,
            line_no=line.line_no,
            code=line.code,
            risk_score=line.risk_score,
            rank=line.rank,
            reason=line.reason,
        )
        for line in scored
    ]
    db.add_all(lines)
    db.flush()
    return AVAILABLE, lines


def _decide_conclusion(repository: Repository, risk_prediction: RiskPrediction) -> tuple[str, str]:
    """US-47: merge-blocking is off by default and configured per repo — a
    high-risk PR is only ever reported as a failing check when the repo has
    explicitly opted in."""
    title = f"Risk: {risk_prediction.risk_level} ({risk_prediction.calibrated_probability:.0%})"
    if not repository.merge_blocking_enabled:
        return "neutral", title

    is_high_risk = risk_prediction.calibrated_probability >= repository.risk_threshold
    return ("failure" if is_high_risk else "success"), title


def _format_summary(risk_prediction: RiskPrediction, explanation_text: str) -> str:
    return (
        f"Calibrated risk: {risk_prediction.calibrated_probability:.0%} "
        f"(confidence {risk_prediction.confidence:.0%}).\n\n{explanation_text}"
    )


def _format_comment(
    risk_prediction: RiskPrediction,
    explanation_text: str,
    line_risk_status: str,
    lines: list[LineRisk],
) -> str:
    body = (
        f"### BugFlow risk assessment\n\n"
        f"**Risk level:** {risk_prediction.risk_level} "
        f"({risk_prediction.calibrated_probability:.0%} calibrated probability, "
        f"{risk_prediction.confidence:.0%} confidence)\n\n"
        f"{explanation_text}"
    )

    if line_risk_status == NOT_APPLICABLE:
        body += "\n\nThis change did not meet the threshold for line-level analysis."
    elif lines:
        body += "\n\n**Riskiest lines:**\n\n" + "\n".join(
            f"- `{line.file_path}:{line.line_no}` — {line.reason}"
            for line in sorted(lines, key=lambda line: line.rank)
        )

    body += (
        f"\n\n<sub>Model version `{risk_prediction.model_version}` — automated, may be wrong.</sub>"
    )
    return body


def _complete_check(
    db: Session,
    client: GitHubClient,
    repository: Repository,
    pr: PullRequest,
    *,
    conclusion: str,
    title: str,
    summary: str,
) -> None:
    if pr.check_run_id:
        client.update_check_run(
            repository, pr.check_run_id, conclusion=conclusion, title=title, summary=summary
        )
    pr.check_status = "completed"
    pr.check_conclusion = conclusion
    pr.check_summary = summary
    db.commit()


def score_pull_request(db: Session, pull_request_id: int, tracking_uri: str | None = None) -> None:
    """US-08/US-11/US-46/US-47, C7: scores a PR's head commit and posts the
    result (real GitHub API, or the fake sink in offline/replay mode —
    app.integrations.github.client picks which). Reuses Phase 3's
    prediction_service unchanged; a PR is just "score this one commit."""
    pr = db.get(PullRequest, pull_request_id)
    if pr is None:
        raise ValueError(f"PullRequest {pull_request_id} not found")

    repository = db.get(Repository, pr.repository_id)
    if repository is None:
        raise ValueError(f"Repository {pr.repository_id} not found")

    client = get_github_client(repository)

    try:
        commit = db.scalar(
            select(Commit).where(Commit.repository_id == repository.id, Commit.sha == pr.head_sha)
        )
        if commit is None:
            commit = _fetch_and_store_commit(db, repository, pr.head_sha, client)

        if commit is None:
            _complete_check(
                db,
                client,
                repository,
                pr,
                conclusion="neutral",
                title="No assessment available",
                summary=(
                    "This commit hasn't been mined yet, and no GitHub API access is "
                    "configured to fetch it live."
                ),
            )
            return

        try:
            risk_prediction, explanation = predict_commit(db, commit, tracking_uri=tracking_uri)
        except LookupError as exc:
            # US-08 AC2: no trained model yet — say so, don't block.
            _complete_check(
                db,
                client,
                repository,
                pr,
                conclusion="neutral",
                title="No assessment available",
                summary=f"No trained risk model is available yet. ({exc})",
            )
            return

        if pr.author_id is None and commit.author_id is not None:
            pr.author_id = commit.author_id

        line_risk_status, line_risks = _score_line_risk(
            db, repository, commit, risk_prediction, tracking_uri
        )

        conclusion, title = _decide_conclusion(repository, risk_prediction)
        summary = _format_summary(risk_prediction, explanation.text)
        _complete_check(
            db, client, repository, pr, conclusion=conclusion, title=title, summary=summary
        )

        comment_body = _format_comment(
            risk_prediction, explanation.text, line_risk_status, line_risks
        )
        comment_id = client.upsert_comment(repository, pr.number, pr.comment_id, comment_body)
        pr.comment_id = comment_id
        pr.comment_body = comment_body
        db.commit()
    except Exception as exc:
        # C7: our own service failing must never block a merge or leave a
        # stale/wrong result — report "unavailable" with neutral conclusion
        # regardless of this repo's merge-blocking setting.
        db.rollback()
        pr = db.get(PullRequest, pull_request_id)
        logger.warning("pr_scoring_failed", pull_request_id=pull_request_id, error=str(exc))
        if pr is None:
            raise
        try:
            _complete_check(
                db,
                client,
                repository,
                pr,
                conclusion="neutral",
                title="BugFlow unavailable",
                summary=f"Risk assessment could not be computed: {exc}",
            )
        except Exception:
            logger.warning(
                "pr_scoring_unavailable_update_also_failed", pull_request_id=pull_request_id
            )
        raise
