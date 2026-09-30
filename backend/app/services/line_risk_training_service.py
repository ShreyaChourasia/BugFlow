import math
from datetime import UTC, datetime
from typing import Any

import mlflow
from bugflow_ml.mining.git_miner import mine_specific_commits
from bugflow_ml.models.line_risk import LineExample, train_line_risk
from sklearn.pipeline import Pipeline
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.commit import Commit
from app.models.ml import MLModel
from app.models.repository import Repository

logger = get_logger(__name__)

TASK = "line_risk"


def _set_tracking_uri(tracking_uri: str | None) -> None:
    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)


def _json_safe(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else value


def _load_line_examples(db: Session) -> list[LineExample]:
    """Re-walks each repository's local checkout to recover the added-line
    text mining doesn't persist on `Commit` (see docs/decisions/003 —
    `Commit.features` only stores aggregate numbers). Grouped by repository
    so each repo is only cloned once, not once per commit."""
    commits = list(db.scalars(select(Commit).order_by(Commit.timestamp)))
    by_repository: dict[int, list[Commit]] = {}
    for commit in commits:
        by_repository.setdefault(commit.repository_id, []).append(commit)

    examples: list[LineExample] = []
    for repository_id, repo_commits in by_repository.items():
        repository = db.get(Repository, repository_id)
        if repository is None:
            continue
        shas = [c.sha for c in repo_commits]
        label_by_sha = {c.sha: c.is_bug_inducing for c in repo_commits}
        timestamp_by_sha = {c.sha: c.timestamp for c in repo_commits}

        try:
            mined_by_sha = {m.sha: m for m in mine_specific_commits(repository.url, shas)}
        except Exception as exc:
            logger.warning(
                "line_risk_training_skipped_repository", repository_id=repository_id, error=str(exc)
            )
            continue

        for sha, mined in mined_by_sha.items():
            if mined.is_merge:
                continue
            label = label_by_sha[sha]
            for file in mined.files:
                for line_no, text in file.diff_added:
                    examples.append(
                        LineExample(
                            text=text,
                            label=label,
                            timestamp=timestamp_by_sha[sha],
                            commit_sha=sha,
                            file_path=file.path,
                            line_no=line_no,
                        )
                    )
    return examples


def train_and_register_champion(
    db: Session, seed: int = 42, tracking_uri: str | None = None
) -> MLModel:
    """US-13/US-14: JITLine-style line-level model, trained on every mined
    repository's added lines with weak labels inherited from their commit's
    `is_bug_inducing` flag."""
    _set_tracking_uri(tracking_uri)
    mlflow.set_experiment(TASK)

    examples = _load_line_examples(db)
    result = train_line_risk(examples, seed=seed)

    with mlflow.start_run() as run:
        mlflow.log_param("seed", seed)
        mlflow.log_param("data_version", result.data_version)
        mlflow.log_param("train_size", result.train_size)
        mlflow.log_param("test_size", result.test_size)

        for key, value in result.metrics.items():
            if not math.isnan(value):
                mlflow.log_metric(key, value)

        pipeline = Pipeline([("tfidf", result.vectorizer), ("clf", result.model)])
        # Same reasoning as commit_risk: artifact_path=, plain pickle — the
        # vectorizer's custom tokenizer isn't skops-safe, and we only ever
        # load this back through our own runs:/ URI.
        mlflow.sklearn.log_model(
            pipeline, artifact_path="line_risk_pipeline", serialization_format="pickle"
        )

        run_id = run.info.run_id

    existing_champion = db.scalar(
        select(MLModel).where(MLModel.task == TASK, MLModel.stage == "champion")
    )
    new_score = result.metrics["top_k_accuracy"]
    old_score = (
        (existing_champion.metrics or {}).get("top_k_accuracy") if existing_champion else None
    )
    should_promote = existing_champion is None or (
        not math.isnan(new_score) and (old_score is None or new_score > old_score)
    )

    stored_metrics: dict[str, Any] = {k: _json_safe(v) for k, v in result.metrics.items()}

    model_row = MLModel(
        task=TASK,
        version=run_id,
        mlflow_run_id=run_id,
        stage="champion" if should_promote else "challenger",
        metrics=stored_metrics,
        data_version=result.data_version,
        seed=seed,
        trained_at=datetime.now(UTC),
    )
    db.add(model_row)

    if should_promote and existing_champion is not None:
        existing_champion.stage = "archived"

    db.commit()
    db.refresh(model_row)
    logger.info(
        "line_risk_training_complete", run_id=run_id, promoted=should_promote, score=new_score
    )
    return model_row
