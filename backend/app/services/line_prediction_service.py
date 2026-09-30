from dataclasses import dataclass
from typing import Any

import mlflow
from bugflow_ml.mining.git_miner import ModifiedFileInfo
from bugflow_ml.models.line_risk import explain_line, score_lines
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.ml import MLModel
from app.services.line_risk_training_service import TASK

# Same "load once, reload only when the champion changes" cache as
# prediction_service — a fresh clone for training already makes line-risk
# training expensive; scoring shouldn't also reload the model every call.
_cache: dict[str, Any] = {"model_id": None}


@dataclass
class ScoredLine:
    file_path: str
    line_no: int
    code: str
    risk_score: float
    rank: int
    reason: str


def _get_champion(db: Session) -> MLModel | None:
    return db.scalar(select(MLModel).where(MLModel.task == TASK, MLModel.stage == "champion"))


def _load_champion_pipeline(
    db: Session, tracking_uri: str | None = None
) -> tuple[MLModel, Any, Any]:
    champion = _get_champion(db)
    if champion is None:
        raise LookupError("No champion line-risk model has been trained yet.")

    if _cache.get("model_id") == champion.id:
        return champion, _cache["vectorizer"], _cache["model"]

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    pipeline = mlflow.sklearn.load_model(f"runs:/{champion.mlflow_run_id}/line_risk_pipeline")
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["clf"]

    _cache.update(model_id=champion.id, vectorizer=vectorizer, model=model)
    return champion, vectorizer, model


def _format_reason(contributions: list[tuple[str, float]]) -> str:
    return ", ".join(f"{token} ({value:+.2f})" for token, value in contributions)


def score_and_explain_lines(
    db: Session, files: list[ModifiedFileInfo], top_n: int, tracking_uri: str | None = None
) -> list[ScoredLine]:
    """US-13/US-14: ranks every added line across `files` and returns the
    top-N with a token-level reason each. Raises LookupError if no line-risk
    champion has been trained yet — same "no assessment available" signal
    predict_commit uses."""
    _, vectorizer, model = _load_champion_pipeline(db, tracking_uri)

    candidates = [(file.path, line_no, text) for file in files for line_no, text in file.diff_added]
    if not candidates:
        return []

    scores = score_lines(vectorizer, model, [text for _, _, text in candidates])
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])[:top_n]

    return [
        ScoredLine(
            file_path=file_path,
            line_no=line_no,
            code=text,
            risk_score=score,
            rank=rank,
            reason=_format_reason(explain_line(vectorizer, model, text)),
        )
        for rank, ((file_path, line_no, text), score) in enumerate(ranked, start=1)
    ]
