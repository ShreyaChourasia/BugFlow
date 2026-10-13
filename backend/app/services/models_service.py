from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from app.models.ml import MLModel
from app.models.pull_request import RiskPrediction
from app.schemas.models import ShadowComparisonRead
from app.services.training_service import TASK as COMMIT_RISK_TASK


def list_models(db: Session, task: str | None = None) -> list[MLModel]:
    """US-41/43/45: every trained model's version/stage/metrics, for the
    ML Engineer's Models page."""
    query = select(MLModel).order_by(MLModel.task, MLModel.trained_at.desc())
    if task is not None:
        query = query.where(MLModel.task == task)
    return list(db.scalars(query))


def rollback(db: Session, model_id: int) -> MLModel:
    """US-43: promotes any previous version back to champion, archiving
    whatever is currently champion for that task — the exact same
    promote/demote pair training already does, just triggered by a human
    picking an older version instead of a new training run winning on
    its metric."""
    target = db.get(MLModel, model_id)
    if target is None:
        raise ValueError(f"No model with id {model_id}")

    current_champion = db.scalar(
        select(MLModel).where(MLModel.task == target.task, MLModel.stage == "champion")
    )
    if current_champion is not None and current_champion.id != target.id:
        current_champion.stage = "archived"
    target.stage = "champion"
    db.commit()
    db.refresh(target)
    return target


def get_shadow_comparison(db: Session) -> ShadowComparisonRead:
    """US-45: the commit-risk challenger's shadow scores against the
    champion's real scores on the exact same commits (see docs/decisions/008
    for why this is scoped to commit-risk only). Self-joins RiskPrediction
    on commit_id: one side is the champion's real (non-shadow) row, the
    other is the challenger's shadow row for that same commit."""
    champion = db.scalar(
        select(MLModel).where(MLModel.task == COMMIT_RISK_TASK, MLModel.stage == "champion")
    )
    challenger = db.scalar(
        select(MLModel)
        .where(MLModel.task == COMMIT_RISK_TASK, MLModel.stage == "challenger")
        .order_by(MLModel.trained_at.desc())
    )
    if champion is None or challenger is None:
        return ShadowComparisonRead(status="unavailable")

    champ_row = aliased(RiskPrediction)
    shadow_row = aliased(RiskPrediction)
    pairs = db.execute(
        select(
            champ_row.calibrated_probability,
            champ_row.risk_level,
            shadow_row.calibrated_probability,
            shadow_row.risk_level,
        )
        .join(
            shadow_row,
            (shadow_row.commit_id == champ_row.commit_id)
            & (shadow_row.is_shadow.is_(True))
            & (shadow_row.model_version == challenger.version),
        )
        .where(champ_row.is_shadow.is_(False), champ_row.model_version == champion.version)
    ).all()

    if not pairs:
        return ShadowComparisonRead(
            status="unavailable",
            champion_version=champion.version,
            challenger_version=challenger.version,
        )

    n = len(pairs)
    abs_diffs = [abs(c - s) for c, _, s, _ in pairs]
    agreements = sum(1 for _, c_level, _, s_level in pairs if c_level == s_level)
    return ShadowComparisonRead(
        status="available",
        champion_version=champion.version,
        challenger_version=challenger.version,
        n_compared=n,
        mean_absolute_difference=sum(abs_diffs) / n,
        risk_level_agreement_rate=agreements / n,
        mean_champion_probability=sum(c for c, _, _, _ in pairs) / n,
        mean_challenger_probability=sum(s for _, _, s, _ in pairs) / n,
    )
