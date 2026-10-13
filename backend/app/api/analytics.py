from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import require_role
from app.models.enums import Role
from app.schemas.analytics import (
    AcceptanceRateRead,
    FairnessRead,
    HotspotsRead,
    RiskTrendsRead,
    TopRiskFactorsRead,
)
from app.services import analytics_service

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    dependencies=[Depends(require_role(Role.MANAGER, Role.QA))],
)


@router.get("/risk-trends", response_model=RiskTrendsRead)
def get_risk_trends(
    days: int = Query(90, ge=1, le=365), db: Session = Depends(get_db)
) -> RiskTrendsRead:
    return analytics_service.get_risk_trends(db, days=days)


@router.get("/acceptance-rate", response_model=AcceptanceRateRead)
def get_acceptance_rate(db: Session = Depends(get_db)) -> AcceptanceRateRead:
    return analytics_service.get_acceptance_rate(db)


@router.get("/hotspots", response_model=HotspotsRead)
def get_hotspots(db: Session = Depends(get_db)) -> HotspotsRead:
    return analytics_service.get_hotspots(db)


@router.get("/top-risk-factors", response_model=TopRiskFactorsRead)
def get_top_risk_factors(db: Session = Depends(get_db)) -> TopRiskFactorsRead:
    return analytics_service.get_top_risk_factors(db)


@router.get("/fairness", response_model=FairnessRead)
def get_fairness(db: Session = Depends(get_db)) -> FairnessRead:
    return analytics_service.get_fairness(db)
