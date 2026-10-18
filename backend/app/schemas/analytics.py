from pydantic import BaseModel


class RiskTrendPoint(BaseModel):
    week_start: str
    mean_calibrated_probability: float
    count: int


class RiskTrendsRead(BaseModel):
    points: list[RiskTrendPoint]


class AcceptanceRateEntry(BaseModel):
    decision_type: str
    total: int
    accepted: int
    acceptance_rate: float


class AcceptanceRateRead(BaseModel):
    entries: list[AcceptanceRateEntry]
    overall_rate: float | None


class HotspotEntry(BaseModel):
    component: str
    defect_count: int


class HotspotsRead(BaseModel):
    entries: list[HotspotEntry]


class TopRiskFactorEntry(BaseModel):
    feature: str
    occurrence_count: int
    mean_shap_value: float


class TopRiskFactorsRead(BaseModel):
    entries: list[TopRiskFactorEntry]


class FairnessRead(BaseModel):
    status: str  # "available" | "unavailable"
    junior_mean_risk: float | None = None
    senior_mean_risk: float | None = None
    junior_count: int = 0
    senior_count: int = 0
    disparity: float | None = None
    junior_fpr: float | None = None
    senior_fpr: float | None = None
    fpr_disparity: float | None = None
    model_version: str | None = None
