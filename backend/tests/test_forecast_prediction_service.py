from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.forecast_prediction_service import get_forecast
from app.services.forecast_training_service import train_and_register_champion


@pytest.fixture(autouse=True)
def _clean(db_session: Session) -> None:
    db_session.execute(text("TRUNCATE defect_reports CASCADE"))


@pytest.fixture
def champion_tracking_uri(db_session: Session, tmp_path: Path) -> str:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)
    return tracking_uri


@pytest.mark.story("US-31")
def test_get_forecast_returns_none_when_no_champion_exists(db_session: Session) -> None:
    prediction = get_forecast(
        db_session, "blocker", "P1", "login", tracking_uri="sqlite:///:memory:"
    )

    assert prediction is None


@pytest.mark.story("US-31")
@pytest.mark.story("US-32")
def test_get_forecast_returns_a_curve_with_median_and_p90(
    db_session: Session, champion_tracking_uri: str
) -> None:
    prediction = get_forecast(db_session, "blocker", "P1", "login", champion_tracking_uri)

    assert prediction is not None
    assert prediction.median_days > 0
    assert prediction.p90_days >= prediction.median_days
    assert len(prediction.curve) > 0
