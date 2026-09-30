from pathlib import Path

import pytest
from bugflow_ml.mining.git_miner import ModifiedFileInfo
from sqlalchemy.orm import Session

from app.models.repository import Repository
from app.services.line_prediction_service import score_and_explain_lines
from app.services.line_risk_training_service import train_and_register_champion


@pytest.fixture
def champion_tracking_uri(
    db_session: Session, repository_with_commits: Repository, tmp_path: Path
) -> str:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)
    return tracking_uri


@pytest.mark.story("US-13")
@pytest.mark.story("US-14")
def test_score_and_explain_lines_ranks_risky_lines_first(
    db_session: Session, champion_tracking_uri: str
) -> None:
    files = [
        ModifiedFileInfo(
            path="a.py",
            added_lines=2,
            deleted_lines=0,
            diff_added=[(1, "return result"), (2, "except: pass")],
        )
    ]

    scored = score_and_explain_lines(db_session, files, top_n=2, tracking_uri=champion_tracking_uri)

    assert len(scored) == 2
    assert scored[0].rank == 1
    assert scored[0].code == "except: pass"
    assert scored[0].risk_score > scored[1].risk_score
    assert scored[0].reason


@pytest.mark.story("US-13")
def test_score_and_explain_lines_respects_top_n(
    db_session: Session, champion_tracking_uri: str
) -> None:
    files = [
        ModifiedFileInfo(
            path="a.py",
            added_lines=3,
            deleted_lines=0,
            diff_added=[(1, "return result"), (2, "except: pass"), (3, "except: pass")],
        )
    ]

    scored = score_and_explain_lines(db_session, files, top_n=1, tracking_uri=champion_tracking_uri)

    assert len(scored) == 1


def test_score_and_explain_lines_with_no_champion_raises(db_session: Session) -> None:
    with pytest.raises(LookupError):
        score_and_explain_lines(db_session, [], top_n=5, tracking_uri="sqlite:///:memory:")
