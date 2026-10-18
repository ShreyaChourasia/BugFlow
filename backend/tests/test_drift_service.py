import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.commit import Commit
from app.services.drift_service import run_drift_check
from app.services.training_service import train_and_register_champion

from .conftest import make_commits_for_training


@pytest.mark.story("US-41")
def test_drift_check_unavailable_without_a_champion(db_session: Session) -> None:
    result = run_drift_check(db_session, tracking_uri="sqlite:///:memory:")
    assert result.status == "unavailable"


@pytest.mark.story("US-41")
def test_drift_check_flags_a_genuinely_shifted_feature(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_commits = make_commits_for_training(db_session, n=150)
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    # New commits, mined after training, with churn far outside anything
    # the champion ever saw (reference range was 1-200) — an almost
    # non-overlapping shift (800-1000), so it should come back "severe",
    # not just barely over the "moderate" threshold.
    rng = random.Random(99)
    repo_id = training_commits[0].repository_id
    base = datetime(2025, 1, 1, tzinfo=UTC)
    for i in range(40):
        churn = rng.randint(800, 1000)
        files_changed = rng.randint(1, 10)
        db_session.add(
            Commit(
                sha=f"drifted-{i}",
                repository_id=repo_id,
                message=f"drifted commit {i}",
                timestamp=base + timedelta(hours=i),
                lines_added=churn // 2,
                lines_deleted=churn // 2,
                files_changed=files_changed,
                features={
                    "lines_added": churn // 2,
                    "lines_deleted": churn // 2,
                    "churn": churn,
                    "files_changed": files_changed,
                    "directories_touched": rng.randint(1, 5),
                    "subsystems_touched": rng.randint(1, 3),
                    "entropy": rng.random() * 2,
                    "author_prior_commits": i,
                    "is_fix": False,
                },
            )
        )
    db_session.commit()

    result = run_drift_check(db_session, tracking_uri=tracking_uri)

    assert result.status == "available"
    assert result.checked_features > 0
    alerts_by_feature = {a.feature: a for a in result.alerts}
    # churn's reference (1-200) and current (800-1000) ranges barely
    # overlap at all — an unambiguous shift, not noise, so it should always
    # land in the top severity bucket regardless of which other features a
    # small current sample happens to nudge past the threshold by chance.
    assert alerts_by_feature["churn"].severity == "severe"

    # §8: the threshold is configurable (DRIFT_PSI_THRESHOLD) — raise it far
    # past any real PSI and the same shift no longer raises an alert.
    monkeypatch.setattr(get_settings(), "drift_psi_threshold", 1e6)
    assert run_drift_check(db_session, tracking_uri=tracking_uri).alerts == []
