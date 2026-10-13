import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.api.models as models_module
from app.models.enums import Role
from app.models.ml import MLModel
from app.models.pull_request import RiskPrediction

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple] = []

    def enqueue(self, func, *args, **kwargs) -> None:
        self.enqueued.append((func, args, kwargs))


@pytest.fixture
def fake_queue(monkeypatch: pytest.MonkeyPatch) -> _FakeQueue:
    queue = _FakeQueue()
    monkeypatch.setattr(models_module, "get_queue", lambda: queue)
    return queue


@pytest.fixture(autouse=True)
def _clean(db_session: Session) -> None:
    db_session.execute(text("TRUNCATE risk_predictions, drift_alerts, ml_models, commits CASCADE"))


def _ml_engineer_headers(client: TestClient, db_session: Session, email: str) -> dict[str, str]:
    make_user(db_session, email, Role.ML_ENGINEER, password="s3cret")
    return _login(client, email, "s3cret")


@pytest.mark.story("US-43")
def test_models_requires_ml_engineer_role(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "reporter-m1@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-m1@example.com", "s3cret")

    response = client.get("/models", headers=headers)

    assert response.status_code == 403


@pytest.mark.story("US-43")
def test_list_models_filters_by_task(client: TestClient, db_session: Session) -> None:
    headers = _ml_engineer_headers(client, db_session, "mleng-m1@example.com")
    db_session.add_all(
        [
            MLModel(
                task="commit_risk",
                version="r1",
                mlflow_run_id="r1",
                stage="champion",
                metrics={},
                data_version="d1",
                seed=42,
            ),
            MLModel(
                task="line_risk",
                version="r2",
                mlflow_run_id="r2",
                stage="champion",
                metrics={},
                data_version="d1",
                seed=42,
            ),
        ]
    )
    db_session.commit()

    response = client.get("/models", headers=headers, params={"task": "commit_risk"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["task"] == "commit_risk"


@pytest.mark.story("US-41")
@pytest.mark.story("US-43")
@pytest.mark.story("US-45")
def test_retrain_enqueues_a_background_job(
    client: TestClient, db_session: Session, fake_queue: _FakeQueue
) -> None:
    headers = _ml_engineer_headers(client, db_session, "mleng-m2@example.com")

    response = client.post(
        "/models/retrain", headers=headers, json={"group": "commit_risk", "seed": 7}
    )

    assert response.status_code == 202
    assert response.json() == {"status": "queued", "group": "commit_risk"}
    assert len(fake_queue.enqueued) == 1
    _func, args, _kwargs = fake_queue.enqueued[0]
    assert args == ("commit_risk", 7)


@pytest.mark.story("US-43")
def test_rollback_promotes_a_previous_version_and_archives_the_champion(
    client: TestClient, db_session: Session
) -> None:
    headers = _ml_engineer_headers(client, db_session, "mleng-m3@example.com")
    old = MLModel(
        task="commit_risk",
        version="old",
        mlflow_run_id="old",
        stage="archived",
        metrics={},
        data_version="d1",
        seed=42,
    )
    current = MLModel(
        task="commit_risk",
        version="new",
        mlflow_run_id="new",
        stage="champion",
        metrics={},
        data_version="d2",
        seed=42,
    )
    db_session.add_all([old, current])
    db_session.commit()

    response = client.post(f"/models/{old.id}/rollback", headers=headers)

    assert response.status_code == 200
    assert response.json()["stage"] == "champion"
    db_session.refresh(current)
    assert current.stage == "archived"


@pytest.mark.story("US-43")
def test_rollback_404s_for_an_unknown_model(client: TestClient, db_session: Session) -> None:
    headers = _ml_engineer_headers(client, db_session, "mleng-m4@example.com")

    response = client.post("/models/999999/rollback", headers=headers)

    assert response.status_code == 404


@pytest.mark.story("US-45")
def test_shadow_comparison_unavailable_without_a_challenger(
    client: TestClient, db_session: Session
) -> None:
    headers = _ml_engineer_headers(client, db_session, "mleng-m5@example.com")
    db_session.add(
        MLModel(
            task="commit_risk",
            version="champ",
            mlflow_run_id="champ",
            stage="champion",
            metrics={},
            data_version="d1",
            seed=42,
        )
    )
    db_session.commit()

    response = client.get("/models/shadow-comparison", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


@pytest.mark.story("US-45")
def test_shadow_comparison_compares_paired_predictions(
    client: TestClient, db_session: Session
) -> None:
    from datetime import UTC, datetime

    from app.models.commit import Commit
    from app.models.repository import Repository

    headers = _ml_engineer_headers(client, db_session, "mleng-m6@example.com")
    repo = Repository(name="demo", url="/tmp/x")
    db_session.add(repo)
    db_session.commit()
    commit = Commit(
        sha="s1", repository_id=repo.id, message="x", timestamp=datetime.now(UTC), features={}
    )
    db_session.add(commit)
    db_session.add_all(
        [
            MLModel(
                task="commit_risk",
                version="champ",
                mlflow_run_id="champ",
                stage="champion",
                metrics={},
                data_version="d1",
                seed=42,
            ),
            MLModel(
                task="commit_risk",
                version="chal",
                mlflow_run_id="chal",
                stage="challenger",
                metrics={},
                data_version="d1",
                seed=42,
            ),
        ]
    )
    db_session.commit()
    db_session.add_all(
        [
            RiskPrediction(
                commit_id=commit.id,
                model_version="champ",
                probability=0.4,
                calibrated_probability=0.4,
                confidence=0.2,
                risk_level="low",
                latency_ms=1.0,
            ),
            RiskPrediction(
                commit_id=commit.id,
                model_version="chal",
                probability=0.6,
                calibrated_probability=0.6,
                confidence=0.2,
                risk_level="medium",
                latency_ms=1.0,
                is_shadow=True,
            ),
        ]
    )
    db_session.commit()

    response = client.get("/models/shadow-comparison", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["n_compared"] == 1
    assert body["mean_absolute_difference"] == pytest.approx(0.2)
    assert body["risk_level_agreement_rate"] == pytest.approx(0.0)


@pytest.mark.story("US-41")
def test_drift_check_unavailable_without_a_champion(
    client: TestClient, db_session: Session
) -> None:
    headers = _ml_engineer_headers(client, db_session, "mleng-m7@example.com")

    response = client.post("/models/drift-check", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


@pytest.mark.story("US-41")
def test_drift_alerts_lists_persisted_alerts(client: TestClient, db_session: Session) -> None:
    headers = _ml_engineer_headers(client, db_session, "mleng-m8@example.com")
    from app.models.ml import DriftAlert

    model = MLModel(
        task="commit_risk",
        version="v1",
        mlflow_run_id="v1",
        stage="champion",
        metrics={},
        data_version="d1",
        seed=42,
    )
    db_session.add(model)
    db_session.commit()
    db_session.add(DriftAlert(model_id=model.id, feature="churn", psi=0.4, severity="moderate"))
    db_session.commit()

    response = client.get("/models/drift-alerts", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["feature"] == "churn"
