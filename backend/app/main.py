from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.defect_reports import router as defect_reports_router
from app.api.health import router as health_router
from app.api.mining import router as mining_router
from app.api.pull_requests import line_risk_router
from app.api.pull_requests import router as pull_requests_router
from app.api.repositories import router as repositories_router
from app.api.risk import router as risk_router
from app.api.users import router as users_router
from app.api.webhooks import router as webhooks_router
from app.core import native_libs  # noqa: F401 -- must load before any request imports torch
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("bugflow_api_started", env=settings.api_env)
    yield


app = FastAPI(title="BugFlow API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(repositories_router)
app.include_router(mining_router)
app.include_router(pull_requests_router)
app.include_router(line_risk_router)
app.include_router(risk_router)
app.include_router(webhooks_router)
app.include_router(admin_router)
app.include_router(defect_reports_router)
