from redis import Redis
from rq import Queue

from app.core.config import get_settings

settings = get_settings()
_redis = Redis.from_url(settings.redis_url)


def get_queue() -> Queue:
    return Queue("default", connection=_redis)


def get_shadow_queue() -> Queue:
    """US-45: challenger shadow-scoring. The worker drains "default" first,
    so shadow jobs never delay a PR check."""
    return Queue("shadow", connection=_redis)
