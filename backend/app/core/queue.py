from redis import Redis
from rq import Queue

from app.core.config import get_settings

settings = get_settings()
_redis = Redis.from_url(settings.redis_url)


def get_queue() -> Queue:
    return Queue("default", connection=_redis)
