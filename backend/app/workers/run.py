"""Entry point for the RQ worker container. Jobs (mining, scoring, training) are
added in later phases; this listens on the default queue so `docker compose up`
brings up a working worker from Phase 0 onward."""

from redis import Redis
from rq import Queue, Worker

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


def main() -> None:
    conn = Redis.from_url(settings.redis_url)
    queue = Queue("default", connection=conn)
    logger.info("bugflow_worker_started")
    Worker([queue], connection=conn).work()


if __name__ == "__main__":
    main()
