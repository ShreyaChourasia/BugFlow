"""Entry point for the RQ worker container: mining jobs today, scoring and
training jobs added in later phases."""

from rq import Worker

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.queue import get_queue

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


def main() -> None:
    queue = get_queue()
    logger.info("bugflow_worker_started")
    Worker([queue], connection=queue.connection).work()


if __name__ == "__main__":
    main()
