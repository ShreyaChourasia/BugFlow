"""Entry point for the RQ worker container: mining, PR scoring, and training
jobs (later phases)."""

from rq import SimpleWorker

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.queue import get_queue

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


def main() -> None:
    queue = get_queue()
    logger.info("bugflow_worker_started")
    # SimpleWorker, not RQ's default Worker: the default forks a fresh child
    # process per job, which would silently defeat prediction_service's
    # module-level champion-model cache (US-10: "load the champion once")
    # every single time — found by actually measuring PR-scoring latency
    # under k6, not by unit tests, which run in-process and never exercise
    # RQ's forking at all. Fine to run in-process for one worker replica;
    # ordinary per-job exceptions are still caught (see pr_scoring_service),
    # docker-compose's `restart: unless-stopped` covers the rest.
    SimpleWorker([queue], connection=queue.connection).work()


if __name__ == "__main__":
    main()
