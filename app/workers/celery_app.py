"""Celery application configuration."""
from celery import Celery

from app.config import settings

# Create Celery app
celery = Celery(
    "hr_ats",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"]
)

# Configure Celery
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)

# Configure task routes
celery.conf.task_routes = {
    "ingest_file": {"queue": "ingest"},
    "rank_role": {"queue": "ranking"},
    "train_reranker": {"queue": "ml"},
}

if __name__ == "__main__":
    celery.start()
