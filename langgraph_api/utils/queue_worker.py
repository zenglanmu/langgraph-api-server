import os
from multiprocessing import Process
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
import rq
from rq import Queue
from rq.worker_pool import WorkerPool
from rq.cron import CronScheduler
from langgraph_api.registry import _settings


_redis_client: AsyncRedis | None = None
_sync_redis_client: Redis | None = None

_QUEUE_NAME = "langgragh_api_worker"
RUN_EVENTS_STREAM_TTL_SECONDS = int(os.getenv("RUN_EVENTS_STREAM_TTL_SECONDS", "7200"))
RUN_EVENTS_STREAM_NUM_WORKERS = int(os.getenv("RUN_EVENTS_STREAM_NUM_WORKERS", "8"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_rq_queue: Queue | None = None
_cron_scheduler: CronScheduler | None = None


async def get_redis_client() -> AsyncRedis:
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    redis = AsyncRedis.from_url(_settings.redis_url, decode_responses=True)
    try:
        await redis.ping()
    except Exception as e:
        try:
            await redis.aclose()
        except Exception:
            pass
        raise RuntimeError(f"Redis connection failed ({_settings.redis_url}): {e}") from e

    _redis_client = redis
    return _redis_client


def get_sync_redis_client() -> Redis:
    global _sync_redis_client
    if _sync_redis_client is not None:
        return _sync_redis_client
    _sync_redis_client = Redis.from_url(_settings.redis_url, decode_responses=True)
    return _sync_redis_client


def get_rq_queue() -> Queue:
    global _rq_queue
    if _rq_queue is not None:
        return _rq_queue

    _rq_queue = Queue(
        _QUEUE_NAME,
        connection=Redis.from_url(_settings.redis_url),
        default_timeout=RUN_EVENTS_STREAM_TTL_SECONDS,
    )
    return _rq_queue


def get_cron_scheduler() -> CronScheduler:
    global _cron_scheduler
    if _cron_scheduler is not None:
        return _cron_scheduler

    queue = get_rq_queue()
    _cron_scheduler = CronScheduler(
        connection=queue.connection,
        logging_level=LOG_LEVEL,
        name="langgraph_api_cron",
    )
    return _cron_scheduler


def _start_worker_pool():
    serializer = rq.serializers.DefaultSerializer
    worker_class = rq.worker.Worker
    job_class = rq.job.Job

    queue_names = [_QUEUE_NAME]
    queue = get_rq_queue()

    pool = WorkerPool(
        queue_names,
        connection=queue.connection,
        num_workers=RUN_EVENTS_STREAM_NUM_WORKERS,
        serializer=serializer,
        worker_class=worker_class,
        job_class=job_class,
        with_scheduler=True,
    )
    try:
        pool.start(burst=False, logging_level=LOG_LEVEL)
        print("\n[*] langgraph api worker start...")
    except KeyboardInterrupt:
        print("\n[*] langgraph api worker stopping...")


def _worker_process_target(settings_data: dict):
    _settings.load(settings_data)
    _start_worker_pool()


def _cron_process_target(settings_data: dict):
    _settings.load(settings_data)

    from ..services.cron_service import sync_crons_to_rq_scheduler, listen_cron_sync_events
    
    # TODO, 是否支持scheduler运行过程中动态刷新
    sync_crons_to_rq_scheduler()

    import threading
    cron_listener_thread = threading.Thread(
        target=listen_cron_sync_events,
        daemon=True,
        name="cron-sync-listener",
    )
    cron_listener_thread.start()

    scheduler = get_cron_scheduler()
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[*] langgraph api cron scheduler stopping...")


def backgroud_worker_pool() -> int:
    settings_data = _settings.snapshot()
    p = Process(target=_worker_process_target, args=(settings_data,))
    p.start()
    return p.pid


def backgroud_cron() -> int:
    settings_data = _settings.snapshot()
    p = Process(target=_cron_process_target, args=(settings_data,))
    p.start()
    return p.pid
