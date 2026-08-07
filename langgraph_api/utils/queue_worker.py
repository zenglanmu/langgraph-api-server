import os
import signal
import logging
from multiprocessing import get_context
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
import rq
from rq import Queue
from rq.worker_pool import WorkerPool
from rq.cron import CronScheduler
from langgraph_api.registry import _settings


logger = logging.getLogger(__name__)

_redis_client: AsyncRedis | None = None
_sync_redis_client: Redis | None = None

_QUEUE_NAME = "langgragh_api_worker"
_RUN_EVENTS_STREAM_TTL_SECONDS = int(os.getenv("RUN_EVENTS_STREAM_TTL_SECONDS", "7200"))
RUN_EVENTS_STREAM_TTL_SECONDS = _RUN_EVENTS_STREAM_TTL_SECONDS
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


def cleanup_stale_workers() -> None:
    """Kill stale Redis client connections (BLMOVE/SUBSCRIBE) and remove stale
    RQ worker registration keys so that zombie workers from previous runs do
    not steal jobs from the queue.

    Must be called *before* starting the worker pool.
    """
    try:
        conn = Redis.from_url(_settings.redis_url, decode_responses=True)
    except Exception:
        return

    queue_key = f"rq:queue:{_QUEUE_NAME}"

    killed = 0
    try:
        for client in conn.client_list():
            cmd = client.get("cmd", "").lower()
            if cmd in ("blmove", "brpop", "blpop", "bzmpop"):
                try:
                    conn.client_kill(client["addr"])
                    killed += 1
                except Exception:
                    pass
    except Exception:
        pass

    if killed:
        logger.info("cleanup_stale_workers: killed %d stale blocking clients", killed)

    removed = 0
    try:
        for key in conn.scan_iter("rq:worker:*"):
            conn.delete(key)
            removed += 1
    except Exception:
        pass

    if removed:
        logger.info("cleanup_stale_workers: removed %d stale worker keys", removed)

    try:
        conn.delete(f"rq:scheduler-lock:{_QUEUE_NAME}")
        conn.srem("rq:queues", queue_key)
        conn.delete(queue_key)
    except Exception:
        pass

    try:
        conn.close()
    except Exception:
        pass


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
        print("\n[*] langgraph api worker starting...")
        pool.start(burst=False, logging_level=LOG_LEVEL)        
    except KeyboardInterrupt:
        print("\n[*] langgraph api worker stopping...")


def _worker_process_target(settings_data: dict):
    _settings.load(settings_data)
    cleanup_stale_workers()
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


def _setsid_wrapper(target, settings_data: dict):
    """Module-level wrapper that creates a new session before running target.

    Placed at module scope so it can be pickled by the 'spawn' start method.
    """
    try:
        os.setsid()
    except OSError:
        pass
    target(settings_data)


def _spawn_background_process(target, settings_data: dict) -> int:
    """Use spawn instead of fork to avoid inheriting broken asyncio/DB state.

    The child creates a new session (setsid) so that we can later kill the
    entire process tree (worker pool -> forked workers -> scheduler) via
    os.killpg on shutdown.
    """
    ctx = get_context("spawn")
    p = ctx.Process(
        target=_setsid_wrapper,
        args=(target, settings_data),
    )
    p.start()
    pid = p.pid
    if pid is None:
        raise RuntimeError("Failed to spawn background process")
    return pid


def backgroud_worker_pool() -> int:
    settings_data = _settings.snapshot()
    return _spawn_background_process(_worker_process_target, settings_data)


def backgroud_cron() -> int:
    settings_data = _settings.snapshot()
    return _spawn_background_process(_cron_process_target, settings_data)


def kill_background_process(pid: int) -> None:
    """Kill a background process and its entire process group.

    The child was started with setsid(), so it is a session/group leader.
    Killing the process group ensures that forked workers and scheduler
    processes are also terminated, preventing zombie workers from stealing
    jobs on the next run.
    """
    if pid is None:
        return
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    else:
        import time
        for _ in range(10):
            try:
                os.killpg(pgid, 0)
            except (OSError, ProcessLookupError):
                break
            time.sleep(0.1)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


async def close_redis_client() -> None:
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None
