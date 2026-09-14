'''
后台任务基础设施（基于 ARQ 的协程方式）。

ARQ worker 与 cron scheduler 均以 asyncio task 的形式运行在当前进程内：
- 无需 spawn 子进程，无需 kill 进程组，无需清理残留 worker / 陈旧 Redis 连接
- 多进程部署（如 uvicorn --workers）时每个进程各跑一份：
  - 队列消费由 Redis 队列语义保证每个 job 只被一个进程执行
  - cron 触发通过 Redis SETNX 去重保证全局只执行一次（见 cron_service）
- 数据库 setup() 通过 Redis 锁保证只执行一次
'''
import asyncio
import logging
import os

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.worker import Worker
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from ..registry import _settings


logger = logging.getLogger(__name__)

_redis_client: AsyncRedis | None = None
_arq_pool: ArqRedis | None = None
_worker_task: asyncio.Task | None = None
_current_worker: Worker | None = None

_QUEUE_NAME = "langgraph_api_worker"

RUN_EVENTS_STREAM_TTL_SECONDS = int(os.getenv("RUN_EVENTS_STREAM_TTL_SECONDS", "7200"))

_SETUP_LOCK_KEY = "langgraph_api:bg_setup_lock"
_SETUP_LOCK_TTL_SECONDS = int(os.getenv("LANGGRAPH_SETUP_LOCK_TTL_SECONDS", "300"))
_SETUP_DONE_TTL_SECONDS = int(os.getenv("LANGGRAPH_SETUP_DONE_TTL_SECONDS", "86400"))


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


async def close_redis_client() -> None:
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None


async def get_arq_pool() -> ArqRedis:
    '''获取用于 enqueue job 的 ARQ redis pool'''
    global _arq_pool
    if _arq_pool is not None:
        return _arq_pool
    _arq_pool = await create_pool(
        RedisSettings.from_dsn(_settings.redis_url),
        default_queue_name=_QUEUE_NAME,
    )
    return _arq_pool


async def close_arq_pool() -> None:
    global _arq_pool
    if _arq_pool is not None:
        try:
            await _arq_pool.close(close_connection_pool=True)
        except Exception:
            pass
        _arq_pool = None


# ── Database setup (run once across all processes) ──────────────────────

async def setup_database_once() -> None:
    '''数据库初始化，通过 Redis 锁保证多进程部署时只执行一次。

    - 抢到锁的进程执行 setup()，完成后将锁标记为 done（带 TTL）
    - 未抢到锁的进程轮询等待 done 状态
    - 持锁进程崩溃时锁自动过期，其他进程重新抢锁执行
    '''
    from ..persistants import setup

    redis = await get_redis_client()
    while True:
        if await redis.set(_SETUP_LOCK_KEY, "running", nx=True, ex=_SETUP_LOCK_TTL_SECONDS):
            try:
                await setup()
            except Exception:
                await redis.delete(_SETUP_LOCK_KEY)
                raise
            await redis.set(_SETUP_LOCK_KEY, "done", ex=_SETUP_DONE_TTL_SECONDS)
            return

        # 未抢到锁：等待持锁进程完成 setup
        for _ in range(_SETUP_LOCK_TTL_SECONDS * 2):
            state = await redis.get(_SETUP_LOCK_KEY)
            if state == "done":
                return
            if state is None:
                break  # 持锁进程已退出（崩溃），重新抢锁
            await asyncio.sleep(0.5)


# ── In-process ARQ worker ───────────────────────────────────────────────

def _build_arq_worker() -> Worker:
    from ..services.run_queue_service import run_graph_job

    return Worker(
        [run_graph_job],
        queue_name=_QUEUE_NAME,
        redis_settings=RedisSettings.from_dsn(_settings.redis_url),
        max_jobs=int(os.getenv("RUN_EVENTS_STREAM_NUM_WORKERS", "8")),
        job_timeout=RUN_EVENTS_STREAM_TTL_SECONDS,
        keep_result=0,
        max_tries=1,
        handle_signals=False,
    )


async def _arq_worker_loop() -> None:
    while True:
        global _current_worker
        worker = _build_arq_worker()
        _current_worker = worker
        try:
            await worker.async_run()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("arq worker crashed, restarting in 5 seconds")
            try:
                if worker._pool is not None:
                    await worker.close()
            except Exception:
                pass
            await asyncio.sleep(5)


async def start_arq_worker() -> None:
    '''在当前进程的事件循环中启动 ARQ worker（asyncio task）'''
    global _worker_task
    if _worker_task is not None:
        return
    logger.info("[*] langgraph api arq worker starting...")
    _worker_task = asyncio.create_task(_arq_worker_loop())


async def stop_arq_worker() -> None:
    global _worker_task, _current_worker
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    _worker_task = None

    worker, _current_worker = _current_worker, None
    if worker is not None and worker._pool is not None:
        try:
            await worker.close()
        except Exception:
            pass
