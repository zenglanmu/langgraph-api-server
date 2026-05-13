'''
重新用langchain v1的api设计agents
并用他的stream
目前来说langgraph_api虽然提供了和React的集成，但相当封闭不好hack
而且实现的很神奇，是用python通过grpc端口调用go的langsmith后端，也就是说无法独立运行
开源的缺乏auth等模块，且官方只有InMemorySaver， 开源的有一个postgres saver
本质上是想卖企业级的langsmith服务
所以从实现的角度，还是考虑用langserve集成fastapi,或者干脆写fastapi端点
ai请求链路跟踪用langfuse
而前端框架因为是vue,只能仿照官方React的api实现，幸运的是vue3有React hook的类似物
独立成和app并行的目录，且不和appn依赖，方便后面放到别的项目下用
'''
import os
import signal
import errno
import logging
from contextlib import asynccontextmanager
from typing import Callable, Awaitable, AsyncIterator
from fastapi import APIRouter, FastAPI


from .api import runs, threads, assistants, store, crons
from .registry import _settings, UserIdCallback, GraphRegistry, get_graph_store, get_graph_checkpointer
from .persistants import setup
from .utils.queue_worker import backgroud_worker_pool, backgroud_cron, get_redis_client


logger = logging.getLogger(__name__)

lg_api_router = APIRouter()
lg_api_router.include_router(runs.router)
lg_api_router.include_router(threads.router)
lg_api_router.include_router(assistants.router)
lg_api_router.include_router(store.router)
lg_api_router.include_router(crons.router)

_WORKER_PID: int | None = None
_CRON_PID: int | None = None
_IS_LOCK_OWNER: bool = False

_STARTUP_LOCK_KEY = "langgraph_api:bg_startup_lock"
_STARTUP_LOCK_TTL = 60


@asynccontextmanager
async def _lg_lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _WORKER_PID, _CRON_PID, _IS_LOCK_OWNER

    redis = await get_redis_client()
    acquired = await redis.set(
        _STARTUP_LOCK_KEY, str(os.getpid()), nx=True, ex=_STARTUP_LOCK_TTL
    )

    if acquired:
        _IS_LOCK_OWNER = True
        logger.info("langgraph_api: acquired startup lock (pid=%s), running setup & background processes", os.getpid())
        await setup()
        _WORKER_PID = backgroud_worker_pool()
        _CRON_PID = backgroud_cron()
    else:
        _IS_LOCK_OWNER = False
        logger.info("langgraph_api: startup lock held by another process (pid=%s), skipping setup & background processes",
                     await redis.get(_STARTUP_LOCK_KEY))

    yield

    if _IS_LOCK_OWNER:
        for pid in (_WORKER_PID, _CRON_PID):
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError as e:
                    if e.errno != errno.ESRCH:
                        raise
        await redis.delete(_STARTUP_LOCK_KEY)
        _IS_LOCK_OWNER = False


def setup_api(
    *,
    router: APIRouter | FastAPI,
    redis_url: str,
    langgraph_database_uri: str,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_base_url: str | None = None,
    include_router_kwargs: dict = None,
    user_id_callback: UserIdCallback = None,
    embeding_model_name: str | None = None,
    embeding_dim: int | None = None,
    embeding_base_url: str | None = None,
    embeding_api_key: str | None = None,
) -> AsyncIterator[None]:
    _kwargs = include_router_kwargs if include_router_kwargs else {}
    if "prefix" in _kwargs:
        prefix = _kwargs.pop("prefix")
    else:
        prefix = "/langgragh_api"
    router.include_router(lg_api_router, prefix=prefix, **_kwargs)

    _settings.configure(
        redis_url=redis_url,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_base_url=langfuse_base_url,
        langgraph_database_uri=langgraph_database_uri,
        user_id_callback=user_id_callback,
        embeding_model_name=embeding_model_name,
        embeding_dim=embeding_dim,
        embeding_base_url=embeding_base_url,
        embeding_api_key=embeding_api_key,
    )

    return _lg_lifespan


__all__ = ["setup_api", "GraphRegistry", "get_graph_store", "get_graph_checkpointer"]