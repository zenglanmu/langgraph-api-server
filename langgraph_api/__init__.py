'''
重新用langchain v1的api设计agents
并用他的stream
目前来说langgraph_api虽然提供了和React的集成，但相当封闭不好hack
而且实现的很神奇，是用python通过grpc端口调用go的langsmith后端，也就是说无法独立运行
开源缺乏auth等模块，且官方只有InMemorySaver， 开源的有一个postgres saver
本质上是想卖企业级的langsmith服务
所以从实现的角度，还是考虑用langserve集成fastapi,或者干脆写fastapi端点
ai请求链路跟踪用langfuse
而前端框架因为是vue,只能仿照官方React的api实现，幸运的是vue3有React hook的类似物
独立成和app并行的目录，且不和appn依赖，方便后面放到别的项目下用
'''
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import APIRouter, FastAPI

from .api import lg_api_router
from .registry import _settings, UserIdCallback, GraphRegistry, get_graph_store, get_graph_checkpointer
from .utils.queue_worker import (
    close_arq_pool,
    close_redis_client,
    setup_database_once,
    start_arq_worker,
    stop_arq_worker,
)
from .services.cron_service import start_cron_scheduler, stop_cron_scheduler


logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lg_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 数据库初始化：多进程部署时通过 Redis 锁保证只执行一次
    await setup_database_once()

    # ARQ worker 与 cron scheduler 以 asyncio task 形式运行在当前进程内
    await start_arq_worker()
    await start_cron_scheduler()

    yield

    await stop_arq_worker()
    await stop_cron_scheduler()
    await close_arq_pool()
    await close_redis_client()


def setup_api(
    *,
    router: APIRouter | FastAPI,
    redis_url: str,
    langgraph_database_uri: str,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_base_url: str | None = None,
    include_router_kwargs: dict | None = None,
    user_id_callback: UserIdCallback | None = None,
    embeding_model_name: str | None = None,
    embeding_dim: int | None = None,
    embeding_base_url: str | None = None,
    embeding_api_key: str | None = None,
):
    _kwargs = include_router_kwargs or {}
    if "prefix" in _kwargs:
        prefix = _kwargs.pop("prefix")
    else:
        prefix = "/langgraph_api"
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
