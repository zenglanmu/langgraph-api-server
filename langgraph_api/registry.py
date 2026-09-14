import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import os
from inspect import isawaitable
from typing import AsyncIterator, Awaitable, Callable
from langchain.embeddings import init_embeddings
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres.base import PostgresIndexConfig, ANNIndexConfig
from langgraph.checkpoint.postgres.aio import _ainternal
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from .persistants.thread import AsyncPostgresThread
from .persistants.cron import AsyncPostgresCron
from .persistants.assistant import AsyncPostgresAssistant

'''
return user id key, to sperate user thread
'''
UserIdCallback = Callable[[], str | int | None | Awaitable[str | int | None]]

'''
worker 与 API 运行在同一进程内（asyncio task），
graph builder 在注册表中直接以函数形式保存，无需序列化，见:
https://github.com/langchain-ai/langgraph/issues/3289
'''
CompileGraphCallback = Callable[[], CompiledStateGraph]


class ApiGlobalSettings:
    '''run time config'''
    __slots__ = [
        'graph_registry', 'redis_url',
        'langfuse_public_key', 'langfuse_secret_key', 'langfuse_base_url',
        'langgraph_database_uri', 'user_id_callback', 'embeding_model_name',
        'embeding_dim', 'embeding_base_url', 'embeding_api_key'
    ]
    
    def __init__(self):
        self.graph_registry: dict[str, CompileGraphCallback] = {}
        self.langfuse_public_key: str | None = None
        self.langfuse_secret_key: str | None = None
        self.langfuse_base_url: str | None = None
        self.langgraph_database_uri: str | None = None
        self.user_id_callback: UserIdCallback | None = None
        self.redis_url: str | None = None
        
        '''向量化相关，embeding_model_name例子openai:Qwen/Qwen3-VL-Embedding-2B，openai表示协议'''
        self.embeding_model_name: str | None = None
        self.embeding_dim: int | None = None
        self.embeding_base_url: str | None = None
        self.embeding_api_key: str | None = None

    def configure(
        self,
        *,
        redis_url: str,
        langgraph_database_uri: str,
        langfuse_public_key: str | None = None,
        langfuse_secret_key: str | None = None,
        langfuse_base_url: str | None = None,
        user_id_callback: UserIdCallback | None = None,
        embeding_model_name: str | None = None,
        embeding_dim: int | None = None,
        embeding_base_url: str | None = None,
        embeding_api_key: str | None = None,
    ):
        '''设置运行时配置，user_id_callback 可能依赖 fastapi 请求上下文，在 worker task 中调用时不一定可用'''
        self.redis_url = redis_url
        self.langfuse_public_key = langfuse_public_key
        self.langfuse_secret_key = langfuse_secret_key
        self.langfuse_base_url = langfuse_base_url
        self._setup_langfuse_env()
        self.langgraph_database_uri = langgraph_database_uri
        self.user_id_callback = user_id_callback
        self.embeding_model_name = embeding_model_name
        self.embeding_dim = embeding_dim
        self.embeding_base_url = embeding_base_url
        self.embeding_api_key = embeding_api_key

    @property
    def langfuse_enabled(self)->bool:
        return self.langfuse_public_key and self.langfuse_secret_key 
    
    @property
    def embeding_enabled(self)->bool:
        return self.embeding_model_name and self.embeding_dim
    
    def _setup_langfuse_env(self):
        '''
        langfuse using env variables for config
        '''        
        if self.langfuse_enabled:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_secret_key
        if self.langfuse_base_url:
            os.environ["LANGFUSE_HOST"] = self.langfuse_base_url


# Singleton config
_settings = ApiGlobalSettings()

class GraphRegistry:  
    @classmethod
    async def count(cls):
        return len(_settings.graph_registry)
    
    @classmethod  
    def registy_lg_graph(cls, name: str, lg_runnable: CompileGraphCallback):
        '''
        narrow down to graph, not more broad runnable
        '''
        
        if name in _settings.graph_registry:
            raise RuntimeError(f'duplicate runnable name {name}')
        _settings.graph_registry[name] = lg_runnable
    
    @classmethod 
    def get_lg_graph(cls, name: str)->CompiledStateGraph:
        if name not in _settings.graph_registry:
            raise RuntimeError(f'runnable name {name} not exists')
        build_graph_func = _settings.graph_registry[name]
        agent = build_graph_func()
        return agent


async def get_user_id() -> str | int | None:
    '''调用 user_id_callback，同步/异步回调透明支持，无回调时返回 None'''
    if not _settings.user_id_callback:
        return None
    result = _settings.user_id_callback()
    if isawaitable(result):
        result = await result
    return result


def get_postgres_index_config()->PostgresIndexConfig | None:
    '''
    获取用于store的向量化配置
    如果没有传，设置为None
    '''
    if not _settings.embeding_enabled:
        return None
    else:
        # default to openai provider
        # provider embedding model should support MRL cause dimensions are forcely set
        embed_class =  init_embeddings(
            model=_settings.embeding_model_name,
            provider='openai',
            api_key=_settings.embeding_api_key,
            base_url=_settings.embeding_base_url,
            dimensions=_settings.embeding_dim
        )
        # notice ann kind of hnsw not support more than 2000 dimensions 
        index_config = PostgresIndexConfig(
            dims=_settings.embeding_dim,
            embed=embed_class,
            ann_index_config=ANNIndexConfig(kind='hnsw', vector_type='vector'),
            distance_type='cosine'
        )
        return index_config


def _require_database_uri():
    if not _settings.langgraph_database_uri:
        raise RuntimeError("langgraph_database_uri is required but not configured")


@asynccontextmanager
async def get_graph_conn() -> AsyncIterator[_ainternal.Conn]:
    '''
    创建共享数据库连接，供 checkpointer 和 store 复用。
    '''
    _require_database_uri()
    async with await AsyncConnection.connect(
        _settings.langgraph_database_uri,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    ) as conn:
        yield conn


@asynccontextmanager
async def get_graph_checkpointer(
    conn: _ainternal.Conn | None = None,
) -> AsyncIterator[AsyncPostgresSaver]:
    '''
    返回 checkpointer。
    如果传入 conn，则复用该连接；
    否则自行从 langgraph_database_uri 创建连接。
    '''
    _require_database_uri()
    async with AsyncExitStack() as stack:
        if conn is not None:
            checkpointer = AsyncPostgresSaver(conn=conn)
        else:
            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(_settings.langgraph_database_uri)
            )
        yield checkpointer


async def _close_batched_store(store: AsyncPostgresStore) -> None:
    """关闭 AsyncBatchedBaseStore 的后台 batch 任务，避免 pending task 警告。"""
    if hasattr(store, "stop_ttl_sweeper"):
        try:
            await store.stop_ttl_sweeper(timeout=1.0)
        except Exception:
            pass

    task = getattr(store, "_task", None)
    if task is None or task.done():
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def get_graph_store(
    conn: _ainternal.Conn | None = None,
) -> AsyncIterator[AsyncPostgresStore]:
    '''
    返回 store。
    如果传入 conn，则复用该连接；
    否则自行从 langgraph_database_uri 创建连接。
    '''
    _require_database_uri()
    index_config = get_postgres_index_config()
    async with AsyncExitStack() as stack:
        if conn is not None:
            store = AsyncPostgresStore(conn=conn, index=index_config)
        else:
            store = await stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(
                    _settings.langgraph_database_uri, index=index_config
                )
            )
        try:
            yield store
        finally:
            await _close_batched_store(store)


@asynccontextmanager
async def get_thread_store(
    conn: _ainternal.Conn | None = None,
) -> AsyncIterator[AsyncPostgresThread]:
    '''
    返回 thread store。
    如果传入 conn，则复用该连接；
    否则自行从 langgraph_database_uri 创建连接。
    '''
    _require_database_uri()
    index_config = get_postgres_index_config()
    async with AsyncExitStack() as stack:
        if conn is not None:
            store = AsyncPostgresThread(conn=conn, index=index_config)
        else:
            store = await stack.enter_async_context(
                AsyncPostgresThread.from_conn_string(
                    _settings.langgraph_database_uri, index=index_config
                )
            )
        yield store


@asynccontextmanager
async def get_cron_store(
    conn: _ainternal.Conn | None = None,
) -> AsyncIterator[AsyncPostgresCron]:
    '''
    返回 cron store。
    如果传入 conn，则复用该连接；
    否则自行从 langgraph_database_uri 创建连接。
    '''
    _require_database_uri()
    async with AsyncExitStack() as stack:
        if conn is not None:
            store = AsyncPostgresCron(conn=conn)
        else:
            store = await stack.enter_async_context(
                AsyncPostgresCron.from_conn_string(
                    _settings.langgraph_database_uri
                )
            )
        yield store


@asynccontextmanager
async def get_assistant_store(
    conn: _ainternal.Conn | None = None,
) -> AsyncIterator[AsyncPostgresAssistant]:
    '''
    返回 assistant store。
    如果传入 conn，则复用该连接；
    否则自行从 langgraph_database_uri 创建连接。
    '''
    _require_database_uri()
    index_config = get_postgres_index_config()
    async with AsyncExitStack() as stack:
        if conn is not None:
            store = AsyncPostgresAssistant(conn=conn, index=index_config)
        else:
            store = await stack.enter_async_context(
                AsyncPostgresAssistant.from_conn_string(
                    _settings.langgraph_database_uri, index=index_config
                )
            )
        yield store
