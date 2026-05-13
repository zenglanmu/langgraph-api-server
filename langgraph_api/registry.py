from contextlib import AsyncExitStack, asynccontextmanager
import os
from inspect import isawaitable
from typing import AsyncIterator, Awaitable, Callable
from langchain.embeddings import init_embeddings
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres.base import PostgresIndexConfig, ANNIndexConfig
from langgraph.checkpoint.postgres.aio import _ainternal
import importlib
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
as compiled langchain graph can not pickled into new process
and we plan to run graph in backgroud process
a func for creating compiled graph is instead, see:
https://github.com/langchain-ai/langgraph/issues/3289
'''
CompileGraphCallback = Callable[[], CompiledStateGraph]


def _func_to_dotted_path(func: Callable) -> str:
    module = func.__module__
    qualname = func.__qualname__
    return f"{module}:{qualname}"


def _dotted_path_to_func(dotted_path: str) -> Callable:
    module_path, qualname = dotted_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


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
        '''在主进程中设置运行时配置，user_id_callback 仅在主进程可用，不参与序列化'''
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

    def snapshot(self) -> dict:
        '''将当前配置序列化为可跨进程传递的字典，graph_registry 中的函数转为模块路径字符串，user_id_callback 不参与序列化'''
        graph_paths = {
            name: _func_to_dotted_path(func)
            for name, func in self.graph_registry.items()
        }
        return {
            'redis_url': self.redis_url,
            'langfuse_public_key': self.langfuse_public_key,
            'langfuse_secret_key': self.langfuse_secret_key,
            'langfuse_base_url': self.langfuse_base_url,
            'langgraph_database_uri': self.langgraph_database_uri,
            'graph_registry': graph_paths,
            'embeding_model_name': self.embeding_model_name,
            'embeding_dim': self.embeding_dim,
            'embeding_base_url': self.embeding_base_url,
            'embeding_api_key': self.embeding_api_key,            
        }

    def load(self, data: dict):
        '''
        从 snapshot 产生的字典中恢复配置，graph_registry 中的函数通过模块路径动态导入还原
        注意虽然user_id_callback在新进程中恢复了，但是他可能依赖fastapi的请求上下文，所以不一定能工作
        '''
        self.redis_url = data.get('redis_url')
        self.langfuse_public_key = data.get('langfuse_public_key')
        self.langfuse_secret_key = data.get('langfuse_secret_key')
        self.langfuse_base_url = data.get('langfuse_base_url')
        self._setup_langfuse_env()
                   
        self.langgraph_database_uri = data.get('langgraph_database_uri')
        graph_paths = data.get('graph_registry', {})
        self.graph_registry = {
            name: _dotted_path_to_func(path)
            for name, path in graph_paths.items()
        }
            
        self.embeding_model_name = data.get('embeding_model_name')
        self.embeding_dim = data.get('embeding_dim')
        self.embeding_base_url = data.get('embeding_base_url')
        self.embeding_api_key = data.get('embeding_api_key')
    
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
        yield store


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
