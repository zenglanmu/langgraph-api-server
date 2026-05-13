'''
持久化保持，扩展langgraph保存类
'''
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from .thread import AsyncPostgresThread
from .cron import AsyncPostgresCron
from .assistant import AsyncPostgresAssistant


async def setup():
    '''
    initialize db
    '''    
    from ..registry import get_graph_conn, get_postgres_index_config
    
    async with get_graph_conn() as conn:
        saver = AsyncPostgresSaver(conn)
        index = get_postgres_index_config()
        store = AsyncPostgresStore(conn, index=index)
        thread = AsyncPostgresThread(conn, index=index)
        cron = AsyncPostgresCron(conn)
        assistant = AsyncPostgresAssistant(conn, index=index)
        
        await saver.setup()
        await store.setup()
        await thread.setup()
        await cron.setup()
        await assistant.setup()
