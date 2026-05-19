'''
run langgraph with custom settings and fix..
后台运行langgraph graph，将结果写入Redis stream，
供前端通过join_stream API读取。

核心设计：
- enqueue_run() 在入队前生成 run_id，立即返回给API层
- RQ worker 调用 run_lg_graph_to_redis_sync()，使用预生成的 run_id
- run_lg_graph_to_redis() 执行 graph 并将事件写入 Redis stream
- graph_run_service.stream_agent_run_events() 从 Redis stream 轮询事件，以 SSE 返回
'''
import asyncio
import base64
from datetime import UTC, datetime
import json
import os
from logging import getLogger
from typing import Any, AsyncIterable, TypedDict, cast
from uuid import UUID, uuid4
from contextlib import AsyncExitStack
from pydantic import BaseModel
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamPart
from langfuse import observe, propagate_attributes, Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from uuid_utils import uuid7

from ..registry import GraphRegistry, _settings, get_graph_checkpointer, get_user_id
from ..utils.models import InputModel, StreamRunRequest
from ..utils.queue_worker import get_redis_client


logger = getLogger(__name__)


RUN_CANCEL_KEY_TTL_SECONDS = int(os.getenv("RUN_CANCEL_KEY_TTL_SECONDS", "1800"))
RUN_EVENTS_STREAM_TTL_SECONDS = int(os.getenv("RUN_EVENTS_STREAM_TTL_SECONDS", "7200"))
RUN_EVENTS_STREAM_MAXLEN = int(os.getenv("RUN_EVENTS_STREAM_MAXLEN", "0"))

TERMINAL_RUN_STATUSES = frozenset({"success", "error", "cancelled", "timeout"})


def generate_run_id() -> str:
    return str(uuid7())


async def get_lg_graph_with_custom_fix(
    payload: StreamRunRequest,
    thread_id: str | UUID | None = None,
    run_id: str | None = None,
) -> tuple[CompiledStateGraph, RunnableConfig]:
    agent = GraphRegistry.get_lg_graph(payload.assistant_id)

    if run_id is None:
        run_id = str(uuid7())
    if thread_id is None:
        thread_id = str(uuid7())

    if _settings.langfuse_enabled:
        langfuse_callback = LangfuseCallbackHandler()
        callbacks = [langfuse_callback]
    else:
        callbacks = None

    user_id = await get_user_id()

    configurable = {
        "thread_id": thread_id,
        "graph_id": payload.assistant_id,
        "agent_id": payload.assistant_id,
        "user_id": user_id,
    }
    base_config = RunnableConfig(
        run_id=UUID(run_id),
        configurable=configurable,
        metadata=payload.metadata,
        callbacks=callbacks,
    )
    if payload.config:
        config = merge_configs(base_config, payload.config)
    else:
        config = base_config

    return agent, config


class EventData(TypedDict):
    event: str
    data: Any
    id: str


async def _stream_run_lg_graph_base(
    *,
    agent,
    config,
    thread_id: str,
    payload: StreamRunRequest,
    run_id: str | None = None,
    temporary: bool = False,
) -> AsyncIterable[EventData]:
    
    input = None
    command = None

    if payload.command:
        cmd = payload.command
        command_parts = {k: v for k, v in cmd.model_dump().items() if v is not None}
        if command_parts:
            command = command_parts

    if command is not None:
        input = None
    elif payload.input:
        input = InputModel.model_dump(payload.input)
    else:
        input = {}

    id = 0

    yield EventData(
        data={
            "run_id": str(config["run_id"]),
            "thread_id": str(config["configurable"]["thread_id"]),
        },
        event="metadata",
        id=str(id),
    )

    stream_mode = payload.stream_mode
    if stream_mode == "messages-tuple":
        stream_mode = "messages"
    elif isinstance(stream_mode, list) and "messages-tuple" in stream_mode:
        stream_mode.remove("messages-tuple")
        if "messages" not in stream_mode:
            stream_mode.append("messages")

    # Resolve context: merge payload.context with persisted assistant context
    run_context: dict[str, Any] | None = None
    if payload.context:
        run_context = payload.context
    else:
        try:
            from ..registry import get_assistant_store

            async with get_assistant_store() as assistant_store:
                assistant_row = await assistant_store.assistant_get(payload.assistant_id)
                if assistant_row and assistant_row.get("context"):
                    run_context = assistant_row["context"]
        except Exception:
            logger.debug(
                f"Failed to load assistant context for {payload.assistant_id}",
                exc_info=True,
            )

    async with AsyncExitStack() as stack:
        if not temporary:
            checkpointer = await stack.enter_async_context(get_graph_checkpointer())
            agent.checkpointer = checkpointer

        try:
            astream_kwargs: dict[str, Any] = {
                "config": config,
                "stream_mode": stream_mode,
                "interrupt_before": payload.interrupt_before,
                "interrupt_after": payload.interrupt_after,
            }
            if command is not None:
                astream_kwargs["command"] = command
                astream_kwargs["input"] = input or {}
            else:
                astream_kwargs["input"] = input or {}
            if run_context is not None:
                astream_kwargs["context"] = run_context

            async for event in agent.astream(**astream_kwargs, version="v2"):
                id += 1                
                if "type" in event:
                    # in stream v2, event should be StreamPart type, do check to make more verbose
                    stream_mode = event["type"]
                    part = cast(StreamPart, event)
                    data = part["data"]
                else:
                    # fallback
                    if isinstance(event, tuple):
                        stream_mode = event[0]
                        data = event[1]
                    else:
                        stream_mode = 'values'
                        data = event                                
                yield EventData(data=data, event=stream_mode, id=str(id))
        except Exception as e:
            payload_data = {"error": str(e), "run_id": str(config["run_id"])}
            id += 1
            yield EventData(data=payload_data, event="error", id=str(id))
            logger.error(f"Error in graph run: {e}", exc_info=True)

    id += 1
    yield EventData(data=None, event="end", id=str(id))
        

async def stream_run_lg_graph(
    *,
    thread_id: str,
    payload: StreamRunRequest,
    run_id: str | None = None,
    temporary: bool = False,
) -> AsyncIterable[EventData]:
    '''
    add langfuse to _stream_run_lg_graph_base
    if langfuse is configured
    '''
    agent, config = await get_lg_graph_with_custom_fix(
        payload, thread_id=thread_id, run_id=run_id,
    )
    if _settings.langfuse_enabled:    
        with propagate_attributes(
            trace_name=f'graph:{payload.assistant_id}:threads:{thread_id}',
            session_id=run_id
        ):
            run_lg_graph = observe()(_stream_run_lg_graph_base)
            async for event in run_lg_graph(
                agent=agent,
                config=config,
                thread_id=thread_id,
                payload=payload,
                run_id=run_id,
                temporary=temporary
            ):
                yield event
            
            # 刷新langfuse
            if len(config["callbacks"])>0:
                for handler in config["callbacks"]:
                    if isinstance(handler, LangfuseCallbackHandler):
                        client = handler._get_parent_observation(parent_run_id=None)
                        client.flush()
    else:
        async for event in _stream_run_lg_graph_base(
                agent=agent,
                config=config,
                thread_id=thread_id,
                payload=payload,
                run_id=run_id,
                temporary=temporary
            ):
                yield event


# ── Redis key helpers ──────────────────────────────────────────────────

RUN_CANCEL_CHANNEL = "langgraph:run:cancel:ch"


def _cancel_key(run_id: str) -> str:
    return f"langgraph:run:cancel:{run_id}"


def _event_stream_key(run_id: str) -> str:
    return f"langgraph:run:events:{run_id}"


def _run_status_key(run_id: str) -> str:
    return f"langgraph:run:status:{run_id}"


def _is_valid_stream_seq(value: str) -> bool:
    major, sep, minor = value.partition("-")
    if sep != "-":
        return False
    return major.isdigit() and minor.isdigit()


def normalize_after_seq(after_seq: str | int | None) -> str:
    if after_seq is None:
        return "0-0"
    if isinstance(after_seq, int):
        return "0-0"
    text = str(after_seq).strip()
    if not text:
        return "0-0"
    if _is_valid_stream_seq(text):
        return text
    return "0-0"


# ── Redis stream read ──────────────────────────────────────────────────

async def list_run_stream_events(
    run_id: str,
    after_seq: str = "0-0",
    limit: int = 200,
) -> list[dict]:
    redis = await get_redis_client()
    key = _event_stream_key(run_id)
    normalized = normalize_after_seq(after_seq)

    results = await redis.xrange(key, min=normalized, max="+", count=limit + 1)

    events = []
    for stream_id, fields in results:
        if str(stream_id) == normalized and normalized != "0-0":
            continue
        
        data_pack_type = fields.get('data_pack_type', None)
        data_value = fields.get('data_value', None)
        if data_pack_type is not None and data_value is not None:
            serde = JsonPlusSerializer()
            data_value = base64.b64decode(data_value)
            payload = serde.loads_typed((data_pack_type, data_value))
        else:
            payload = {}
            
        events.append({
            "seq": str(stream_id),
            "id": fields.get("id", "0"),
            "event_type": fields.get("event", "values"),
            "payload": payload,
            "ts": fields.get("ts"),
        })

    return events[:limit]


async def get_last_run_stream_seq(run_id: str) -> str:
    redis = await get_redis_client()
    key = _event_stream_key(run_id)
    rows = await redis.xrevrange(key, max="+", min="-", count=1)
    if not rows:
        return "0-0"
    event_id, _ = rows[0]
    return str(event_id)


# ── Run status (Redis + DB) ──────────────────────────────────────────────

async def set_run_status(
    run_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    redis = await get_redis_client()
    key = _run_status_key(run_id)
    value = json.dumps({"status": status, "error_message": error_message}, ensure_ascii=False)
    await redis.set(key, value, ex=RUN_EVENTS_STREAM_TTL_SECONDS)

    db_kwargs: dict = {"status": status}
    if status == "running":
        db_kwargs["started_at"] = datetime.now(tz=UTC)
    elif status in TERMINAL_RUN_STATUSES:
        db_kwargs["finished_at"] = datetime.now(tz=UTC)
    if error_message is not None:
        db_kwargs["error_message"] = error_message

    try:
        from ..registry import get_thread_store

        async with get_thread_store() as store:
            await store.run_update(run_id, **db_kwargs)
    except Exception:
        logger.warning(f"Failed to persist run status to DB for {run_id}", exc_info=True)


async def get_run_status(run_id: str) -> dict | None:
    redis = await get_redis_client()
    key = _run_status_key(run_id)
    raw = await redis.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ── Cancel signal ──────────────────────────────────────────────────────

async def publish_cancel_signal(run_id: str) -> None:
    redis = await get_redis_client()
    key = _cancel_key(run_id)
    try:
        await redis.set(key, "1", ex=RUN_CANCEL_KEY_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Failed to publish cancel signal for run {run_id}: {e}")


async def has_cancel_signal(run_id: str) -> bool:
    redis = await get_redis_client()
    key = _cancel_key(run_id)
    try:
        return bool(await redis.get(key))
    except Exception as e:
        logger.warning(f"Failed to read cancel signal for run {run_id}: {e}")
        return False


async def clear_cancel_signal(run_id: str) -> None:
    redis = await get_redis_client()
    key = _cancel_key(run_id)
    try:
        await redis.delete(key)
    except Exception as e:
        logger.warning(f"Failed to clear cancel signal for run {run_id}: {e}")


# ── Run graph to Redis stream (background worker entry) ────────────────

async def run_lg_graph_to_redis(
    *,
    run_id: str,
    thread_id: str,
    payload: StreamRunRequest,
    temporary: bool = False,
):
    try:
        await set_run_status(run_id, "running")

        async for event_data in stream_run_lg_graph(
            thread_id=thread_id,
            payload=payload,
            run_id=run_id,
            temporary=temporary,
        ):
            if await has_cancel_signal(run_id=run_id):
                await clear_cancel_signal(run_id=run_id)
                await set_run_status(run_id, "cancelled")
                break
            
            if event_data["event"] == "error":
                error_msg = event_data["data"].get("error", "unknown error")
                await set_run_status(run_id, "error", error_message=error_msg)

            redis = await get_redis_client()
            key = _event_stream_key(run_id)
            now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
            
            fields = {
                "id": event_data["id"],
                "event": event_data["event"],
                "ts": str(now_ms),
            }
            if event_data["data"]:
                serde = JsonPlusSerializer()
                data_pack_type, data_value = serde.dumps_typed(event_data["data"])
                fields['data_pack_type'] = data_pack_type
                fields['data_value'] = base64.b64encode(data_value).decode()

            kwargs = {}
            if RUN_EVENTS_STREAM_MAXLEN > 0:
                kwargs["maxlen"] = RUN_EVENTS_STREAM_MAXLEN
                kwargs["approximate"] = True

            await redis.xadd(key, fields, **kwargs)
            await redis.expire(key, RUN_EVENTS_STREAM_TTL_SECONDS)


        await set_run_status(run_id, "success")

    except Exception as e:
        logger.error(f"run_lg_graph_to_redis failed: {e}", exc_info=True)
        await set_run_status(run_id, "error", error_message=str(e))


# ── Sync wrapper for RQ task (RQ runs sync functions) ──────────────────

def run_lg_graph_to_redis_sync(
    run_id: str,
    thread_id: str,
    payload_dict: dict,
    temporary: bool = False,
):
    from ..utils import queue_worker as _qw

    payload = StreamRunRequest.model_validate(payload_dict)

    _qw._redis_client = None

    try:
        asyncio.run(
            run_lg_graph_to_redis(
                run_id=run_id,
                thread_id=thread_id,
                payload=payload,
                temporary=temporary,
            )
        )
    finally:
        _qw._redis_client = None


# ── Enqueue run to RQ ──────────────────────────────────────────────────

async def enqueue_run(
    *,
    thread_id: str,
    payload: StreamRunRequest,
    temporary: bool = False,
) -> str:
    from ..utils.queue_worker import get_rq_queue
    from ..registry import get_thread_store

    run_id = generate_run_id()
    await set_run_status(run_id, "pending")

    if not temporary and thread_id:
        try:
            async with get_thread_store() as store:
                await store.run_put(
                    run_id,
                    thread_id,
                    metadata=payload.metadata,
                    status="pending",
                    assistant_id=payload.assistant_id,
                    multitask_strategy=payload.multitask_strategy,
                )
        except Exception:
            logger.warning(f"Failed to create run record in DB for {run_id}", exc_info=True)

    queue = get_rq_queue()
    payload_dict = payload.model_dump(mode="json")
    queue.enqueue(
        run_lg_graph_to_redis_sync,
        run_id=run_id,
        thread_id=thread_id,
        payload_dict=payload_dict,
        temporary=temporary,
        job_timeout=RUN_EVENTS_STREAM_TTL_SECONDS,
    )

    return run_id


# ── Run CRUD (non-streaming) ──────────────────────────────────────────

async def get_run(run_id: str) -> dict | None:
    from ..registry import get_thread_store

    redis_status = await get_run_status(run_id)

    async with get_thread_store() as store:
        row = await store.run_get(run_id)

    if row is None and redis_status is None:
        return None

    if row is None:
        return {"run_id": run_id, "status": redis_status.get("status", "unknown") if redis_status else "unknown"}

    if redis_status and redis_status.get("status") in TERMINAL_RUN_STATUSES:
        row["status"] = redis_status["status"]
    elif redis_status:
        row["status"] = redis_status.get("status", row.get("status"))

    if redis_status and redis_status.get("error_message"):
        row["error_message"] = redis_status["error_message"]

    return row


async def list_runs(
    thread_id: str,
    *,
    limit: int = 10,
    offset: int = 0,
    status: str | None = None,
) -> list[dict]:
    from ..registry import get_thread_store

    async with get_thread_store() as store:
        return await store.run_list(thread_id, limit=limit, offset=offset, status=status)


async def delete_run(run_id: str) -> None:
    from ..registry import get_thread_store

    await publish_cancel_signal(run_id)

    async with get_thread_store() as store:
        await store.run_delete(run_id)

    redis = await get_redis_client()
    await redis.delete(_event_stream_key(run_id))
    await redis.delete(_run_status_key(run_id))
    await redis.delete(_cancel_key(run_id))


async def cancel_run(
    run_id: str,
    *,
    action: str = "interrupt",
    wait: bool = False,
) -> None:
    from ..registry import get_thread_store

    await publish_cancel_signal(run_id)

    if action == "rollback":
        logger.info(f"Rollback requested for run {run_id} (not yet implemented)")

    if wait:
        for _ in range(600):
            await asyncio.sleep(1)
            status = await get_run_status(run_id)
            if status and status.get("status") in TERMINAL_RUN_STATUSES:
                return

    async with get_thread_store() as store:
        await store.run_update(run_id, status="cancelled")


async def wait_for_run(run_id: str, timeout: int = 600) -> dict:
    elapsed = 0
    interval = 1
    while elapsed < timeout:
        status = await get_run_status(run_id)
        if status and status.get("status") in TERMINAL_RUN_STATUSES:
            return await get_run(run_id) or status
        await asyncio.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Run {run_id} did not complete within {timeout} seconds")
