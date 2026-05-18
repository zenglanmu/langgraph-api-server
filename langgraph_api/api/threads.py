import copy as _copy
from uuid import uuid4
from typing import Any

from langchain_core.runnables import RunnableConfig
from fastapi import APIRouter, Header, HTTPException, Query, Response
from fastapi.sse import EventSourceResponse

from langgraph_api.utils.models import (
    Cron,
    CronCreate,
    Run,
    StreamRunRequest,
    Thread,
    ThreadCreateRequest,
    ThreadUpdateRequest,
    ThreadSearchRequest,
    ThreadPruneRequest,
    ThreadState,
    Checkpoint,
    ThreadUpdateStateRequest,
    ThreadUpdateStateResponse,
    ThreadGetStateRequest,
    ThreadGetHistoryRequest,
    convert_checkpoint_tuple_to_thread_state,
    convert_state_snapshot_to_thread_state,
)
from langgraph_api.registry import get_graph_checkpointer, get_thread_store, get_graph_conn, get_user_id, GraphRegistry, _settings
from langgraph_api.services.run_queue_service import (
    cancel_run,
    delete_run,
    enqueue_run,
    get_run,
    list_runs,
    wait_for_run,
)
from langgraph_api.services.graph_run_service import stream_agent_run_events
from langgraph_api.services.cron_service import create_cron

router = APIRouter(prefix="/threads", tags=["threads"])


def _row_to_thread(row: dict) -> Thread:
    return Thread(
        thread_id=row["thread_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=row["metadata"],
        status=row.get("status"),
        user_id=row.get("user_id"),
    )


async def _require_thread(thread_id: str) -> dict:
    uid = await get_user_id()
    async with get_thread_store() as store:
        row = await store.thread_get(
            thread_id,
            user_id=str(uid) if uid is not None else None,
        )
    if not row:
        raise HTTPException(status_code=404, detail=f"{thread_id} not found in store")
    return row


# ── Thread CRUD ─────────────────────────────────────────────────────────

@router.get("/{thread_id}")
async def get(
    thread_id: str,
    include: str | None = Query(None, description="Additional fields to include (comma-separated, e.g. 'ttl')"),
) -> Thread:
    uid = await get_user_id()
    async with get_thread_store() as store:
        row = await store.thread_get(thread_id, user_id=str(uid) if uid is not None else None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"{thread_id} not found in store")
    return _row_to_thread(row)


@router.post("")
async def create(payload: ThreadCreateRequest) -> Thread:
    if payload.thread_id:
        thread_id = payload.thread_id
    else:
        thread_id = str(uuid4())

    uid = await get_user_id()
    effective_user_id = str(uid) if uid is not None else None

    async with get_thread_store() as store:
        if payload.if_exists == "raise":
            exist = await store.thread_get(
                thread_id,
                user_id=effective_user_id,
            )
            if exist:
                raise HTTPException(
                    status_code=403,
                    detail=f"{thread_id} exists and if_exists strategy is raise",
                )

        row = await store.thread_put(
            thread_id,
            metadata=payload.metadata,
            user_id=effective_user_id,
        )
    return _row_to_thread(row)


@router.patch("/{thread_id}")
async def update(thread_id: str, payload: ThreadUpdateRequest) -> Thread:
    uid = await get_user_id()
    async with get_thread_store() as store:
        exist = await store.thread_get(
            thread_id,
            user_id=str(uid) if uid is not None else None,
        )
        if exist is None:
            raise HTTPException(status_code=404, detail=f"{thread_id} not found in store")

        new_metadata = {**exist["metadata"], **payload.metadata}
        row = await store.thread_put(
            thread_id,
            metadata=new_metadata,
            user_id=exist.get("user_id"),
        )
    return _row_to_thread(row)


@router.delete("/{thread_id}", status_code=204)
async def delete(thread_id: str) -> None:
    uid = await get_user_id()
    async with get_thread_store() as store:
        await store.thread_delete(
            thread_id,
            user_id=str(uid) if uid is not None else None,
        )


@router.post("/search")
async def search(payload: ThreadSearchRequest) -> list[Thread]:
    uid = await get_user_id()
    async with get_thread_store() as store:
        rows = await store.thread_search(
            metadata_filter=payload.metadata,
            values_filter=payload.values,
            status=payload.status,
            ids=payload.ids,
            limit=payload.limit,
            offset=payload.offset,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
            query=getattr(payload, "query", None),
            user_id=str(uid) if uid is not None else None,
        )
    return [_row_to_thread(row) for row in rows]


@router.post("/count")
async def count(payload: ThreadSearchRequest) -> int:
    uid = await get_user_id()
    async with get_thread_store() as store:
        cnt = await store.thread_count(
            metadata_filter=payload.metadata,
            values_filter=payload.values,
            status=payload.status,
            user_id=str(uid) if uid is not None else None,
        )
    return cnt


@router.post("/{thread_id}/copy")
async def copy(thread_id: str) -> Thread:
    uid = await get_user_id()
    effective_user_id = str(uid) if uid is not None else None
    new_thread_id = str(uuid4())
    async with get_graph_conn() as conn:
        async with get_graph_checkpointer(conn=conn) as checkpointer:
            await checkpointer.acopy_thread(thread_id, target_thread_id=new_thread_id)
        async with get_thread_store(conn=conn) as store:
            row = await store.thread_copy(
                thread_id, new_thread_id, user_id=effective_user_id
            )
    if row is None:
        raise HTTPException(status_code=404, detail=f"{thread_id} not found in store")
    return _row_to_thread(row)


@router.post("/prune")
async def prune(payload: ThreadPruneRequest) -> dict[str, int]:
    async with get_graph_conn() as conn:
        async with get_graph_checkpointer(conn=conn) as checkpointer:
            await checkpointer.aprune(thread_ids=payload.thread_ids, strategy=payload.strategy)
        if payload.strategy == "delete":
            async with get_thread_store(conn=conn) as store:
                await store.thread_batch_delete(payload.thread_ids)
    return {"pruned_count": len(payload.thread_ids)}


# ── Thread state ────────────────────────────────────────────────────────

async def _get_thread_state_via_graph(
    thread_id: str,
    checkpoint_id: str | None = None,
    checkpoint_ns: str | None = None,
    subgraphs: bool = False,
) -> ThreadState | None:
    """Get thread state using graph.aget_state() which correctly reconstructs
    DeltaChannel values (e.g. messages), falling back to raw checkpointer
    when the graph is unavailable.

    We shallow-copy the compiled graph and set checkpointer on the copy so
    that achannels_from_checkpoint (called internally by aget_state) can
    access the saver for DeltaChannel reconstruction. The original graph's
    checkpointer is None since graphs are registered as callables.
    """
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    if checkpoint_ns:
        configurable["checkpoint_ns"] = checkpoint_ns

    async with get_graph_conn() as conn:
        async with get_graph_checkpointer(conn=conn) as checkpointer:
            config = RunnableConfig(configurable=configurable)
            res = await checkpointer.aget_tuple(config)
            if res is None:
                return None

            graph_id = res.metadata.get("graph_id") or res.metadata.get("agent_id")
            if graph_id and graph_id in _settings.graph_registry:
                try:
                    graph = GraphRegistry.get_lg_graph(graph_id)
                    g = _copy.copy(graph)
                    g.checkpointer = checkpointer
                    snapshot = await g.aget_state(
                        {"configurable": configurable},
                        subgraphs=subgraphs,
                    )
                    return convert_state_snapshot_to_thread_state(snapshot)
                except Exception:
                    pass

            return convert_checkpoint_tuple_to_thread_state(res)


@router.get("/{thread_id}/state")
async def get_thread_state(thread_id: str, subgraphs: bool = False) -> ThreadState:
    result = await _get_thread_state_via_graph(thread_id, subgraphs=subgraphs)
    if result is None:
        raise HTTPException(status_code=404, detail=f"{thread_id} not found in store")
    return result


@router.get("/{thread_id}/state/{checkpoint_id}")
async def get_thread_state_checkpoint_id(
    thread_id: str, checkpoint_id: str, subgraphs: bool = False
) -> ThreadState:
    result = await _get_thread_state_via_graph(
        thread_id, checkpoint_id=checkpoint_id, subgraphs=subgraphs
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"{thread_id} not found in store")
    return result


@router.post("/{thread_id}/state/checkpoint")
async def get_thread_state_checkpoint(
    thread_id: str, payload: ThreadGetStateRequest
) -> ThreadState:
    checkpoint_id = None
    checkpoint_ns = None
    if payload.checkpoint:
        if payload.checkpoint.checkpoint_ns:
            checkpoint_ns = payload.checkpoint.checkpoint_ns
        if payload.checkpoint.checkpoint_id:
            checkpoint_id = payload.checkpoint.checkpoint_id
    elif payload.checkpoint_id:
        checkpoint_id = payload.checkpoint_id

    result = await _get_thread_state_via_graph(
        thread_id,
        checkpoint_id=checkpoint_id,
        checkpoint_ns=checkpoint_ns,
        subgraphs=payload.subgraphs,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"{thread_id} not found in store")
    return result


@router.post("/{thread_id}/state")
async def update_thread_state(
    thread_id: str, payload: ThreadUpdateStateRequest
) -> ThreadUpdateStateResponse:
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if payload.checkpoint:
        if payload.checkpoint.checkpoint_ns:
            configurable["checkpoint_ns"] = payload.checkpoint.checkpoint_ns
        if payload.checkpoint.checkpoint_id:
            configurable["checkpoint_id"] = payload.checkpoint.checkpoint_id
    elif payload.checkpoint_id:
        configurable["checkpoint_id"] = payload.checkpoint_id

    config = RunnableConfig(configurable=configurable)
    async with get_graph_checkpointer() as checkpointer:
        new_config = await checkpointer.aupdate_state(
            config,
            payload.values,
            as_node=payload.as_node,
        )

    new_configurable = new_config.get("configurable", {})
    checkpoint = Checkpoint(
        thread_id=new_configurable.get("thread_id", thread_id),
        checkpoint_ns=new_configurable.get("checkpoint_ns", ""),
        checkpoint_id=new_configurable.get("checkpoint_id"),
        checkpoint_map=None,
    )
    return ThreadUpdateStateResponse(checkpoint=checkpoint)


@router.post("/{thread_id}/history")
async def get_thread_history(
    thread_id: str, payload: ThreadGetHistoryRequest
) -> list[ThreadState]:
    config = RunnableConfig(configurable={"thread_id": thread_id})
    if payload.checkpoint:
        configurable = dict(config["configurable"])
        if payload.checkpoint.checkpoint_ns:
            configurable["checkpoint_ns"] = payload.checkpoint.checkpoint_ns
        if payload.checkpoint.checkpoint_id:
            configurable["checkpoint_id"] = payload.checkpoint.checkpoint_id
        config = RunnableConfig(configurable=configurable)

    before = None
    if payload.before:
        before_config = RunnableConfig(
            configurable={"thread_id": thread_id, "checkpoint_id": payload.before}
        )
        before = before_config

    filter_by_metadata = payload.metadata

    async with get_graph_conn() as conn:
        async with get_graph_checkpointer(conn=conn) as checkpointer:
            checkpoint_tuples = [
                c
                async for c in checkpointer.alist(
                    config, before=before, limit=payload.limit, filter=filter_by_metadata
                )
            ]

    if not checkpoint_tuples:
        return []

    graph_id = checkpoint_tuples[0].metadata.get("graph_id") or checkpoint_tuples[0].metadata.get("agent_id")
    if graph_id and graph_id in _settings.graph_registry:
        try:
            graph = GraphRegistry.get_lg_graph(graph_id)
            g = _copy.copy(graph)
            g.checkpointer = checkpointer
            history = []
            async with get_graph_conn() as conn:
                async with get_graph_checkpointer(conn=conn) as cp:
                    g.checkpointer = cp
                    for ct in checkpoint_tuples:
                        snapshot = await g._aprepare_state_snapshot(
                            ct.config,
                            ct,
                            apply_pending_writes=False,
                        )
                        history.append(convert_state_snapshot_to_thread_state(snapshot))
            return history
        except Exception:
            pass

    return [convert_checkpoint_tuple_to_thread_state(ct) for ct in checkpoint_tuples]


# ── Thread stream (threads.join_stream) ──────────────────────────────────
# Upstream: GET /threads/{thread_id}/stream
# Client sends Last-Event-ID header for reconnection, stream_mode as query param

@router.get("/{thread_id}/stream", response_class=EventSourceResponse)
async def thread_join_stream(
    thread_id: str,
    stream_mode: str | None = Query("run_modes", description="Stream mode(s)"),
    last_event_id: str | None = Header(None, alias="Last-Event-ID", description="Last event ID for reconnection"),
):
    await _require_thread(thread_id)

    async for event in stream_agent_run_events(
        run_id=None,
        after_seq=last_event_id,
    ):
        yield event


# ── Thread-scoped Run endpoints ────────────────────────────────────────
# Aligned with langgraph_sdk RunsClient:
#   POST /threads/{thread_id}/runs          → create (background)
#   POST /threads/{thread_id}/runs/stream   → stream
#   POST /threads/{thread_id}/runs/wait     → wait (blocking)
#   GET  /threads/{thread_id}/runs          → list
#   GET  /threads/{thread_id}/runs/{run_id} → get
#   POST /threads/{thread_id}/runs/{run_id}/cancel → cancel (wait/action as query params)
#   POST /threads/{thread_id}/runs/{run_id}/join    → join
#   GET  /threads/{thread_id}/runs/{run_id}/stream  → join_stream (Last-Event-ID header)
#   DELETE /threads/{thread_id}/runs/{run_id}       → delete

@router.post("/{thread_id}/runs/stream", response_class=EventSourceResponse)
async def stream_run(
    thread_id: str,
    payload: StreamRunRequest,
):
    await _require_thread(thread_id)

    run_id = await enqueue_run(
        thread_id=thread_id,
        payload=payload,
        temporary=False,
    )

    async for event in stream_agent_run_events(
        run_id=run_id,
    ):
        yield event


@router.post("/{thread_id}/runs")
async def create_run(thread_id: str, payload: StreamRunRequest) -> Run:
    await _require_thread(thread_id)

    run_id = await enqueue_run(
        thread_id=thread_id,
        payload=payload,
        temporary=False,
    )
    run_data = await get_run(run_id)
    if run_data is None:
        raise HTTPException(status_code=500, detail="Failed to create run")
    return Run.model_validate(run_data)


@router.post("/{thread_id}/runs/wait")
async def wait_run(
    thread_id: str,
    payload: StreamRunRequest,
) -> dict:
    await _require_thread(thread_id)

    run_id = await enqueue_run(
        thread_id=thread_id,
        payload=payload,
        temporary=False,
    )

    try:
        run_data = await wait_for_run(run_id, timeout=600)
    except TimeoutError:
        raise HTTPException(status_code=408, detail=f"Run {run_id} timed out")
    return run_data


@router.get("/{thread_id}/runs")
async def list_thread_runs(
    thread_id: str,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
    select: str | None = Query(None, description="Comma-separated list of fields to include"),
) -> list[Run]:
    await _require_thread(thread_id)
    rows = await list_runs(thread_id, limit=limit, offset=offset, status=status)
    return [Run.model_validate(r) for r in rows]


@router.get("/{thread_id}/runs/{run_id}")
async def get_thread_run(thread_id: str, run_id: str) -> Run:
    await _require_thread(thread_id)
    run_data = await get_run(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return Run.model_validate(run_data)


@router.post("/{thread_id}/runs/{run_id}/cancel")
async def cancel_thread_run(
    thread_id: str,
    run_id: str,
    wait: int = Query(0, description="Whether to wait until run has completed (1=yes, 0=no)"),
    action: str = Query("interrupt", description="Cancel action: interrupt or rollback"),
) -> None:
    await _require_thread(thread_id)
    await cancel_run(run_id, action=action, wait=bool(wait))


@router.post("/{thread_id}/runs/{run_id}/join")
async def join_run(thread_id: str, run_id: str) -> dict:
    await _require_thread(thread_id)
    try:
        run_data = await wait_for_run(run_id, timeout=600)
    except TimeoutError:
        raise HTTPException(status_code=408, detail=f"Run {run_id} timed out")
    return run_data


@router.get("/{thread_id}/runs/{run_id}/stream", response_class=EventSourceResponse)
async def join_stream(
    thread_id: str,
    run_id: str,
    cancel_on_disconnect: bool = Query(False),
    stream_mode: str | None = Query(None, description="Stream mode(s)"),
    last_event_id: str | None = Header(None, alias="Last-Event-ID", description="Last event ID for reconnection"),
):
    await _require_thread(thread_id)

    if cancel_on_disconnect:
        try:
            async for event in stream_agent_run_events(
                run_id=run_id,
                after_seq=last_event_id,
            ):
                yield event
        except GeneratorExit:
            await cancel_run(run_id)
            raise
    else:
        async for event in stream_agent_run_events(
            run_id=run_id,
            after_seq=last_event_id,
        ):
            yield event


@router.delete("/{thread_id}/runs/{run_id}", status_code=204)
async def delete_thread_run(thread_id: str, run_id: str) -> None:
    await _require_thread(thread_id)
    await delete_run(run_id)


# ── Thread-scoped Cron endpoints ────────────────────────────────────────
# Aligned with langgraph_sdk CronClient:
#   POST /threads/{thread_id}/runs/crons → create_for_thread

@router.post("/{thread_id}/runs/crons")
async def create_cron_for_thread(thread_id: str, payload: CronCreate) -> Cron:
    """Create a cron job for a thread.

    Corresponds to CronClient.create_for_thread().
    """
    await _require_thread(thread_id)
    cron = await create_cron(
        assistant_id=payload.assistant_id,
        schedule=payload.schedule,
        thread_id=thread_id,
        payload_data=payload,
    )
    return cron
