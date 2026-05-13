from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.sse import EventSourceResponse

from langgraph_api.utils.models import (
    CancelManyRunsRequest,
    Run,
    RunCreate,
    StreamRunRequest,
)
from langgraph_api.services.run_queue_service import (
    cancel_run,
    enqueue_run,
    get_run,
    wait_for_run,
)
from langgraph_api.services.graph_run_service import stream_agent_run_events
from langgraph_api.registry import get_thread_store

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("/stream", response_class=EventSourceResponse)
async def stream_run_stateless(payload: StreamRunRequest):
    run_id = await enqueue_run(
        thread_id="",
        payload=payload,
        temporary=True,
    )

    async for event in stream_agent_run_events(
        run_id=run_id,
    ):
        yield event


@router.post("")
async def create_run_stateless(payload: StreamRunRequest) -> Run:
    run_id = await enqueue_run(
        thread_id="",
        payload=payload,
        temporary=True,
    )
    run_data = await get_run(run_id)
    if run_data is None:
        raise HTTPException(status_code=500, detail="Failed to create run")
    return Run.model_validate(run_data)


@router.post("/batch")
async def create_run_batch(payloads: list[RunCreate]) -> list[Run]:
    results: list[Run] = []
    for p in payloads:
        stream_req = StreamRunRequest.model_validate(p.model_dump(exclude_none=True))
        run_id = await enqueue_run(
            thread_id="",
            payload=stream_req,
            temporary=True,
        )
        run_data = await get_run(run_id)
        if run_data:
            results.append(Run.model_validate(run_data))
    return results


@router.post("/wait")
async def wait_run_stateless(payload: StreamRunRequest) -> dict:
    run_id = await enqueue_run(
        thread_id="",
        payload=payload,
        temporary=True,
    )

    try:
        run_data = await wait_for_run(run_id, timeout=600)
    except TimeoutError:
        raise HTTPException(status_code=408, detail=f"Run {run_id} timed out")
    return run_data


@router.post("/cancel")
async def cancel_many_runs(
    payload: CancelManyRunsRequest,
    action: str = Query("interrupt", description="Cancel action: interrupt or rollback"),
) -> None:
    from langgraph_api.services.run_queue_service import publish_cancel_signal

    cancelled_ids: list[str] = []
    if payload.run_ids:
        for rid in payload.run_ids:
            await publish_cancel_signal(rid)
            await cancel_run(rid, action=action)
            cancelled_ids.append(rid)
    elif payload.thread_id or payload.status:
        async with get_thread_store() as store:
            cancelled_ids = await store.run_cancel_many(
                thread_id=payload.thread_id,
                status=payload.status,
            )
        for rid in cancelled_ids:
            await publish_cancel_signal(rid)
