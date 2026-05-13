from fastapi import APIRouter, HTTPException

from langgraph_api.utils.models import (
    Cron,
    CronCreate,
    CronSearchRequest,
    CronUpdate,
    ItemResponse,
)
from langgraph_api.services.cron_service import (
    count_crons,
    create_cron,
    delete_cron,
    get_cron,
    search_crons,
    update_cron,
)

router = APIRouter(prefix="/runs/crons", tags=["crons"])


@router.post("")
async def create_cron_stateless(payload: CronCreate) -> Cron:
    """Create a cron run (stateless, no thread binding).

    Corresponds to CronClient.create().
    """
    cron = await create_cron(
        assistant_id=payload.assistant_id,
        schedule=payload.schedule,
        thread_id=payload.thread_id,
        payload_data=payload,
    )
    return cron


@router.delete("/{cron_id}")
async def delete_cron_by_id(cron_id: str) -> ItemResponse[bool]:
    """Delete a cron.

    Corresponds to CronClient.delete().
    """
    deleted = await delete_cron(cron_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Cron {cron_id} not found")
    return ItemResponse[bool](msg="ok", data=True)


@router.patch("/{cron_id}")
async def update_cron_by_id(cron_id: str, payload: CronUpdate) -> Cron:
    """Update a cron job by ID.

    Corresponds to CronClient.update().
    """
    cron = await update_cron(cron_id, update_data=payload)
    if cron is None:
        raise HTTPException(status_code=404, detail=f"Cron {cron_id} not found")
    return cron


@router.post("/search")
async def search_cron_jobs(payload: CronSearchRequest) -> list[Cron]:
    """Get a list of cron jobs.

    Corresponds to CronClient.search().
    """
    return await search_crons(
        assistant_id=payload.assistant_id,
        thread_id=payload.thread_id,
        enabled=payload.enabled,
        limit=payload.limit,
        offset=payload.offset,
        sort_by=payload.sort_by,
        sort_order=payload.sort_order,
    )


@router.post("/count")
async def count_cron_jobs(payload: CronSearchRequest) -> int:
    """Count cron jobs matching filters.

    Corresponds to CronClient.count().
    """
    return await count_crons(
        assistant_id=payload.assistant_id,
        thread_id=payload.thread_id,
    )
