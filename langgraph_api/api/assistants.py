from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Response

from langgraph_api.utils.models import (
    Assistant,
    AssistantCreateRequest,
    AssistantSearchRequest,
    AssistantSearchResponse,
    AssistantSetVersionRequest,
    AssistantUpdateRequest,
    AssistantVersion,
)
from langgraph_api.registry import get_assistant_store, get_user_id

router = APIRouter(prefix="/assistants", tags=["assistants"])


def _row_to_assistant(row: dict) -> Assistant:
    return Assistant(
        assistant_id=row["assistant_id"],
        graph_id=row["graph_id"],
        config=row.get("config", {}),
        context=row.get("context", {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=row.get("metadata", {}),
        version=row.get("version", 1),
        name=row.get("name", "Untitled"),
        description=row.get("description"),
        user_id=row.get("user_id"),
    )


async def _require_assistant(assistant_id: str) -> dict:
    uid = await get_user_id()
    async with get_assistant_store() as store:
        row = await store.assistant_get(
            assistant_id,
            user_id=str(uid) if uid is not None else None,
        )
    if not row:
        raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")
    return row


# ── Assistant CRUD ──────────────────────────────────────────────────────

@router.post("")
async def create(payload: AssistantCreateRequest) -> Assistant:
    assistant_id = payload.assistant_id or str(uuid4())
    uid = await get_user_id()
    effective_user_id = str(uid) if uid is not None else None

    async with get_assistant_store() as store:
        if payload.if_exists == "raise":
            exist = await store.assistant_get(
                assistant_id,
                user_id=effective_user_id,
            )
            if exist:
                raise HTTPException(
                    status_code=403,
                    detail=f"Assistant {assistant_id} already exists and if_exists=raise",
                )

        if payload.graph_id is None:
            raise HTTPException(
                status_code=400,
                detail="graph_id is required when creating an assistant",
            )

        row = await store.assistant_create(
            assistant_id=assistant_id,
            graph_id=payload.graph_id,
            config=payload.config,
            context=payload.context,
            metadata=payload.metadata,
            name=payload.name,
            description=payload.description,
            user_id=effective_user_id,
        )
    return _row_to_assistant(row)


@router.get("/{assistant_id}")
async def get(assistant_id: str) -> Assistant:
    uid = await get_user_id()
    async with get_assistant_store() as store:
        row = await store.assistant_get(
            assistant_id,
            user_id=str(uid) if uid is not None else None,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")
    return _row_to_assistant(row)


@router.patch("/{assistant_id}")
async def update(assistant_id: str, payload: AssistantUpdateRequest) -> Assistant:
    uid = await get_user_id()
    async with get_assistant_store() as store:
        exist = await store.assistant_get(
            assistant_id,
            user_id=str(uid) if uid is not None else None,
        )
        if exist is None:
            raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")

        merged_metadata = None
        if payload.metadata:
            merged_metadata = {**exist.get("metadata", {}), **payload.metadata}

        row = await store.assistant_update(
            assistant_id,
            graph_id=payload.graph_id,
            config=payload.config,
            context=payload.context,
            metadata=merged_metadata if payload.metadata else payload.metadata,
            name=payload.name,
            description=payload.description,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")
    return _row_to_assistant(row)


@router.delete("/{assistant_id}", status_code=204)
async def delete(
    assistant_id: str,
    delete_threads: bool = Query(False, description="Delete associated threads"),
) -> None:
    uid = await get_user_id()
    async with get_assistant_store() as store:
        deleted = await store.assistant_delete(
            assistant_id,
            user_id=str(uid) if uid is not None else None,
        )
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")

    if delete_threads:
        from langgraph_api.registry import get_thread_store

        async with get_thread_store() as thread_store:
            rows = await thread_store.thread_search(
                metadata_filter={"assistant_id": assistant_id},
                limit=1000,
                user_id=str(uid) if uid is not None else None,
            )
            if rows:
                await thread_store.thread_batch_delete(
                    [r["thread_id"] for r in rows]
                )


@router.post("/search")
async def search(payload: AssistantSearchRequest) -> list[Assistant]:
    uid = await get_user_id()
    async with get_assistant_store() as store:
        rows = await store.assistant_search(
            metadata_filter=payload.metadata,
            graph_id=payload.graph_id,
            name=payload.name,
            limit=payload.limit,
            offset=payload.offset,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
            user_id=str(uid) if uid is not None else None,
        )
    return [_row_to_assistant(row) for row in rows]


@router.post("/count")
async def count(payload: AssistantSearchRequest) -> int:
    uid = await get_user_id()
    async with get_assistant_store() as store:
        cnt = await store.assistant_count(
            metadata_filter=payload.metadata,
            graph_id=payload.graph_id,
            name=payload.name,
            user_id=str(uid) if uid is not None else None,
        )
    return cnt


# ── Version operations ──────────────────────────────────────────────────

@router.post("/{assistant_id}/versions")
async def get_versions(
    assistant_id: str,
    payload: AssistantSearchRequest | None = None,
    limit: int = Query(10),
    offset: int = Query(0),
) -> list[AssistantVersion]:
    await _require_assistant(assistant_id)
    payload = payload or AssistantSearchRequest()
    async with get_assistant_store() as store:
        rows = await store.assistant_get_versions(
            assistant_id,
            metadata_filter=payload.metadata,
            limit=limit,
            offset=offset,
        )
    return [
        AssistantVersion(
            assistant_id=r["assistant_id"],
            version=r["version"],
            graph_id=r["graph_id"],
            config=r.get("config", {}),
            context=r.get("context", {}),
            metadata=r.get("metadata", {}),
            name=r.get("name", "Untitled"),
            description=r.get("description"),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@router.post("/{assistant_id}/latest")
async def set_latest(
    assistant_id: str,
    payload: AssistantSetVersionRequest,
) -> Assistant:
    uid = await get_user_id()
    async with get_assistant_store() as store:
        row = await store.assistant_set_latest(assistant_id, payload.version)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Assistant {assistant_id} version {payload.version} not found",
        )
    return _row_to_assistant(row)
