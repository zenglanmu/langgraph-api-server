'''
Cron job management service.

Persists cron records in PostgreSQL (via AsyncPostgresCron) and
schedules actual execution through rq's CronScheduler.

Design:
- API layer: create/update/delete crons → persist to DB → sync to CronScheduler via Redis
- Worker process: CronScheduler loads all enabled crons from DB on startup,
  then listens for runtime sync events via Redis pub/sub to add/remove jobs.

The CronScheduler instance is shared between the worker process (which runs it)
and the API process (which registers/unregisters jobs on it).
'''
import asyncio
import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

from croniter import croniter
from rq.cron import CronScheduler, CronJob

from ..registry import get_cron_store, get_user_id, _settings
from ..utils.models import CronCreate, CronUpdate, Cron as CronModel
from ..utils.queue_worker import get_cron_scheduler, get_redis_client, get_rq_queue, get_sync_redis_client

logger = logging.getLogger(__name__)

CRON_SYNC_CHANNEL = "langgraph:cron:sync"


def _build_cron_payload(create_data: CronCreate) -> dict:
    payload: dict = {"assistant_id": create_data.assistant_id}
    if create_data.input is not None:
        payload["input"] = create_data.input.model_dump()
    if create_data.config is not None:
        payload["config"] = create_data.config
    if create_data.context is not None:
        payload["context"] = create_data.context
    if create_data.interrupt_before is not None:
        payload["interrupt_before"] = create_data.interrupt_before
    if create_data.interrupt_after is not None:
        payload["interrupt_after"] = create_data.interrupt_after
    if create_data.webhook is not None:
        payload["webhook"] = create_data.webhook
    if create_data.multitask_strategy is not None:
        payload["multitask_strategy"] = create_data.multitask_strategy
    if create_data.stream_mode is not None:
        payload["stream_mode"] = create_data.stream_mode
    if create_data.stream_subgraphs:
        payload["stream_subgraphs"] = create_data.stream_subgraphs
    if create_data.stream_resumable:
        payload["stream_resumable"] = create_data.stream_resumable
    if create_data.durability is not None:
        payload["durability"] = create_data.durability
    return payload


def _compute_next_run_date(schedule: str) -> datetime:
    now = datetime.now(tz=UTC)
    cron_iter = croniter(schedule, now)
    return cron_iter.get_next(datetime)


def _row_to_cron(row: dict) -> CronModel:
    return CronModel(
        cron_id=row["cron_id"],
        assistant_id=row["assistant_id"],
        thread_id=row.get("thread_id"),
        on_run_completed=row.get("on_run_completed"),
        end_time=row.get("end_time"),
        schedule=row["schedule"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        payload=row.get("payload", {}),
        user_id=row.get("user_id"),
        next_run_date=row.get("next_run_date"),
        metadata=row.get("metadata", {}),
        enabled=row.get("enabled", True),
    )


def _cron_job_meta(cron_id: str) -> dict:
    return {"cron_id": cron_id}


def _find_cron_job_by_id(scheduler: CronScheduler, cron_id: str) -> CronJob | None:
    for job in scheduler.get_jobs():
        if job.job_options.get("meta", {}).get("cron_id") == cron_id:
            return job
    return None


def _remove_cron_job_by_id(scheduler: CronScheduler, cron_id: str) -> bool:
    job = _find_cron_job_by_id(scheduler, cron_id)
    if job is None:
        return False
    scheduler._cron_jobs.remove(job)
    return True


def _register_cron_job(
    scheduler: CronScheduler,
    *,
    cron_id: str,
    assistant_id: str,
    thread_id: str | None,
    schedule: str,
    payload: dict,
    on_run_completed: str | None,
    metadata: dict | None = None,
) -> CronJob:
    _remove_cron_job_by_id(scheduler, cron_id)

    return scheduler.register(
        func=_cron_task_func,
        queue_name=get_rq_queue().name,
        cron=schedule,
        kwargs={
            "cron_id": cron_id,
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "payload_dict": payload,
            "on_run_completed": on_run_completed,
        },
        meta=_cron_job_meta(cron_id),
    )


async def _publish_cron_sync_event(event_type: str, cron_id: str, data: dict | None = None):
    redis = await get_redis_client()
    message = {
        "event": event_type,
        "cron_id": cron_id,
        "data": data,
    }
    await redis.publish(CRON_SYNC_CHANNEL, json.dumps(message, ensure_ascii=False, default=str))


async def create_cron(
    *,
    assistant_id: str,
    schedule: str,
    thread_id: str | None = None,
    payload_data: CronCreate,
) -> CronModel:
    cron_id = str(uuid4())
    user_id = await get_user_id()
    effective_user_id = str(user_id) if user_id is not None else None

    next_run_date = _compute_next_run_date(schedule)
    cron_payload = _build_cron_payload(payload_data)
    on_run_completed = payload_data.on_run_completed
    enabled = payload_data.enabled if payload_data.enabled is not None else True
    metadata = payload_data.metadata or {}

    async with get_cron_store() as store:
        row = await store.cron_put(
            cron_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            schedule=schedule,
            end_time=payload_data.end_time,
            enabled=enabled,
            on_run_completed=on_run_completed,
            payload=cron_payload,
            metadata=metadata,
            next_run_date=next_run_date,
            user_id=effective_user_id,
        )

    cron = _row_to_cron(row)

    if enabled:
        scheduler = get_cron_scheduler()
        _register_cron_job(
            scheduler,
            cron_id=cron_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            schedule=schedule,
            payload=cron_payload,
            on_run_completed=on_run_completed,
            metadata=metadata,
        )
        scheduler.save_jobs_data()

        await _publish_cron_sync_event("create", cron_id, {
            "cron_id": cron_id,
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "schedule": schedule,
            "payload": cron_payload,
            "metadata": metadata,
            "end_time": payload_data.end_time.isoformat() if payload_data.end_time else None,
            "on_run_completed": on_run_completed,
        })

    return cron


async def get_cron(cron_id: str) -> CronModel | None:
    async with get_cron_store() as store:
        row = await store.cron_get(cron_id)
    if row is None:
        return None
    return _row_to_cron(row)


async def update_cron(
    cron_id: str,
    *,
    update_data: CronUpdate,
) -> CronModel | None:
    async with get_cron_store() as store:
        existing = await store.cron_get(cron_id)
        if existing is None:
            return None

        update_kwargs: dict = {}
        new_payload = dict(existing.get("payload", {}))

        if update_data.schedule is not None:
            update_kwargs["schedule"] = update_data.schedule
            update_kwargs["next_run_date"] = _compute_next_run_date(update_data.schedule)
        if update_data.end_time is not None:
            update_kwargs["end_time"] = update_data.end_time
        if update_data.enabled is not None:
            update_kwargs["enabled"] = update_data.enabled
        if update_data.on_run_completed is not None:
            update_kwargs["on_run_completed"] = update_data.on_run_completed
        if update_data.metadata is not None:
            update_kwargs["metadata"] = update_data.metadata
        if update_data.input is not None:
            new_payload["input"] = update_data.input.model_dump()
        if update_data.config is not None:
            new_payload["config"] = update_data.config
        if update_data.context is not None:
            new_payload["context"] = update_data.context
        if update_data.interrupt_before is not None:
            new_payload["interrupt_before"] = update_data.interrupt_before
        if update_data.interrupt_after is not None:
            new_payload["interrupt_after"] = update_data.interrupt_after
        if update_data.webhook is not None:
            new_payload["webhook"] = update_data.webhook
        if update_data.stream_mode is not None:
            new_payload["stream_mode"] = update_data.stream_mode
        if update_data.stream_subgraphs is not None:
            new_payload["stream_subgraphs"] = update_data.stream_subgraphs
        if update_data.stream_resumable is not None:
            new_payload["stream_resumable"] = update_data.stream_resumable
        if update_data.durability is not None:
            new_payload["durability"] = update_data.durability

        if new_payload != existing.get("payload", {}):
            update_kwargs["payload"] = new_payload

        row = await store.cron_update(cron_id, **update_kwargs)

    if row is None:
        return None

    cron = _row_to_cron(row)

    scheduler = get_cron_scheduler()

    if cron.enabled:
        _register_cron_job(
            scheduler,
            cron_id=cron_id,
            assistant_id=cron.assistant_id,
            thread_id=cron.thread_id,
            schedule=cron.schedule,
            payload=cron.payload,
            on_run_completed=cron.on_run_completed,
            metadata=cron.metadata,
        )
        scheduler.save_jobs_data()

        await _publish_cron_sync_event("update", cron_id, {
            "cron_id": cron_id,
            "assistant_id": cron.assistant_id,
            "thread_id": cron.thread_id,
            "schedule": cron.schedule,
            "payload": cron.payload,
            "metadata": cron.metadata,
            "end_time": cron.end_time.isoformat() if cron.end_time else None,
            "on_run_completed": cron.on_run_completed,
        })
    else:
        _remove_cron_job_by_id(scheduler, cron_id)
        scheduler.save_jobs_data()

        await _publish_cron_sync_event("delete", cron_id)

    return cron


async def delete_cron(cron_id: str) -> bool:
    async with get_cron_store() as store:
        existing = await store.cron_get(cron_id)
        if existing is None:
            return False
        await store.cron_delete(cron_id)

    scheduler = get_cron_scheduler()
    _remove_cron_job_by_id(scheduler, cron_id)
    scheduler.save_jobs_data()

    await _publish_cron_sync_event("delete", cron_id)
    return True


async def search_crons(
    *,
    assistant_id: str | None = None,
    thread_id: str | None = None,
    enabled: bool | None = None,
    limit: int = 10,
    offset: int = 0,
    sort_by: str | None = None,
    sort_order: str | None = None,
) -> list[CronModel]:
    user_id = await get_user_id()
    effective_user_id = str(user_id) if user_id is not None else None

    async with get_cron_store() as store:
        rows = await store.cron_search(
            assistant_id=assistant_id,
            thread_id=thread_id,
            enabled=enabled,
            user_id=effective_user_id,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    return [_row_to_cron(row) for row in rows]


async def count_crons(
    *,
    assistant_id: str | None = None,
    thread_id: str | None = None,
) -> int:
    user_id = await get_user_id()
    effective_user_id = str(user_id) if user_id is not None else None

    async with get_cron_store() as store:
        return await store.cron_count(
            assistant_id=assistant_id,
            thread_id=thread_id,
            user_id=effective_user_id,
        )


# ── RQ CronScheduler integration (runs in worker process) ────────────────

def _cron_task_func(
    cron_id: str,
    assistant_id: str,
    thread_id: str | None,
    payload_dict: dict,
    on_run_completed: str | None,
):
    from .run_queue_service import run_lg_graph_to_redis_sync
    from ..utils.models import StreamRunRequest

    stream_req = StreamRunRequest(
        assistant_id=assistant_id,
        input=payload_dict.get("input"),
        config=payload_dict.get("config"),
        context=payload_dict.get("context"),
        metadata=payload_dict.get("metadata"),
        stream_mode=payload_dict.get("stream_mode"),
        stream_subgraphs=payload_dict.get("stream_subgraphs", False),
        stream_resumable=payload_dict.get("stream_resumable", False),
        interrupt_before=payload_dict.get("interrupt_before"),
        interrupt_after=payload_dict.get("interrupt_after"),
        multitask_strategy=payload_dict.get("multitask_strategy"),
        webhook=payload_dict.get("webhook"),
        durability=payload_dict.get("durability"),
        on_completion=on_run_completed,
    )

    effective_thread_id = thread_id or str(uuid4())
    temporary = thread_id is None

    run_lg_graph_to_redis_sync(
        run_id=str(uuid4()),
        thread_id=effective_thread_id,
        payload_dict=stream_req.model_dump(mode="json"),
        temporary=temporary,
    )


def sync_crons_to_rq_scheduler():
    scheduler = get_cron_scheduler()

    try:
        from ..registry import _settings as settings
        from ..persistants.cron import AsyncPostgresCron

        conn_string = settings.langgraph_database_uri
        if not conn_string:
            logger.warning("langgraph_database_uri not configured, skipping cron sync")
            return

        async def _load_crons():
            from psycopg.rows import dict_row
            from psycopg import AsyncConnection

            async with await AsyncConnection.connect(
                conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row,
            ) as conn:
                store = AsyncPostgresCron(conn)
                return await store.cron_search(enabled=True, limit=10000)

        rows = asyncio.run(_load_crons())
    except Exception:
        logger.error("Failed to load crons from DB", exc_info=True)
        return

    for row in rows:
        cron_id = row["cron_id"]
        schedule = row["schedule"]
        payload = row.get("payload", {})
        assistant_id = row.get("assistant_id", "")
        thread_id = row.get("thread_id")
        on_run_completed = row.get("on_run_completed")

        scheduler.register(
            func=_cron_task_func,
            queue_name=get_rq_queue().name,
            cron=schedule,
            kwargs={
                "cron_id": cron_id,
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "payload_dict": payload,
                "on_run_completed": on_run_completed,
            },
            meta=_cron_job_meta(cron_id),
        )

    logger.info(f"Loaded {len(rows)} cron jobs from DB into rq CronScheduler")


def listen_cron_sync_events():
    redis_client = get_sync_redis_client()
    scheduler = get_cron_scheduler()

    pubsub = redis_client.pubsub()
    pubsub.subscribe(CRON_SYNC_CHANNEL)

    logger.info(f"Listening for cron sync events on {CRON_SYNC_CHANNEL}")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            data = json.loads(message["data"])
            event = data["event"]
            cron_id = data["cron_id"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Invalid cron sync message: {message['data']}")
            continue

        if event == "create":
            cron_data = data.get("data", {})
            schedule = cron_data.get("schedule")
            if not schedule:
                continue

            scheduler.register(
                func=_cron_task_func,
                queue_name=get_rq_queue().name,
                cron=schedule,
                kwargs={
                    "cron_id": cron_id,
                    "assistant_id": cron_data.get("assistant_id", ""),
                    "thread_id": cron_data.get("thread_id"),
                    "payload_dict": cron_data.get("payload", {}),
                    "on_run_completed": cron_data.get("on_run_completed"),
                },
                meta=_cron_job_meta(cron_id),
            )
            scheduler.save_jobs_data()
            logger.info(f"Cron created: {cron_id}")

        elif event == "update":
            cron_data = data.get("data", {})
            schedule = cron_data.get("schedule")

            _remove_cron_job_by_id(scheduler, cron_id)

            if schedule:
                scheduler.register(
                    func=_cron_task_func,
                    queue_name=get_rq_queue().name,
                    cron=schedule,
                    kwargs={
                        "cron_id": cron_id,
                        "assistant_id": cron_data.get("assistant_id", ""),
                        "thread_id": cron_data.get("thread_id"),
                        "payload_dict": cron_data.get("payload", {}),
                        "on_run_completed": cron_data.get("on_run_completed"),
                    },
                    meta=_cron_job_meta(cron_id),
                )
            scheduler.save_jobs_data()
            logger.info(f"Cron updated: {cron_id}")

        elif event == "delete":
            _remove_cron_job_by_id(scheduler, cron_id)
            scheduler.save_jobs_data()
            logger.info(f"Cron deleted: {cron_id}")
