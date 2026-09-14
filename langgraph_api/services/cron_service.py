'''
Cron job management service.

Persists cron records in PostgreSQL (via AsyncPostgresCron) and
schedules actual execution through an in-process asyncio scheduler.

Design:
- API layer: create/update/delete crons -> persist to DB -> notify schedulers
  via Redis pub/sub
- Every process runs a lightweight asyncio scheduler which loads all enabled
  crons from DB on startup, then applies runtime sync events via Redis pub/sub
- Multi-process deployments deduplicate fires via a Redis SETNX claim
  (each fire time is only executed once across all processes)
'''
import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from uuid import uuid4

from croniter import croniter
from uuid_utils import uuid7

from ..registry import get_cron_store, get_user_id, _settings
from ..utils.models import CronCreate, CronUpdate, Cron as CronModel
from ..utils.queue_worker import get_arq_pool, get_redis_client

logger = logging.getLogger(__name__)

CRON_SYNC_CHANNEL = "langgraph:cron:sync"

CRON_SCHEDULER_TICK_SECONDS = float(os.getenv("CRON_SCHEDULER_TICK_SECONDS", "30"))
CRON_FIRE_DEDUP_TTL_SECONDS = 600


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

    if cron.enabled:
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
        await _publish_cron_sync_event("delete", cron_id)

    return cron


async def delete_cron(cron_id: str) -> bool:
    async with get_cron_store() as store:
        existing = await store.cron_get(cron_id)
        if existing is None:
            return False
        await store.cron_delete(cron_id)

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


# ── In-process cron scheduler (asyncio tasks) ────────────────────────────

# cron_id -> job details, {assistant_id, thread_id, schedule, payload,
#                          on_run_completed, end_time, next_run}
_cron_jobs: dict[str, dict] = {}
_scheduler_tasks: list[asyncio.Task] = []


def _parse_end_time(value) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _register_cron_job(
    *,
    cron_id: str,
    assistant_id: str,
    thread_id: str | None,
    schedule: str,
    payload: dict,
    on_run_completed: str | None,
    end_time,
) -> None:
    now = datetime.now(tz=UTC)
    try:
        next_run = croniter(schedule, now).get_next(datetime)
    except Exception:
        logger.warning(f"Invalid cron schedule for {cron_id}: {schedule}")
        return
    _cron_jobs[cron_id] = {
        "assistant_id": assistant_id,
        "thread_id": thread_id,
        "schedule": schedule,
        "payload": payload or {},
        "on_run_completed": on_run_completed,
        "end_time": _parse_end_time(end_time),
        "next_run": next_run,
    }


def _apply_cron_sync_event(event: str, cron_id: str, data: dict | None) -> None:
    if event == "delete":
        _cron_jobs.pop(cron_id, None)
        return

    if not data:
        return
    schedule = data.get("schedule")
    if not schedule:
        return
    _register_cron_job(
        cron_id=cron_id,
        assistant_id=data.get("assistant_id", ""),
        thread_id=data.get("thread_id"),
        schedule=schedule,
        payload=data.get("payload", {}),
        on_run_completed=data.get("on_run_completed"),
        end_time=data.get("end_time"),
    )


async def _load_crons_from_db() -> None:
    '''启动时从 DB 加载所有 enabled 的 cron 到内存'''
    now = datetime.now(tz=UTC)
    async with get_cron_store() as store:
        rows = await store.cron_search(enabled=True, limit=10000)
    for row in rows:
        _register_cron_job(
            cron_id=row["cron_id"],
            assistant_id=row.get("assistant_id", ""),
            thread_id=row.get("thread_id"),
            schedule=row["schedule"],
            payload=row.get("payload", {}),
            on_run_completed=row.get("on_run_completed"),
            end_time=row.get("end_time"),
        )
    logger.info(f"Loaded {len(rows)} cron jobs from DB into in-process scheduler")


async def _fire_cron_job(cron_id: str, job: dict) -> None:
    '''将到期的 cron 作为 ARQ job 入队执行'''
    payload_dict = {**job["payload"], "assistant_id": job["assistant_id"]}
    if job["on_run_completed"]:
        payload_dict["on_completion"] = job["on_run_completed"]

    thread_id = job["thread_id"] or str(uuid7())
    run_id = str(uuid7())

    pool = await get_arq_pool()
    await pool.enqueue_job(
        "run_graph_job",
        run_id=run_id,
        thread_id=thread_id,
        payload_dict=payload_dict,
        temporary=job["thread_id"] is None,
        _job_id=run_id,
    )
    logger.info(f"Cron {cron_id} fired: run_id={run_id}, thread_id={thread_id}")


async def _claim_and_fire_cron(cron_id: str, job: dict, fire_time: datetime) -> None:
    '''多进程部署时通过 Redis SETNX 保证同一触发时间全局只执行一次'''
    redis = await get_redis_client()
    minute_slot = int(fire_time.timestamp()) // 60
    dedup_key = f"langgraph:cron:fire:{cron_id}:{minute_slot}"
    if not await redis.set(dedup_key, "1", nx=True, ex=CRON_FIRE_DEDUP_TTL_SECONDS):
        return
    await _fire_cron_job(cron_id, job)


async def _scheduler_tick() -> None:
    now = datetime.now(tz=UTC)
    for cron_id, job in list(_cron_jobs.items()):
        end_time = job["end_time"]
        while job["next_run"] <= now:
            fire_time = job["next_run"]
            job["next_run"] = croniter(job["schedule"], now).get_next(datetime)
            if end_time is not None and fire_time > end_time:
                _cron_jobs.pop(cron_id, None)
                break
            try:
                await _claim_and_fire_cron(cron_id, job, fire_time)
            except Exception:
                logger.exception(f"Failed to fire cron {cron_id}")
                break


async def _scheduler_loop() -> None:
    while True:
        await asyncio.sleep(CRON_SCHEDULER_TICK_SECONDS)
        try:
            await _scheduler_tick()
        except Exception:
            logger.exception("Cron scheduler tick failed")


async def _cron_pubsub_listener() -> None:
    while True:
        redis = await get_redis_client()
        pubsub = redis.pubsub()
        try:
            await pubsub.subscribe(CRON_SYNC_CHANNEL)
            logger.info(f"Listening for cron sync events on {CRON_SYNC_CHANNEL}")
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                    event = data["event"]
                    cron_id = data["cron_id"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    logger.warning(f"Invalid cron sync message: {message['data']}")
                    continue
                _apply_cron_sync_event(event, cron_id, data.get("data"))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Cron sync listener crashed, retrying in 5 seconds")
            await asyncio.sleep(5)
        finally:
            try:
                await pubsub.close()
            except Exception:
                pass


async def start_cron_scheduler() -> None:
    '''在当前进程的事件循环中启动 cron scheduler（asyncio tasks）'''
    global _scheduler_tasks
    if _scheduler_tasks:
        return
    await _load_crons_from_db()
    _scheduler_tasks = [
        asyncio.create_task(_scheduler_loop()),
        asyncio.create_task(_cron_pubsub_listener()),
    ]
    logger.info("[*] langgraph api cron scheduler started")


async def stop_cron_scheduler() -> None:
    for task in _scheduler_tasks:
        task.cancel()
    for task in _scheduler_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
    _scheduler_tasks.clear()
