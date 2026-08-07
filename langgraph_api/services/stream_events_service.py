"""
Thread 级 SSE 事件总线 —— 服务 POST /threads/{thread_id}/stream/events。

Redis key 设计：
- 环形缓冲: langgraph:thread:events:{thread_id}  (Redis List, LTRIM -1000 -1)
- seq 计数器: langgraph:thread:events:seq:{thread_id}  (INCR)
- Pub/Sub: langgraph:thread:events:pub:{thread_id}

run_queue_service 在执行 graph 时调用 publish_thread_event / publish_lifecycle_event
将事件写入上述结构；本模块的 stream_thread_events 生成器先重放 since..now，再订阅
Pub/Sub 实时推送，按 channels/namespaces 过滤后以 SSE data: 行 yield。
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from logging import getLogger
from typing import Any

from fastapi.sse import ServerSentEvent
from uuid_utils import uuid7

from ..utils.queue_worker import get_redis_client

logger = getLogger(__name__)


THREAD_EVENTS_BUFFER_TTL_SECONDS = 7200
THREAD_EVENTS_BUFFER_MAXLEN = 1000

STREAM_MODE_TO_METHOD = {
    "values": "values",
    "messages": "messages",
    "updates": "updates",
    "tools": "tools",
    "custom": "custom",
    "checkpoints": "checkpoints",
    "tasks": "tasks",
}


def _buffer_key(thread_id: str) -> str:
    return f"langgraph:thread:events:{thread_id}"


def _seq_key(thread_id: str) -> str:
    return f"langgraph:thread:events:seq:{thread_id}"


def _pubsub_key(thread_id: str) -> str:
    return f"langgraph:thread:events:pub:{thread_id}"


# ── 发布端（run_queue_service 调用）─────────────────────────────────────

async def publish_thread_event(
    thread_id: str,
    run_id: str,
    stream_mode: str,
    event_data: Any,
    ns: list[str] | None = None,
) -> None:
    """run_queue_service 调用：发布一个 thread 级事件。

    失败只 log warning，不影响主流程。
    """
    try:
        redis = await get_redis_client()
        seq = await redis.incr(_seq_key(thread_id))
        method = STREAM_MODE_TO_METHOD.get(stream_mode, stream_mode)
        event = {
            "type": "event",
            "event_id": str(uuid7()),
            "seq": int(seq),
            "method": method,
            "params": {
                "data": event_data,
                "run_id": run_id,
                "namespace": ns or [],
            },
        }
        payload = json.dumps(event, ensure_ascii=False, default=str)
        await redis.rpush(_buffer_key(thread_id), payload)
        await redis.ltrim(_buffer_key(thread_id), -THREAD_EVENTS_BUFFER_MAXLEN, -1)
        await redis.expire(_buffer_key(thread_id), THREAD_EVENTS_BUFFER_TTL_SECONDS)
        await redis.publish(_pubsub_key(thread_id), payload)
    except Exception as e:
        logger.warning(
            f"publish_thread_event failed for thread {thread_id}: {e}",
            exc_info=True,
        )


async def publish_lifecycle_event(
    thread_id: str,
    run_id: str,
    lifecycle_event: str,
) -> None:
    """发布 lifecycle 事件: started/running/completed/failed/interrupted"""
    try:
        redis = await get_redis_client()
        seq = await redis.incr(_seq_key(thread_id))
        event = {
            "type": "event",
            "event_id": str(uuid7()),
            "seq": int(seq),
            "method": "lifecycle",
            "params": {
                "data": {"event": lifecycle_event, "run_id": run_id},
                "namespace": [],
            },
        }
        payload = json.dumps(event, ensure_ascii=False, default=str)
        await redis.rpush(_buffer_key(thread_id), payload)
        await redis.ltrim(_buffer_key(thread_id), -THREAD_EVENTS_BUFFER_MAXLEN, -1)
        await redis.expire(_buffer_key(thread_id), THREAD_EVENTS_BUFFER_TTL_SECONDS)
        await redis.publish(_pubsub_key(thread_id), payload)
    except Exception as e:
        logger.warning(
            f"publish_lifecycle_event failed for thread {thread_id}: {e}",
            exc_info=True,
        )


async def publish_input_requested_event(
    thread_id: str,
    run_id: str,
    interrupt_id: str,
    payload: Any,
) -> None:
    """发布中断请求事件"""
    try:
        redis = await get_redis_client()
        seq = await redis.incr(_seq_key(thread_id))
        event = {
            "type": "event",
            "event_id": str(uuid7()),
            "seq": int(seq),
            "method": "input.requested",
            "params": {
                "data": {
                    "interrupt_id": interrupt_id,
                    "payload": payload,
                    "run_id": run_id,
                },
                "namespace": [],
            },
        }
        sse_payload = json.dumps(event, ensure_ascii=False, default=str)
        await redis.rpush(_buffer_key(thread_id), sse_payload)
        await redis.ltrim(_buffer_key(thread_id), -THREAD_EVENTS_BUFFER_MAXLEN, -1)
        await redis.expire(_buffer_key(thread_id), THREAD_EVENTS_BUFFER_TTL_SECONDS)
        await redis.publish(_pubsub_key(thread_id), sse_payload)
    except Exception as e:
        logger.warning(
            f"publish_input_requested_event failed for thread {thread_id}: {e}",
            exc_info=True,
        )


# ── 订阅端（stream/events 端点调用）─────────────────────────────────────

def _match_channels(event: dict, channels: list[str]) -> bool:
    """channel 过滤：method == channel 或 (method=="custom" 且 channel startswith "custom:")"""
    if not channels:
        return True
    method = event.get("method") or ""
    for ch in channels:
        if ch == method:
            return True
        # "custom:xxx" 形式：method == "custom" 且 channel 以 "custom:" 开头
        if ch.startswith("custom:") and method == "custom":
            return True
        # lifecycle / input.requested 等控制类事件总是放行（前端依赖）
        if method in ("lifecycle", "input.requested") and ch in ("lifecycle", "input"):
            return True
    return False


def _match_namespaces(
    event: dict,
    namespaces: list[list[str]] | None,
    depth: int | None = None,
) -> bool:
    """namespace + depth 过滤，与 SDK 的 namespaceMatches 语义一致。

    - namespaces 为 None/空 → 放行
    - event 的 namespace 从 params.namespace 获取（list[str]）
    - 前缀匹配：event_ns[:len(prefix)] == prefix
    - depth 过滤：len(event_ns) - len(prefix) <= depth
    """
    if not namespaces:
        return True
    params = event.get("params") or {}
    event_ns = params.get("namespace") if isinstance(params, dict) else None
    if not isinstance(event_ns, list):
        event_ns = []
    for prefix in namespaces:
        # 前缀匹配
        if event_ns[: len(prefix)] != prefix:
            continue
        # depth 过滤
        if depth is not None:
            if len(event_ns) - len(prefix) > depth:
                continue
        return True
    return False


async def _read_buffer_events(
    thread_id: str,
    since: int | None,
) -> list[dict]:
    """从 Redis List 读取缓冲事件，过滤 since 之后的。"""
    redis = await get_redis_client()
    raw_list = await redis.lrange(_buffer_key(thread_id), 0, -1)
    events: list[dict] = []
    for raw in raw_list:
        try:
            ev = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        seq = ev.get("seq")
        if since is not None and isinstance(seq, int) and seq <= since:
            continue
        events.append(ev)
    return events


async def stream_thread_events(
    thread_id: str,
    channels: list[str],
    namespaces: list[list[str]] | None,
    depth: int | None,
    since: int | None,
) -> AsyncIterator[ServerSentEvent]:
    """POST /threads/{thread_id}/stream/events 的生成器，yield ServerSentEvent。

    为避免"先读 List 再订阅 Pub/Sub"产生的发布-订阅竞态丢事件，
    采用"先订阅 Pub/Sub，再读 List 缓冲，用 seq 去重"的顺序：
      1. 先 subscribe，确保订阅之后发布的事件都能被 Pub/Sub 捕获
      2. 再 lrange 重放历史缓冲（含订阅前/订阅瞬间发布的事件）
      3. 实时阶段用 seq 去重，跳过已通过缓冲重放过的重复事件
    """
    # 先订阅 Pub/Sub，确保订阅之后发布的事件不会丢失
    redis = await get_redis_client()
    pubsub = redis.pubsub()
    await pubsub.subscribe(_pubsub_key(thread_id))

    # 再读 List 缓冲，重放 since..now 的历史事件
    buffered = await _read_buffer_events(thread_id, since)
    last_seq = since or 0
    for ev in buffered:
        if not _match_channels(ev, channels):
            continue
        if not _match_namespaces(ev, namespaces, depth):
            continue
        seq = ev.get("seq")
        if isinstance(seq, int) and seq > last_seq:
            last_seq = seq
        yield ServerSentEvent(
            data=ev,
            event=ev.get("method") or "message",
            id=str(seq) if seq is not None else None,
        )

    # Pub/Sub 实时阶段：用 seq 去重，跳过已通过缓冲重放的事件
    try:
        # 心跳计数，避免长时间无事件时连接被中间代理断开
        heartbeat_counter = 0
        while True:
            try:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=15.0,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"pubsub read error for thread {thread_id}: {e}")
                await asyncio.sleep(1.0)
                continue

            if msg is None:
                heartbeat_counter += 1
                if heartbeat_counter % 4 == 0:
                    yield ServerSentEvent(data="", event="ping")
                continue

            heartbeat_counter = 0
            data = msg.get("data")
            if not data or not isinstance(data, str):
                continue
            try:
                ev = json.loads(data)
            except (TypeError, json.JSONDecodeError):
                continue
            seq = ev.get("seq")
            if isinstance(seq, int) and isinstance(last_seq, int) and seq <= last_seq:
                continue
            if isinstance(seq, int):
                last_seq = seq
            if not _match_channels(ev, channels):
                continue
            if not _match_namespaces(ev, namespaces, depth):
                continue
            yield ServerSentEvent(
                data=ev,
                event=ev.get("method") or "message",
                id=str(seq) if seq is not None else None,
            )
    finally:
        try:
            await pubsub.unsubscribe(_pubsub_key(thread_id))
            await pubsub.close()
        except Exception:
            pass