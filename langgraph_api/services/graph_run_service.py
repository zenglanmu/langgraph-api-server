'''
从Redis stream中轮询run事件，以SSE格式返回给客户端。
支持通过run_id和last_event_id进行断线重连。
'''
import asyncio
import json
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fastapi.sse import ServerSentEvent

from .run_queue_service import (
    get_last_run_stream_seq,
    get_run_status,
    list_run_stream_events,
    normalize_after_seq,
    TERMINAL_RUN_STATUSES,
)

_TOOL_EVENT_MAP = {
    "tool-started": "on_tool_start",
    "tool-output-delta": "on_tool_event",
    "tool-finished": "on_tool_end",
    "tool-error": "on_tool_error",
}


def _translate_tools_payload(payload: dict) -> dict:
    inner = payload if "event" in payload else payload.get("data", payload)
    ev = inner.get("event", "")
    if ev not in _TOOL_EVENT_MAP:
        return payload
    translated = {**inner, "event": _TOOL_EVENT_MAP[ev]}
    if "tool_name" in translated:
        translated["name"] = translated.pop("tool_name")
    if "delta" in translated:
        translated["data"] = translated.pop("delta")
    if "message" in translated and ev == "tool-error":
        translated["error"] = translated.pop("message")
    if "data" in payload and inner is not payload:
        return {**payload, "data": translated}
    return translated


SSE_HEARTBEAT_SECONDS = int(os.getenv("RUN_SSE_HEARTBEAT_SECONDS", "15"))
SSE_MAX_CONNECTION_MINUTES = int(os.getenv("RUN_SSE_MAX_CONNECTION_MINUTES", "30"))
SSE_POLL_INTERVAL_SECONDS = float(os.getenv("RUN_SSE_POLL_INTERVAL_SECONDS", "0.05"))


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


async def stream_agent_run_events(
    *,
    run_id: str,
    after_seq: str | int | None = None,
) -> AsyncIterator[ServerSentEvent]:
    started_at = _utc_now()
    last_seq = normalize_after_seq(after_seq)

    try:
        while True:
            try:
                events = await list_run_stream_events(run_id, after_seq=last_seq, limit=200)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                from traceback import print_exception
                print_exception(e)
                yield ServerSentEvent(data={"run_id": run_id, "message": str(e)}, event="values", id=last_seq)
                return

            for event in events:
                seq = event["seq"]
                last_seq = seq
                payload = event.get("payload") or {}
                if event.get("event_type") == "tools":                    
                    # fix to frontend langgraph sdk wanted format
                    payload = _translate_tools_payload(payload)
                yield ServerSentEvent(
                    data=payload,
                    event=event.get("event_type") or "values",
                    id=last_seq
                )

            run_status = await get_run_status(run_id)
            current_status = run_status.get("status") if run_status else None

            if current_status in TERMINAL_RUN_STATUSES and not events:
                terminal_seq = last_seq
                if terminal_seq in {"", "0", "0-0"}:
                    terminal_seq = await get_last_run_stream_seq(run_id)

                close_data = {
                    "run_id": run_id,
                    "status": current_status,
                    "last_seq": terminal_seq,
                }
                if run_status and run_status.get("error_message"):
                    close_data["error_message"] = run_status["error_message"]

                yield ServerSentEvent(data=close_data, event='values', id=last_seq)
                return
           
            await asyncio.sleep(SSE_POLL_INTERVAL_SECONDS)

    except asyncio.CancelledError:
        return
