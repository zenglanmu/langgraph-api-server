"""
Thread Command Handler —— 处理 POST /threads/{thread_id}/commands 的所有 method。

对应 langgraph-sdk v1.9.28 thread-centric 协议：
- run.start: 启动一次 run（懒建线程）
- input.respond: 响应中断（构造 Command(resume/goto/update)）
- input.inject: 直接向 checkpointer 注入消息
- state.get: 读取线程状态
- agent.getTree: 从 state.tasks 构建任务树
- state.listCheckpoints: 列出 checkpoint
- state.fork: 从 checkpoint 分叉新 run
"""
from __future__ import annotations

import json
from logging import getLogger
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from ..registry import (
    GraphRegistry,
    _settings,
    get_graph_checkpointer,
    get_graph_conn,
    get_thread_store,
    get_user_id,
)
from ..utils.models import (
    Command,
    InputModel,
    StreamRunRequest,
    ThreadState,
    convert_state_snapshot_to_thread_state,
)
from .run_queue_service import cancel_run, enqueue_run

logger = getLogger(__name__)


class ThreadCommandHandler:
    """处理 POST /threads/{thread_id}/commands 的所有 method"""

    def __init__(self, thread_id: str):
        self.thread_id = thread_id

    # ── dispatch ─────────────────────────────────────────────────────────

    async def dispatch(self, cmd_id: int, method: str, params: dict) -> dict | None:
        handler = self._handlers.get(method)
        if handler is None:
            return self._error(cmd_id, "unknown_command", f"Unknown method: {method}")
        try:
            result = await handler(self, params)
            if result is None:
                return None  # 204
            return self._success(cmd_id, result)
        except _CommandError as e:
            return self._error(cmd_id, e.code, e.message)
        except Exception as e:
            logger.exception(f"Command {method} failed")
            return self._error(cmd_id, "unknown_error", str(e))

    # ── response helpers ─────────────────────────────────────────────────

    @staticmethod
    def _success(cmd_id: int, result: dict | None) -> dict:
        return {
            "type": "success",
            "id": cmd_id,
            "result": result or {},
            "meta": {"applied_through_seq": 0},
        }

    @staticmethod
    def _error(cmd_id: int | None, code: str, message: str) -> dict:
        return {
            "type": "error",
            "id": cmd_id,
            "error": code,
            "message": message,
            "stacktrace": None,
            "meta": {},
        }

    # ── thread helpers ───────────────────────────────────────────────────

    async def _ensure_thread(self) -> dict:
        """懒建线程：不存在则创建。返回 thread row。"""
        uid = await get_user_id()
        effective_user_id = str(uid) if uid is not None else None
        async with get_thread_store() as store:
            row = await store.thread_get(
                self.thread_id, user_id=effective_user_id
            )
            if row is None:
                row = await store.thread_put(
                    self.thread_id,
                    metadata={},
                    user_id=effective_user_id,
                )
        return row

    async def _cancel_running_runs(self) -> None:
        """取消 thread 最近处于 running/pending 的 run（multitask_strategy=interrupt 语义）。"""
        try:
            async with get_thread_store() as store:
                last = await store.run_get_last(self.thread_id)
            if last is None:
                return
            status = last.get("status")
            if status in ("running", "pending"):
                await cancel_run(last["run_id"], action="interrupt", wait=False)
        except Exception:
            logger.warning(
                f"Failed to cancel running runs for thread {self.thread_id}",
                exc_info=True,
            )

    # ── method: run.start ───────────────────────────────────────────────

    async def _handle_run_start(self, params: dict) -> dict:
        await self._ensure_thread()

        assistant_id = params.get("assistant_id")
        if not assistant_id:
            raise _CommandError("invalid_argument", "assistant_id is required")

        # 构造 StreamRunRequest
        input_data = params.get("input")
        config = params.get("config")
        metadata = params.get("metadata")
        stream_mode = params.get("stream_mode") or ["values", "messages"]
        command = params.get("command")

        # input 可能是 dict，包装为 InputModel
        input_model: InputModel | None = None
        if input_data is not None:
            if isinstance(input_data, InputModel):
                input_model = input_data
            else:
                input_model = InputModel.model_validate(input_data)

        command_model: Command | None = None
        if command is not None:
            if isinstance(command, Command):
                command_model = command
            else:
                command_model = Command.model_validate(command)

        payload = StreamRunRequest(
            assistant_id=assistant_id,
            input=input_model,
            command=command_model,
            config=config,
            metadata=metadata,
            stream_mode=stream_mode,
            multitask_strategy=params.get("multitask_strategy", "interrupt"),
            if_not_exists=params.get("if_not_exists", "create"),
        )

        run_id = await enqueue_run(
            thread_id=self.thread_id,
            payload=payload,
            temporary=False,
        )
        return {"run_id": run_id}

    # ── method: input.respond ───────────────────────────────────────────

    async def _handle_input_respond(self, params: dict) -> dict:
        await self._ensure_thread()

        # 支持单条与批量
        if "responses" in params:
            responses = params.get("responses") or []
        else:
            responses = [params]

        results: list[dict[str, Any]] = []
        for resp in responses:
            run_id = await self._enqueue_respond(resp)
            results.append({"run_id": run_id})
        if len(results) == 1:
            return results[0]
        return {"runs": results}

    async def _enqueue_respond(self, resp: dict) -> str:
        # 先取消正在运行的 run
        await self._cancel_running_runs()

        assistant_id = resp.get("assistant_id")
        if not assistant_id:
            # 尝试从最近 run 取 assistant_id
            async with get_thread_store() as store:
                last = await store.run_get_last(self.thread_id)
            if last is None or not last.get("assistant_id"):
                raise _CommandError(
                    "invalid_argument",
                    "assistant_id is required and no prior run to infer from",
                )
            assistant_id = last["assistant_id"]

        response = resp.get("response")
        update = resp.get("update")
        goto = resp.get("goto")

        command = Command(resume=response, update=update, goto=goto)

        namespace = resp.get("namespace") or []
        checkpoint_ns = ":".join(namespace) if namespace else ""

        config = resp.get("config") or {}
        if checkpoint_ns:
            config.setdefault("configurable", {})["checkpoint_ns"] = checkpoint_ns

        metadata = resp.get("metadata")

        payload = StreamRunRequest(
            assistant_id=assistant_id,
            input=None,
            command=command,
            config=config,
            metadata=metadata,
            stream_mode=["values", "messages"],
            multitask_strategy="interrupt",
            if_not_exists="create",
        )

        run_id = await enqueue_run(
            thread_id=self.thread_id,
            payload=payload,
            temporary=False,
        )
        return run_id

    # ── method: input.inject ────────────────────────────────────────────

    async def _handle_input_inject(self, params: dict) -> dict:
        await self._ensure_thread()

        message = params.get("message")
        if message is None:
            raise _CommandError("invalid_argument", "message is required")

        namespace = params.get("namespace") or []
        checkpoint_ns = ":".join(namespace) if namespace else ""

        configurable: dict[str, Any] = {"thread_id": self.thread_id}
        if checkpoint_ns:
            configurable["checkpoint_ns"] = checkpoint_ns
        checkpoint_id = params.get("checkpoint_id")
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id

        config = RunnableConfig(configurable=configurable)
        async with get_graph_checkpointer() as checkpointer:
            await checkpointer.aupdate_state(
                config,
                {"messages": [message]},
            )
        return {}

    # ── method: state.get ──────────────────────────────────────────────

    async def _handle_state_get(self, params: dict) -> dict:
        from ..api.threads import _get_thread_state_via_graph

        namespace = params.get("namespace") or []
        checkpoint_ns = ":".join(namespace) if namespace else None
        checkpoint_id = params.get("checkpoint_id")

        state = await _get_thread_state_via_graph(
            self.thread_id,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
            subgraphs=False,
        )
        if state is None:
            raise _CommandError("no_such_thread", f"{self.thread_id} not found")

        # keys 过滤
        keys = params.get("keys")
        result = state.model_dump(mode="json")
        if keys:
            values = result.get("values", {})
            result["values"] = {k: values.get(k) for k in keys if k in values}
        return result

    # ── method: agent.getTree ───────────────────────────────────────────

    async def _handle_agent_get_tree(self, params: dict) -> dict:
        from ..api.threads import _get_thread_state_via_graph

        state = await _get_thread_state_via_graph(self.thread_id)
        if state is None:
            raise _CommandError("no_such_thread", f"{self.thread_id} not found")

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        # 根节点
        root_id = state.checkpoint.checkpoint_id or self.thread_id
        nodes.append({
            "id": root_id,
            "name": "root",
            "checkpoint": state.checkpoint.model_dump(mode="json"),
        })

        for task in state.tasks:
            task_id = task.id
            nodes.append({
                "id": task_id,
                "name": task.name,
                "checkpoint": task.checkpoint.model_dump(mode="json") if task.checkpoint else None,
            })
            edges.append({"from": root_id, "to": task_id})

        # 尝试取最近 run_id
        run_id = None
        try:
            async with get_thread_store() as store:
                last = await store.run_get_last(self.thread_id)
            if last:
                run_id = last.get("run_id")
        except Exception:
            pass

        return {"nodes": nodes, "edges": edges, "run_id": run_id}

    # ── method: state.listCheckpoints ──────────────────────────────────

    async def _handle_state_list_checkpoints(self, params: dict) -> dict:
        namespace = params.get("namespace") or []
        checkpoint_ns = ":".join(namespace) if namespace else ""
        limit = params.get("limit", 10)
        before = params.get("before")

        configurable: dict[str, Any] = {"thread_id": self.thread_id}
        if checkpoint_ns:
            configurable["checkpoint_ns"] = checkpoint_ns

        config = RunnableConfig(configurable=configurable)
        before_config = None
        if before:
            before_config = RunnableConfig(
                configurable={"thread_id": self.thread_id, "checkpoint_id": before}
            )

        checkpoints: list[dict[str, Any]] = []
        async with get_graph_conn() as conn:
            async with get_graph_checkpointer(conn=conn) as checkpointer:
                async for ct in checkpointer.alist(
                    config, before=before_config, limit=limit, filter=None
                ):
                    checkpoints.append({
                        "checkpoint_id": ct.checkpoint.id,
                        "checkpoint_ns": ct.config.get("configurable", {}).get("checkpoint_ns", ""),
                        "parent_checkpoint_id": (
                            ct.parent_config.get("configurable", {}).get("checkpoint_id")
                            if ct.parent_config else None
                        ),
                        "metadata": ct.metadata,
                        "ts": ct.checkpoint.get("ts"),
                    })
        return {"checkpoints": checkpoints}

    # ── method: state.fork ──────────────────────────────────────────────

    async def _handle_state_fork(self, params: dict) -> dict:
        checkpoint_id = params.get("checkpoint_id")
        if not checkpoint_id:
            raise _CommandError("invalid_argument", "checkpoint_id is required")

        assistant_id = params.get("assistant_id")
        if not assistant_id:
            async with get_thread_store() as store:
                last = await store.run_get_last(self.thread_id)
            if last is None or not last.get("assistant_id"):
                raise _CommandError(
                    "invalid_argument",
                    "assistant_id is required and no prior run to infer from",
                )
            assistant_id = last["assistant_id"]

        namespace = params.get("namespace") or []
        checkpoint_ns = ":".join(namespace) if namespace else ""

        configurable: dict[str, Any] = {
            "thread_id": self.thread_id,
            "checkpoint_id": checkpoint_id,
        }
        if checkpoint_ns:
            configurable["checkpoint_ns"] = checkpoint_ns

        config = RunnableConfig(configurable=configurable)
        payload = StreamRunRequest(
            assistant_id=assistant_id,
            input=None,
            config={"configurable": configurable},
            metadata=params.get("metadata"),
            stream_mode=["values", "messages"],
            multitask_strategy="interrupt",
            if_not_exists="create",
        )

        run_id = await enqueue_run(
            thread_id=self.thread_id,
            payload=payload,
            temporary=False,
        )
        return {"run_id": run_id}

    # ── handler 表 ──────────────────────────────────────────────────────

    _handlers = {
        "run.start": _handle_run_start,
        "input.respond": _handle_input_respond,
        "input.inject": _handle_input_inject,
        "state.get": _handle_state_get,
        "agent.getTree": _handle_agent_get_tree,
        "state.listCheckpoints": _handle_state_list_checkpoints,
        "state.fork": _handle_state_fork,
    }


class _CommandError(Exception):
    """携带 ErrorCode 的内部异常"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message