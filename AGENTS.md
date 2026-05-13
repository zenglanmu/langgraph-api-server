# AGENTS.md

## Project Overview

FastAPI-based LangGraph API server. A standalone alternative to the official LangGraph server (which requires a Go-based LangSmith backend). The `langgraph_api/` package is a library designed to be mounted into any FastAPI app via `setup_api()`.

## Prerequisites

- **Python >=3.12** (enforced in `pyproject.toml`)
- **PostgreSQL** + **Redis** must be running before starting the server
- Environment variables required: `REDIS_URL`, `LANGGRAPH_DATABASE_URI`
- Optional env vars for embedding: `LANGGRAPH_EMBED_MODEL`, `LANGGRAPH_EMBED_DIMENSION`, `LANGGRAPH_EMBED_MODEL_BASE_URL`, `LANGGRAPH_EMBED_MODEL_API_KEY`
- Optional env vars for Langfuse tracing: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## Running

```bash
# Install (uses uv)
uv sync

# Run the example server (loads .env via python-dotenv)
uv run python examples/main.py
# Serves on http://127.0.0.1:2024
```

## Architecture

- **`langgraph_api/`** — the reusable library package
  - `__init__.py` — exports `setup_api`, `GraphRegistry`, `get_graph_store`, `get_graph_checkpointer`; mounts all API routes
  - `api/` — FastAPI routers: `runs`, `threads`, `assistants`, `store`, `crons`
  - `services/` — business logic: `graph_run_service` (SSE streaming), `run_queue_service`, `cron_service`
  - `persistants/` — PostgreSQL persistence extending langgraph's savers: `thread`, `cron`, `assistant`; `setup()` initializes all DB tables
  - `utils/queue_worker.py` — spawns rq worker pool + cron scheduler as child processes
  - `registry.py` — singleton `ApiGlobalSettings` + `GraphRegistry`; serializes config to subprocesses via `snapshot()`/`load()`
- **`examples/`** — example FastAPI app demonstrating integration

## Key Design Decisions

- **Agents must be registered as `CompileGraphCallback`** (a callable returning a `CompiledStateGraph`), not as pre-compiled graphs. Compiled graphs cannot be pickled into rq worker subprocesses. See: https://github.com/langchain-ai/langgraph/issues/3289
- **Default API prefix is `/langgragh_api`** (note: intentional typo in code, `langgragh` not `langgraph`). Override via `include_router_kwargs={"prefix": "/your_prefix"}`.
- **Startup lock**: Only one process in a multi-worker deployment runs DB setup and background workers (Redis-based lock with key `langgraph_api:bg_startup_lock`).
- **Settings propagation**: `ApiGlobalSettings.snapshot()` converts registered graph functions to dotted module paths for subprocess deserialization via `load()`. `user_id_callback` is intentionally excluded from serialization since it may depend on FastAPI request context.
- **Langfuse config**: Set via `os.environ` (not passed programmatically) — langfuse reads env vars directly.
- **Embedding config**: Uses `init_embeddings` with `provider='openai'` and HNSW index. ANN index does not support >2000 dimensions.

## No Tests / CI / Lint Config

No test framework, linter, type checker, or CI workflows are configured. There is no `pytest`, `ruff`, `mypy`, or similar setup.
