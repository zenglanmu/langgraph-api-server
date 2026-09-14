# AGENTS.md

## Project Overview

FastAPI-based LangGraph API server with a React frontend. A standalone alternative to the official LangGraph server (which requires a Go-based LangSmith backend). The `langgraph_api/` package is a library designed to be mounted into any FastAPI app via `setup_api()`.

## Two Packages

| | Backend | Frontend |
|---|---|---|
| **Dir** | `langgraph_api/` | `frontend/` |
| **Stack** | Python >=3.12, FastAPI, uv | React 19, TypeScript, Vite, TailwindCSS 4, pnpm |
| **Package mgr** | `uv` | `pnpm` (v10.5.1) |
| **Entry** | `examples/main.py` (runs on port 2024) | `frontend/src/App.tsx` (dev on port 5173) |
| **Lint** | none | `pnpm lint` (eslint), `pnpm format:check` (prettier) |
| **Build** | — | `pnpm build` (`tsc -b && vite build`), output to `frontend/dist/` |

## Running

```bash
# Backend only
uv sync
uv run python examples/main.py          # http://localhost:2024

# Frontend only (from frontend/)
pnpm install
pnpm dev                                # http://localhost:5173

# Both together (from repo root)
python dev.py                            # starts backend + frontend, auto-copies .env.example → .env
```

## Frontend Details

- **App name**: `agent-chat-ui` — a LangGraph agent chat interface
- **Vite proxy**: `/api` → `http://localhost:2024` (strips `/api` prefix), so frontend requests go to the backend
- **Env vars** (in `frontend/.env`): `VITE_API_URL` (default `http://localhost:2024`), `VITE_ASSISTANT_ID` (default `agent`), `VITE_AUTH_SCHEME`
- **Path alias**: `@` → `frontend/src/`
- **Key providers**: `StreamProvider`, `ThreadProvider`, `ArtifactProvider` wrap `<Thread />` in `App.tsx`
- **LangGraph SDK**: uses `@langchain/langgraph-sdk` to communicate with the backend
- **UI lib**: Radix UI primitives + shadcn/ui components in `src/components/ui/`
- **Prettier**: uses `prettier-plugin-tailwindcss` (must run via `pnpm format` not plain prettier)

## Prerequisites

- **Python >=3.12**, **PostgreSQL**, **Redis** must be running before starting the backend
- Required env vars: `REDIS_URL`, `LANGGRAPH_DATABASE_URI`
- Optional env vars for embedding: `LANGGRAPH_EMBED_MODEL`, `LANGGRAPH_EMBED_DIMENSION`, `LANGGRAPH_EMBED_MODEL_BASE_URL`, `LANGGRAPH_EMBED_MODEL_API_KEY`
- Optional env vars for Langfuse tracing: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## Architecture

- **`langgraph_api/`** — the reusable library package
  - `__init__.py` — exports `setup_api`, `GraphRegistry`, `get_graph_store`, `get_graph_checkpointer`; mounts all API routes
  - `api/` — FastAPI routers: `runs`, `threads`, `assistants`, `store`, `crons`
  - `services/` — business logic: `graph_run_service` (SSE streaming), `run_queue_service`, `cron_service`
  - `persistants/` — PostgreSQL persistence extending langgraph's savers: `thread`, `cron`, `assistant`; `setup()` initializes all DB tables
  - `utils/queue_worker.py` — in-process ARQ worker + DB setup guard (`setup_database_once`)
  - `registry.py` — singleton `ApiGlobalSettings` + `GraphRegistry`
- **`frontend/`** — React chat UI
  - `src/providers/` — `Stream`, `Thread`, `client` (LangGraph SDK client setup)
  - `src/components/thread/` — main chat thread view, message types (AI/human/tool-calls), markdown rendering, agent-inbox interrupt handling
  - `src/components/ui/` — shadcn/ui primitives
- **`examples/`** — example FastAPI app + agent registrations
- **`dev.py`** — starts both backend and frontend with SIGINT cleanup

## Key Design Decisions

- **Agents must be registered as `CompileGraphCallback`** (a callable returning a `CompiledStateGraph`), not as pre-compiled graphs. See: https://github.com/langchain-ai/langgraph/issues/3289
- **Default API prefix is `/langgragh_api`** (note: intentional typo in code, `langgragh` not `langgraph`). Override via `include_router_kwargs={"prefix": "/your_prefix"}`.
- **DB setup guard**: In multi-worker deployments, DB setup runs exactly once — the first process acquires a Redis lock (key `langgraph_api:bg_setup_lock`), others wait for completion. If the lock owner crashes, the lock expires and another process retries.
- **In-process ARQ worker**: Every process runs an ARQ worker (asyncio task). Jobs are enqueued via `enqueue_run()` (`run_graph_job`, idempotent via `_job_id=run_id`). No subprocess spawning, no cleanup of residual workers.
- **Cron scheduling**: Every process runs a lightweight asyncio cron scheduler (croniter-based). Crons load from PostgreSQL on startup and sync across processes via Redis pub/sub (`langgraph:cron:sync`). Multi-process deduplication via Redis claim key `langgraph:cron:fire:{cron_id}:{minute}` ensures each fire time executes once.
- **Langfuse config**: Set via `os.environ` (not passed programmatically) — langfuse reads env vars directly.
- **Embedding config**: Uses `init_embeddings` with `provider='openai'` and HNSW index. ANN index does not support >2000 dimensions.
- **Vite proxy vs env var**: In dev, Vite proxies `/api` to the backend. In production build, `VITE_API_URL` is used directly by the client.

## No Tests / CI Config

No test framework, linter, type checker, or CI workflows are configured for the backend. The frontend has `eslint` and `prettier` only.
