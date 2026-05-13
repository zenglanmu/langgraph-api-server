# LangGraph API Server

A standalone FastAPI-based LangGraph API server — an alternative to the official LangGraph server, which requires a Go-based LangSmith backend and cannot run independently.

## Why This Project?

The official LangGraph server is tightly coupled with LangSmith, making it inaccessible for many users due to:

- **Network restrictions** — LangSmith may be unreachable from your environment
- **Policy & compliance** — data residency or regulatory requirements prevent use of cloud-hosted services
- **Self-hosting costs** — the official open-source version lacks auth and other modules; full self-hosted deployment requires an enterprise LangSmith license
- **Deployment overhead** — the official server runs Python via gRPC to a Go backend, meaning you can't just deploy it as a simple Python service

This project was reverse-engineered from the LangGraph client-side SDK, so behavior is **not guaranteed to be identical** to the official server. If you find inconsistencies, please open an issue.

## Features

- **Embeddable** — mount into any existing FastAPI app via `setup_api()`, no need to deploy a standalone server
- **PostgreSQL + Redis backed** — persistent storage for threads, assistants, crons, and store
- **Background execution** — long-running agent tasks are offloaded to rq workers via Redis, keeping the FastAPI process responsive
- **SSE streaming** — supports server-sent events for real-time run output
- **Langfuse tracing** — optional integration for AI request observability
- **Vector store** — optional embedding support for the LangGraph store (HNSW index via pgvector)

## Prerequisites

- Python >= 3.12
- PostgreSQL
- Redis

## Quick Start

### 1. Install

```bash
uv sync
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your values (`.env` is excluded from version control via `.gitignore`):

```bash
cp .env.example .env
```

See `.env.example` for all available variables.

### 3. Register Your Agents

Agents must be registered as a **callable that returns a `CompiledStateGraph`**, not a pre-compiled graph. This is because compiled graphs cannot be pickled into rq worker subprocesses. See [langchain-ai/langgraph#3289](https://github.com/langchain-ai/langgraph/issues/3289).

```python
# examples/agents/weather.py
from langgraph_api import GraphRegistry

def build_graph():
    agent = create_your_agent(...)  # returns CompiledStateGraph
    return agent

GraphRegistry.registy_lg_graph("agent", build_graph)
```

### 4. Mount into FastAPI

See [`examples/main.py`](examples/main.py) for a complete example:

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph_api import setup_api
from dotenv import load_dotenv

load_dotenv()

import examples.agents  # registers agents via GraphRegistry

langgraph_api_lifespan = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global langgraph_api_lifespan
    async with langgraph_api_lifespan(app):
        yield

app = FastAPI(title="langgraph api server", lifespan=lifespan)

langgraph_api_lifespan = setup_api(
    router=app,
    redis_url=os.environ["REDIS_URL"],
    langgraph_database_uri=os.environ["LANGGRAPH_DATABASE_URI"],
    embeding_model_name=os.environ.get("LANGGRAPH_EMBED_MODEL"),
    embeding_dim=int(os.environ["LANGGRAPH_EMBED_DIMENSION"]),
    embeding_base_url=os.environ.get("LANGGRAPH_EMBED_MODEL_BASE_URL"),
    embeding_api_key=os.environ.get("LANGGRAPH_EMBED_MODEL_API_KEY"),
    include_router_kwargs=dict(prefix=""),  # default is /langgraph_api
)

# Now run: uvicorn app:app --host 0.0.0.0 --port 2024
```

You can also mount into an `APIRouter` instead of the app directly:

```python
from fastapi import APIRouter

router = APIRouter()
lifespan = setup_api(router=router, ...)
app.include_router(router, prefix="/my_api")
```

### 5. Run

```bash
uv run python examples/main.py
# Serves on http://127.0.0.1:2024
```

## API Reference

`setup_api()` parameters:

| Parameter | Type | Required | Description |
|---|---|---|---|
| `router` | `FastAPI \| APIRouter` | Yes | Router or app to mount routes into |
| `redis_url` | `str` | Yes | Redis connection URL |
| `langgraph_database_uri` | `str` | Yes | PostgreSQL connection URI |
| `include_router_kwargs` | `dict` | No | Passed to `router.include_router()`. Use `{"prefix": "/custom"}` to change API prefix (default: `/langgraph_api`) |
| `user_id_callback` | `Callable` | No | Sync or async callable returning a user ID for thread isolation |
| `langfuse_public_key` | `str` | No | Langfuse public key |
| `langfuse_secret_key` | `str` | No | Langfuse secret key |
| `langfuse_base_url` | `str` | No | Langfuse host URL |
| `embeding_model_name` | `str` | No | Embedding model name (format: `provider:model`, e.g. `openai:text-embedding-3-small`) |
| `embeding_dim` | `int` | No | Embedding dimensions (max 2000 for HNSW) |
| `embeding_base_url` | `str` | No | Embedding API base URL |
| `embeding_api_key` | `str` | No | Embedding API key |

`setup_api()` returns an async context manager (lifespan) that handles DB setup and background worker processes.

## Exported API

The package exports:

- **`setup_api()`** — mount all LangGraph API routes and return a lifespan context manager
- **`GraphRegistry`** — register agent graph builders via `GraphRegistry.registy_lg_graph(name, build_fn)`
- **`get_graph_store()`** — async context manager to get a `AsyncPostgresStore` instance
- **`get_graph_checkpointer()`** — async context manager to get an `AsyncPostgresSaver` instance

## Architecture

```
langgraph_api/
├── __init__.py          # setup_api(), exports
├── registry.py          # GraphRegistry, settings, DB connection helpers
├── api/                 # FastAPI routers
│   ├── runs.py          # run execution & SSE streaming
│   ├── threads.py       # thread management
│   ├── assistants.py    # assistant management
│   ├── store.py         # key-value store
│   └── crons.py         # cron/scheduled runs
├── services/            # business logic
│   ├── graph_run_service.py
│   ├── run_queue_service.py
│   └── cron_service.py
├── persistants/         # PostgreSQL persistence
│   ├── thread.py
│   ├── cron.py
│   ├── assistant.py
│   └── setup.py         # DB table initialization
└── utils/
    └── queue_worker.py  # rq worker pool + cron scheduler
```

**Startup**: Only one process in a multi-worker deployment runs DB setup and spawns background workers (Redis-based distributed lock with key `langgraph_api:bg_startup_lock`).

**Agent execution**: When a run is submitted, it's enqueued via Redis rq. Worker subprocesses pick up the job, deserialize the graph builder via its dotted module path, compile the graph, and execute it. This keeps the FastAPI process free from CPU-heavy agent work.

**Settings propagation**: `ApiGlobalSettings.snapshot()` serializes registered graph functions to dotted module paths (e.g. `examples.agents.weather:build_graph`) so worker subprocesses can import them via `load()`.

## Caveats

- The default API prefix is `/langgraph_api` (note the typo — `langgragh` in code). Override it via `include_router_kwargs={"prefix": "/your_prefix"}`.
- ANN (HNSW) index does not support embeddings with more than 2000 dimensions.
- `user_id_callback` is excluded from subprocess serialization since it may depend on FastAPI request context.
- This is a community implementation reverse-engineered from the client SDK — API behavior may differ from the official LangGraph server.

## License

MIT
