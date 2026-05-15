# LangGraph API Server

基于 FastAPI 的独立 LangGraph API 服务器，内置 React 聊天界面——官方 LangGraph 服务器的替代方案。官方服务器依赖 Go 语言编写的 LangSmith 后端，无法独立运行。

**[English](README.md)**

## 为什么需要这个项目？

官方 LangGraph 服务器与 LangSmith 紧密耦合，许多用户因以下原因无法使用：

- **网络限制** — LangSmith 在你的环境中可能无法访问
- **政策合规** — 数据驻留或监管要求禁止使用云端托管服务
- **私有化部署成本** — 官方开源版缺少认证等模块，完整私有化部署需要 LangSmith 企业版授权
- **部署开销大** — 官方服务器通过 gRPC 从 Python 调用 Go 后端，无法作为纯 Python 服务部署

本项目通过 LangGraph 客户端 SDK 逆向实现，**不保证行为与官方服务器完全一致**。如发现不一致之处，欢迎提 Issue。

## 特性

- **可嵌入** — 通过 `setup_api()` 挂载到任意 FastAPI 应用中，无需单独部署
- **内置聊天界面** — React 实现的 Agent 聊天 UI（`agent-chat-ui`），位于 `frontend/` 目录
- **PostgreSQL + Redis 持久化** — 支持线程、助手、定时任务和存储的持久化
- **后台执行** — 长耗时 agent 任务通过 Redis rq 队列在后台 Worker 进程中执行，不阻塞 FastAPI 主进程
- **SSE 流式输出** — 支持 Server-Sent Events 实时返回运行结果
- **Langfuse 链路追踪** — 可选集成，用于 AI 请求可观测性
- **向量存储** — 可选的 Embedding 支持，用于 LangGraph Store（基于 pgvector 的 HNSW 索引）

## 前置条件

- Python >= 3.12
- PostgreSQL
- Redis
- Node.js（前端开发需要）

## 快速开始

### 1. 安装

```bash
# 后端
uv sync

# 前端（在 frontend/ 目录下）
cd frontend && pnpm install && cd ..
```

### 2. 配置环境变量

复制 `.env.example` 到 `.env` 并填入你的配置（`.env` 已在 `.gitignore` 中排除，不会提交到版本库）：

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```

所有可用变量见 `.env.example`。

### 3. 注册 Agent

Agent 必须注册为**返回 `CompiledStateGraph` 的可调用对象**，而非预编译的图。因为编译后的图无法被 pickle 序列化到 rq Worker 子进程中。详见 [langchain-ai/langgraph#3289](https://github.com/langchain-ai/langgraph/issues/3289)。

```python
# examples/agents/weather.py
from langgraph_api import GraphRegistry

def build_graph():
    agent = create_your_agent(...)  # 返回 CompiledStateGraph
    return agent

GraphRegistry.registy_lg_graph("agent", build_graph)
```

### 4. 挂载到 FastAPI

完整示例见 [`examples/main.py`](examples/main.py)：

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph_api import setup_api
from dotenv import load_dotenv

load_dotenv()

import examples.agents  # 通过 GraphRegistry 注册 agent

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
    embeding_dim=int(os.environ["LANGGRAPH_EMBED_DIMENSION"]) if os.environ.get("LANGGRAPH_EMBED_DIMENSION") else None,
    embeding_base_url=os.environ.get("LANGGRAPH_EMBED_MODEL_BASE_URL"),
    embeding_api_key=os.environ.get("LANGGRAPH_EMBED_MODEL_API_KEY"),
    include_router_kwargs=dict(prefix=""),  # 默认前缀为 /langgraph_api
)

# 启动: uvicorn app:app --host 0.0.0.0 --port 2024
```

也可以挂载到 `APIRouter` 而非直接挂载到 app：

```python
from fastapi import APIRouter

router = APIRouter()
lifespan = setup_api(router=router, ...)
app.include_router(router, prefix="/my_api")
```

### 5. 运行

**后端 + 前端一起启动（推荐）：**

```bash
uv run python dev.py
# 后端  -> http://localhost:2024
# 前端  -> http://localhost:5173
```

一条命令同时启动后端 API 服务和前端开发服务器，并自动将 `.env.example` 复制为 `.env`（如果 `.env` 不存在）。

**仅后端：**

```bash
uv run python examples/main.py
# 服务地址: http://127.0.0.1:2024
```

**仅前端：**

```bash
cd frontend && pnpm dev
# 服务地址: http://localhost:5173
```

## 前端

`frontend/` 目录包含 **agent-chat-ui**，一个通过 LangGraph SDK 与后端通信的 React 聊天界面。

- **技术栈**: React 19, TypeScript, Vite, TailwindCSS 4, pnpm
- **Vite 代理**: 开发模式下，`/api` 请求会被代理到 `http://localhost:2024`（自动去除 `/api` 前缀）。生产构建中使用 `VITE_API_URL` 直接访问后端。
- **环境变量**（`frontend/.env`）：
  - `VITE_API_URL` — 后端地址（默认: `http://localhost:2024`）
  - `VITE_ASSISTANT_ID` — 默认助手 ID（默认: `agent`）
  - `VITE_AUTH_SCHEME` — Agent Builder 部署的认证方式（如 `langsmith-api-key`）
- **构建**: `pnpm build` 执行 `tsc -b && vite build`，输出到 `frontend/dist/`
- **代码检查**: `pnpm lint`（eslint），`pnpm format:check`（prettier + `prettier-plugin-tailwindcss`）

## API 参考

`setup_api()` 参数说明：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `router` | `FastAPI \| APIRouter` | 是 | 要挂载路由的 FastAPI 应用或 APIRouter |
| `redis_url` | `str` | 是 | Redis 连接地址 |
| `langgraph_database_uri` | `str` | 是 | PostgreSQL 连接 URI |
| `include_router_kwargs` | `dict` | 否 | 传递给 `router.include_router()` 的参数，可用 `{"prefix": "/custom"}` 修改 API 前缀（默认 `/langgraph_api`） |
| `user_id_callback` | `Callable` | 否 | 同步或异步的可调用对象，返回用户 ID，用于线程隔离 |
| `langfuse_public_key` | `str` | 否 | Langfuse 公钥 |
| `langfuse_secret_key` | `str` | 否 | Langfuse 密钥 |
| `langfuse_base_url` | `str` | 否 | Langfuse 服务地址 |
| `embeding_model_name` | `str` | 否 | Embedding 模型名称（格式：`provider:model`，如 `openai:text-embedding-3-small`） |
| `embeding_dim` | `int` | 否 | Embedding 维度（HNSW 索引上限 2000） |
| `embeding_base_url` | `str` | 否 | Embedding API 地址 |
| `embeding_api_key` | `str` | 否 | Embedding API 密钥 |

`setup_api()` 返回一个异步上下文管理器（lifespan），负责数据库初始化和后台 Worker 进程的管理。

## 导出接口

包导出以下内容：

- **`setup_api()`** — 挂载所有 LangGraph API 路由并返回 lifespan 上下文管理器
- **`GraphRegistry`** — 通过 `GraphRegistry.registy_lg_graph(name, build_fn)` 注册 agent 图构建器
- **`get_graph_store()`** — 异步上下文管理器，获取 `AsyncPostgresStore` 实例
- **`get_graph_checkpointer()`** — 异步上下文管理器，获取 `AsyncPostgresSaver` 实例

## 架构

```
langgraph_api/
├── __init__.py          # setup_api(), 导出
├── registry.py          # GraphRegistry, 全局配置, 数据库连接工具
├── api/                 # FastAPI 路由
│   ├── runs.py          # 运行执行与 SSE 流式输出
│   ├── threads.py       # 线程管理
│   ├── assistants.py    # 助手管理
│   ├── store.py         # 键值存储
│   └── crons.py         # 定时/计划任务
├── services/            # 业务逻辑
│   ├── graph_run_service.py
│   ├── run_queue_service.py
│   └── cron_service.py
├── persistants/         # PostgreSQL 持久化
│   ├── thread.py
│   ├── cron.py
│   ├── assistant.py
│   └── setup.py         # 数据库表初始化
└── utils/
    └── queue_worker.py  # rq Worker 池 + 定时任务调度器

frontend/
├── src/
│   ├── App.tsx                    # StreamProvider > ThreadProvider > ArtifactProvider > Thread
│   ├── providers/
│   │   ├── client.ts              # LangGraph SDK 客户端配置
│   │   ├── Stream.tsx             # SSE 流式 Provider
│   │   └── Thread.tsx             # 线程状态 Provider
│   └── components/
│       ├── thread/                # 聊天线程 UI、消息、Markdown 渲染、agent-inbox 中断处理
│       └── ui/                    # shadcn/ui 基础组件（基于 Radix UI）
├── vite.config.ts                 # 开发代理: /api → http://localhost:2024
└── package.json                   # agent-chat-ui
```

**启动机制**：多 Worker 部署时，仅有一个进程执行数据库初始化和启动后台进程（基于 Redis 分布式锁，key 为 `langgraph_api:bg_startup_lock`）。

**Agent 执行流程**：提交运行请求后，任务通过 Redis rq 入队。Worker 子进程从队列中获取任务，通过模块路径字符串反序列化图构建函数，编译图后执行。这样确保 CPU 密集型的 agent 工作不会拖垮 FastAPI 进程。

**配置传递**：`ApiGlobalSettings.snapshot()` 将注册的图函数序列化为模块路径字符串（如 `examples.agents.weather:build_graph`），Worker 子进程通过 `load()` 动态导入还原。

## 注意事项

- 默认 API 前缀为 `/langgraph_api`（代码中为 `langgragh`，历史拼写错误），可通过 `include_router_kwargs={"prefix": "/your_prefix"}` 覆盖。
- HNSW 索引不支持超过 2000 维的 Embedding。
- `user_id_callback` 不参与子进程序列化，因为它可能依赖 FastAPI 请求上下文。
- 本项目为 vibe coding 参考官方客户端 SDK 实现，API 行为可能与官方 LangGraph 服务器存在差异。

## 许可证

MIT
