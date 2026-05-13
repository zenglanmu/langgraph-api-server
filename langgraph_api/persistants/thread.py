import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, NamedTuple

import orjson
from psycopg import AsyncConnection, AsyncCursor
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from langgraph.checkpoint.postgres import _ainternal
from langgraph.store.postgres.base import (
    PLACEHOLDER,
    PostgresIndexConfig,
    _ensure_index_config,
    _get_vector_type_ops,
    get_distance_operator,
)
from langgraph.store.base import get_text_at_path, tokenize_path

from ._sql import split_sql_statements

logger = logging.getLogger(__name__)


class Migration(NamedTuple):
    sql: str
    params: dict[str, Any] | None = None
    condition: Callable[["AsyncPostgresThread"], bool] | None = None


class AsyncPostgresThread:
    """Async Postgres-backed thread and run management with optional vector search.

    Uses raw SQL via psycopg for all operations. Supports vector similarity search
    on threads.metadata and threads.values JSONB columns when a PostgresIndexConfig
    is provided.
    """

    MIGRATIONS: list[str] = [
        """\
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    status TEXT DEFAULT 'idle',
    values JSONB DEFAULT '{}'::jsonb,
    user_id TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_threads_user_id ON threads(user_id);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    assistant_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    status TEXT DEFAULT 'pending',
    metadata JSONB DEFAULT '{}'::jsonb,
    multitask_strategy TEXT DEFAULT 'reject',
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_thread_id ON runs(thread_id);
CREATE INDEX IF NOT EXISTS idx_runs_thread_id_created_at
ON runs(thread_id, created_at DESC);
""",
    ]

    VECTOR_MIGRATIONS: list[Migration] = [
        Migration(
            sql="""\
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE EXTENSION vector;
    END IF;
END $$;
""",
        ),
        Migration(
            sql="""\
CREATE TABLE IF NOT EXISTS thread_vectors (
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    embedding %(vector_type)s(%(dims)s),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, field_name)
);
""",
            params={
                "dims": lambda store: store.index_config["dims"],
                "vector_type": lambda store: store.index_config.get(
                    "ann_index_config", {}
                ).get("vector_type", "vector"),
            },
        ),
        Migration(
            sql="""\
CREATE INDEX IF NOT EXISTS thread_vectors_embedding_idx
ON thread_vectors USING %(index_type)s (embedding %(ops)s);
""",
            params={
                "index_type": lambda store: store.index_config.get(
                    "ann_index_config", {}
                ).get("kind", "hnsw"),
                "ops": lambda store: _get_vector_type_ops(store),
            },
            condition=lambda store: store.index_config.get("ann_index_config", {}).get(
                "kind", "hnsw"
            )
            != "flat",
        ),
    ]

    __slots__ = (
        "_deserializer",
        "conn",
        "lock",
        "index_config",
        "embeddings",
    )

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None = None,
        index: PostgresIndexConfig | None = None,
    ) -> None:
        self.conn = conn
        self._deserializer = deserializer
        self.lock = asyncio.Lock()
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: PostgresIndexConfig | None = None,
    ) -> AsyncIterator["AsyncPostgresThread"]:
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            yield cls(conn=conn, index=index)

    async def setup(self) -> None:
        """Set up the thread database.

        Creates necessary tables and indexes if they don't already exist and
        runs database migrations. MUST be called directly by the user the
        first time the store is used.
        """

        async def _get_version(cur: AsyncCursor[DictRow], table: str) -> int:
            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
            )
            await cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
            row = await cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            return version

        async with self._cursor() as cur:
            version = await _get_version(cur, table="thread_migrations")
            for v, sql in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                try:
                    for stmt in split_sql_statements(sql):
                        await cur.execute(stmt)
                    await cur.execute(
                        "INSERT INTO thread_migrations (v) VALUES (%s)", (v,)
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to apply migration {v}.\nSql={sql}\nError={e}"
                    )
                    raise

            if self.index_config:
                version = await _get_version(cur, table="thread_vector_migrations")
                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    if migration.condition and not migration.condition(self):
                        continue
                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        if "dims" in params:
                            try:
                                params["dims"] = int(params["dims"])
                            except Exception as e:
                                raise ValueError(
                                    f"Invalid dims for vector index: {params['dims']}"
                                ) from e
                        if "vector_type" in params:
                            vt = str(params["vector_type"])
                            if vt not in ("vector", "halfvec"):
                                raise ValueError(
                                    f"Invalid vector_type for pgvector: {vt}"
                                )
                            params["vector_type"] = vt
                        if "index_type" in params:
                            it = str(params["index_type"])
                            if it not in ("hnsw", "ivfflat"):
                                raise ValueError(
                                    f"Invalid index_type for pgvector: {it}"
                                )
                            params["index_type"] = it
                        sql = sql % params
                    try:
                        await cur.execute(sql)
                        await cur.execute(
                            "INSERT INTO thread_vector_migrations (v) VALUES (%s)",
                            (v,),
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to apply vector migration {v}.\nSql={sql}\nError={e}"
                        )
                        raise

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Acquire a database cursor with connection pooling and lock support."""
        async with _ainternal.get_connection(self.conn) as conn:
            async with self.lock:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    # ── Thread operations ────────────────────────────────────────────────

    async def thread_get(
        self, thread_id: str, *, user_id: str | None = None
    ) -> dict | None:
        """Get a single thread by ID, optionally filtered by user_id."""
        if user_id is not None:
            async with self._cursor() as cur:
                await cur.execute(
                    """SELECT thread_id, created_at, updated_at, metadata, status, "values", user_id
                       FROM threads WHERE thread_id = %s AND user_id = %s""",
                    (thread_id, user_id),
                )
                row = await cur.fetchone()
                return dict(row) if row else None
        async with self._cursor() as cur:
            await cur.execute(
                """SELECT thread_id, created_at, updated_at, metadata, status, "values", user_id
                   FROM threads WHERE thread_id = %s""",
                (thread_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def thread_put(
        self,
        thread_id: str,
        *,
        metadata: dict | None = None,
        values: dict | None = None,
        status: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        """Insert or update a thread. Returns the full thread row."""
        metadata = metadata or {}
        values = values or {}
        status = status or "idle"

        async with self._cursor() as cur:
            await cur.execute(
                """INSERT INTO threads (thread_id, metadata, "values", status, user_id, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                   ON CONFLICT (thread_id) DO UPDATE
                   SET metadata = EXCLUDED.metadata,
                       "values" = EXCLUDED."values",
                       status = EXCLUDED.status,
                       user_id = EXCLUDED.user_id,
                       updated_at = CURRENT_TIMESTAMP
                   RETURNING thread_id, created_at, updated_at, metadata, status, "values", user_id""",
                (thread_id, Jsonb(metadata), Jsonb(values), status, user_id),
            )
            row = await cur.fetchone()

            if self.index_config and self.embeddings:
                await self._upsert_thread_vectors(cur, thread_id, metadata, values)

            return dict(row)

    async def _upsert_thread_vectors(
        self,
        cur: AsyncCursor[DictRow],
        thread_id: str,
        metadata: dict,
        values: dict,
    ) -> None:
        """Compute and upsert vector embeddings for metadata and values fields."""
        texts: list[str] = []
        field_names: list[str] = []
        if metadata:
            texts.append(orjson.dumps(metadata).decode("utf-8"))
            field_names.append("metadata")
        if values:
            texts.append(orjson.dumps(values).decode("utf-8"))
            field_names.append("values")

        if not texts:
            return

        vectors = await self.embeddings.aembed_documents(texts)
        dims = int(self.index_config["dims"])
        ann_config = self.index_config.get("ann_index_config", {})
        vector_type = ann_config.get("vector_type", "vector")
        for field_name, vector in zip(field_names, vectors):
            await cur.execute(
                f"""INSERT INTO thread_vectors (thread_id, field_name, embedding, created_at, updated_at)
                    VALUES (%s, %s, %s::{vector_type}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (thread_id, field_name) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP""",
                (thread_id, field_name, vector),
            )

    async def thread_delete(
        self, thread_id: str, *, user_id: str | None = None
    ) -> None:
        """Delete a thread and its cascade-related records, optionally filtered by user_id."""
        if user_id is not None:
            async with self._cursor() as cur:
                await cur.execute(
                    "DELETE FROM threads WHERE thread_id = %s AND user_id = %s",
                    (thread_id, user_id),
                )
        else:
            async with self._cursor() as cur:
                await cur.execute(
                    "DELETE FROM threads WHERE thread_id = %s",
                    (thread_id,),
                )

    async def thread_search(
        self,
        *,
        metadata_filter: dict[str, Any] | None = None,
        values_filter: dict[str, Any] | None = None,
        status: str | None = None,
        ids: list[str] | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: str | None = None,
        sort_order: str | None = None,
        query: str | None = None,
        user_id: str | None = None,
    ) -> list[dict]:
        """Search threads with optional filters and vector similarity search."""

        if query and self.index_config and self.embeddings:
            return await self._thread_search_vector(
                metadata_filter=metadata_filter,
                values_filter=values_filter,
                status=status,
                ids=ids,
                limit=limit,
                offset=offset,
                query=query,
                user_id=user_id,
            )

        conditions: list[str] = []
        params: list[Any] = []

        if metadata_filter:
            self._build_jsonb_filter(conditions, params, "metadata", metadata_filter)
        if values_filter:
            self._build_jsonb_filter(conditions, params, "values", values_filter)
        if status is not None:
            conditions.append(f"status = %s")
            params.append(status)
        if ids:
            placeholders = ",".join(["%s"] * len(ids))
            conditions.append(f"thread_id IN ({placeholders})")
            params.extend(ids)
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        sort_column = self._resolve_sort_column(sort_by)
        direction = "DESC" if sort_order and sort_order.lower() == "desc" else "ASC"

        sql = f"""SELECT thread_id, created_at, updated_at, metadata, status, "values", user_id
                  FROM threads
                  WHERE {where_clause}
                  ORDER BY {sort_column} {direction}
                  LIMIT %s OFFSET %s"""
        params.extend([limit, offset])

        async with self._cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def _thread_search_vector(
        self,
        *,
        metadata_filter: dict[str, Any] | None,
        values_filter: dict[str, Any] | None,
        status: str | None,
        ids: list[str] | None,
        limit: int,
        offset: int,
        query: str,
        user_id: str | None = None,
    ) -> list[dict]:
        """Perform vector similarity search on thread metadata/values."""
        score_operator, post_operator = get_distance_operator(self)
        ann_config = self.index_config.get("ann_index_config", {})
        vector_type = ann_config.get("vector_type", "vector")
        score_operator = score_operator % ("%s", vector_type)
        score_operator = score_operator.replace("sv", "tv")
        post_operator = post_operator.replace("scored", "uniq")

        conditions: list[str] = []
        params: list[Any] = []

        if metadata_filter:
            self._build_jsonb_filter(conditions, params, "t.metadata", metadata_filter)
        if values_filter:
            self._build_jsonb_filter(conditions, params, 't."values"', values_filter)
        if status is not None:
            conditions.append("t.status = %s")
            params.append(status)
        if ids:
            placeholders = ",".join(["%s"] * len(ids))
            conditions.append(f"t.thread_id IN ({placeholders})")
            params.extend(ids)
        if user_id is not None:
            conditions.append("t.user_id = %s")
            params.append(user_id)

        extra_where = (" AND " + " AND ".join(conditions)) if conditions else ""

        vectors_per_doc_estimate = self.index_config.get("__estimated_num_vectors", 2)
        expanded_limit = (limit * vectors_per_doc_estimate * 2) + 1

        vector_search_cte = f"""
            SELECT t.thread_id, t.created_at, t.updated_at, t.metadata, t.status, t."values", t.user_id,
                {score_operator} AS neg_score
            FROM threads t
            JOIN thread_vectors tv ON t.thread_id = tv.thread_id
            WHERE TRUE {extra_where}
            ORDER BY {score_operator} ASC
            LIMIT %s
        """

        search_results_sql = f"""
            WITH scored AS (
                {vector_search_cte}
            )
            SELECT uniq.thread_id, uniq.created_at, uniq.updated_at, uniq.metadata, uniq.status, uniq."values", uniq.user_id,
                {post_operator} AS score
            FROM (
                SELECT DISTINCT ON (scored.thread_id)
                    scored.*
                FROM scored
                ORDER BY scored.thread_id, scored.neg_score ASC
            ) uniq
            ORDER BY score DESC
            LIMIT %s
            OFFSET %s
        """

        query_vector = (await self.embeddings.aembed_documents([query]))[0]
        search_params = [
            query_vector,
            *params,
            query_vector,
            expanded_limit,
            limit,
            offset,
        ]

        async with self._cursor() as cur:
            await cur.execute(search_results_sql, search_params)
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def _build_jsonb_filter(
        conditions: list[str],
        params: list[Any],
        column: str,
        filter_dict: dict[str, Any],
    ) -> None:
        """Build WHERE conditions for JSONB column equality/operator filters."""
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                for op_name, val in value.items():
                    condition, p = AsyncPostgresThread._get_filter_condition(
                        column, key, op_name, val
                    )
                    conditions.append(condition)
                    params.extend(p)
            else:
                conditions.append(f"{column}->>%s = %s")
                params.extend([key, str(value)])

    @staticmethod
    def _get_filter_condition(
        column: str, key: str, op: str, value: Any
    ) -> tuple[str, list]:
        """Generate a JSONB filter condition for a given operator."""
        if op == "$eq":
            return f"{column}->%s = %s::jsonb", [key, json.dumps(value)]
        elif op == "$gt":
            return f"{column}->>%s > %s", [key, str(value)]
        elif op == "$gte":
            return f"{column}->>%s >= %s", [key, str(value)]
        elif op == "$lt":
            return f"{column}->>%s < %s", [key, str(value)]
        elif op == "$lte":
            return f"{column}->>%s <= %s", [key, str(value)]
        elif op == "$ne":
            return f"{column}->%s != %s::jsonb", [key, json.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")

    @staticmethod
    def _resolve_sort_column(sort_by: str | None) -> str:
        """Map sort field name to safe column reference."""
        allowed = {
            "thread_id": "thread_id",
            "status": "status",
            "created_at": "created_at",
            "updated_at": "updated_at",
            "state_updated_at": "updated_at",
        }
        return allowed.get(sort_by, "updated_at")

    async def thread_count(
        self,
        *,
        metadata_filter: dict[str, Any] | None = None,
        values_filter: dict[str, Any] | None = None,
        status: str | None = None,
        user_id: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[Any] = []

        if metadata_filter:
            self._build_jsonb_filter(conditions, params, "metadata", metadata_filter)
        if values_filter:
            self._build_jsonb_filter(conditions, params, "values", values_filter)
        if status is not None:
            conditions.append("status = %s")
            params.append(status)
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        async with self._cursor() as cur:
            await cur.execute(
                f"SELECT COUNT(*) AS cnt FROM threads WHERE {where_clause}",
                params,
            )
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    async def thread_batch_delete(self, thread_ids: list[str]) -> None:
        """Delete multiple threads by ID."""
        if not thread_ids:
            return
        async with self._cursor() as cur:
            await cur.execute(
                "DELETE FROM threads WHERE thread_id = ANY(%s)",
                (thread_ids,),
            )

    async def thread_copy(
        self,
        thread_id: str,
        new_thread_id: str,
        *,
        user_id: str | None = None,
    ) -> dict | None:
        """Copy a thread to a new ID. Returns the new thread row."""
        async with self._cursor() as cur:
            await cur.execute(
                """INSERT INTO threads (thread_id, metadata, "values", status, user_id, created_at, updated_at)
                   SELECT %s, metadata, "values", status, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                   FROM threads
                   WHERE thread_id = %s
                   RETURNING thread_id, created_at, updated_at, metadata, status, "values", user_id""",
                (new_thread_id, user_id, thread_id),
            )
            row = await cur.fetchone()
            if row is None:
                return None

            result = dict(row)

            if self.index_config and self.embeddings:
                metadata = result.get("metadata")
                values = result.get("values")
                if metadata or values:
                    await self._upsert_thread_vectors(
                        cur, new_thread_id, metadata or {}, values or {}
                    )

            return result

    # ── Run operations ───────────────────────────────────────────────────

    async def run_put(
        self,
        run_id: str,
        thread_id: str,
        *,
        metadata: dict | None = None,
        status: str | None = None,
        assistant_id: str | None = None,
        multitask_strategy: str | None = None,
    ) -> dict:
        """Create a new run record. Returns the full run row."""
        metadata = metadata or {}
        status = status or "pending"
        multitask_strategy = multitask_strategy or "reject"

        async with self._cursor() as cur:
            await cur.execute(
                """INSERT INTO runs (run_id, thread_id, assistant_id, metadata, status,
                                     multitask_strategy, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                   RETURNING run_id, thread_id, assistant_id, created_at, updated_at,
                             status, metadata, multitask_strategy, started_at, finished_at,
                             error_message""",
                (run_id, thread_id, assistant_id, Jsonb(metadata), status, multitask_strategy),
            )
            return dict(await cur.fetchone())

    async def run_get_last(self, thread_id: str) -> dict | None:
        """Get the most recent run for a thread."""
        async with self._cursor() as cur:
            await cur.execute(
                """SELECT run_id, thread_id, assistant_id, created_at, updated_at,
                          status, metadata, multitask_strategy, started_at, finished_at,
                          error_message
                   FROM runs
                   WHERE thread_id = %s
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (thread_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def run_update(
        self,
        run_id: str,
        *,
        metadata: dict | None = None,
        status: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error_message: str | None = None,
    ) -> dict | None:
        """Update a run's fields. Only non-None values are updated. Returns the full row."""
        set_parts = ["updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = []

        if metadata is not None:
            set_parts.append("metadata = %s")
            params.append(Jsonb(metadata))
        if status is not None:
            set_parts.append("status = %s")
            params.append(status)
        if started_at is not None:
            set_parts.append("started_at = %s")
            params.append(started_at)
        if finished_at is not None:
            set_parts.append("finished_at = %s")
            params.append(finished_at)
        if error_message is not None:
            set_parts.append("error_message = %s")
            params.append(error_message)

        if len(params) == 0:
            return None

        params.append(run_id)

        async with self._cursor() as cur:
            await cur.execute(
                f"""UPDATE runs SET {", ".join(set_parts)}
                    WHERE run_id = %s
                    RETURNING run_id, thread_id, assistant_id, created_at, updated_at,
                              status, metadata, multitask_strategy, started_at, finished_at,
                              error_message""",
                params,
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def run_get(self, run_id: str) -> dict | None:
        """Get a single run by ID."""
        async with self._cursor() as cur:
            await cur.execute(
                """SELECT run_id, thread_id, assistant_id, created_at, updated_at,
                          status, metadata, multitask_strategy, started_at, finished_at,
                          error_message
                   FROM runs WHERE run_id = %s""",
                (run_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def run_list(
        self,
        thread_id: str,
        *,
        limit: int = 10,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict]:
        """List runs for a thread with optional filtering."""
        conditions = ["thread_id = %s"]
        params: list[Any] = [thread_id]

        if status is not None:
            conditions.append("status = %s")
            params.append(status)

        where_clause = " AND ".join(conditions)

        params.extend([limit, offset])

        async with self._cursor() as cur:
            await cur.execute(
                f"""SELECT run_id, thread_id, assistant_id, created_at, updated_at,
                           status, metadata, multitask_strategy, started_at, finished_at,
                           error_message
                    FROM runs
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s""",
                params,
            )
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def run_delete(self, run_id: str) -> None:
        """Delete a run by ID."""
        async with self._cursor() as cur:
            await cur.execute(
                "DELETE FROM runs WHERE run_id = %s",
                (run_id,),
            )

    async def run_cancel_many(
        self,
        *,
        thread_id: str | None = None,
        run_ids: list[str] | None = None,
        status: str | None = None,
    ) -> list[str]:
        """Cancel (set status to cancelled) matching runs. Returns list of cancelled run IDs."""
        conditions: list[str] = []
        params: list[Any] = []

        if thread_id is not None:
            conditions.append("thread_id = %s")
            params.append(thread_id)
        if run_ids is not None:
            placeholders = ",".join(["%s"] * len(run_ids))
            conditions.append(f"run_id IN ({placeholders})")
            params.extend(run_ids)
        if status is not None:
            if status == "all":
                pass
            else:
                conditions.append("status = %s")
                params.append(status)

        non_terminal = ["pending", "running"]
        terminal_placeholder = ",".join(["%s"] * len(non_terminal))
        conditions.append(f"status IN ({terminal_placeholder})")
        params.extend(non_terminal)

        where_clause = " AND ".join(conditions)

        async with self._cursor() as cur:
            await cur.execute(
                f"""UPDATE runs
                    SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP, finished_at = CURRENT_TIMESTAMP
                    WHERE {where_clause}
                    RETURNING run_id""",
                params,
            )
            rows = await cur.fetchall()
            return [row["run_id"] for row in rows]


