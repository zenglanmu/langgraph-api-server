import asyncio
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
    PostgresIndexConfig,
    _ensure_index_config,
    _get_vector_type_ops,
    get_distance_operator,
)

from ._sql import split_sql_statements

logger = logging.getLogger(__name__)


class Migration(NamedTuple):
    sql: str
    params: dict[str, Any] | None = None
    condition: Callable[["AsyncPostgresAssistant"], bool] | None = None


class AsyncPostgresAssistant:
    """Async Postgres-backed assistant persistence with optional vector search.

    Uses raw SQL via psycopg for all operations. Supports vector similarity search
    on assistants.metadata JSONB column when a PostgresIndexConfig is provided.
    Includes user_id isolation like threads.
    """

    MIGRATIONS: list[str] = [
        """\
CREATE TABLE IF NOT EXISTS assistants (
    assistant_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    context JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    version INTEGER DEFAULT 1 NOT NULL,
    name TEXT DEFAULT 'Untitled',
    description TEXT DEFAULT NULL,
    user_id TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_assistants_graph_id ON assistants(graph_id);
CREATE INDEX IF NOT EXISTS idx_assistants_user_id ON assistants(user_id);
CREATE INDEX IF NOT EXISTS idx_assistants_name ON assistants(name);
""",
        """\
CREATE TABLE IF NOT EXISTS assistant_versions (
    assistant_id TEXT NOT NULL REFERENCES assistants(assistant_id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    graph_id TEXT NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    context JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    name TEXT DEFAULT 'Untitled',
    description TEXT DEFAULT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY (assistant_id, version)
);
CREATE INDEX IF NOT EXISTS idx_assistant_versions_assistant_id ON assistant_versions(assistant_id);
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
CREATE TABLE IF NOT EXISTS assistant_vectors (
    assistant_id TEXT NOT NULL REFERENCES assistants(assistant_id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    embedding %(vector_type)s(%(dims)s),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (assistant_id, field_name)
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
CREATE INDEX IF NOT EXISTS assistant_vectors_embedding_idx
ON assistant_vectors USING %(index_type)s (embedding %(ops)s);
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
        "conn",
        "lock",
        "index_config",
        "embeddings",
    )

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        index: PostgresIndexConfig | None = None,
    ) -> None:
        self.conn = conn
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
    ) -> AsyncIterator["AsyncPostgresAssistant"]:
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            yield cls(conn=conn, index=index)

    async def setup(self) -> None:
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
            version = await _get_version(cur, table="assistant_migrations")
            for v, sql in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                try:
                    for stmt in split_sql_statements(sql):
                        await cur.execute(stmt)
                    await cur.execute(
                        "INSERT INTO assistant_migrations (v) VALUES (%s)", (v,)
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to apply migration {v}.\nSql={sql}\nError={e}"
                    )
                    raise

            if self.index_config:
                version = await _get_version(
                    cur, table="assistant_vector_migrations"
                )
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
                            "INSERT INTO assistant_vector_migrations (v) VALUES (%s)",
                            (v,),
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to apply vector migration {v}.\nSql={sql}\nError={e}"
                        )
                        raise

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[AsyncCursor[DictRow]]:
        async with _ainternal.get_connection(self.conn) as conn:
            async with self.lock:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    SELECT_COLUMNS = """assistant_id, graph_id, config, context, created_at, updated_at, metadata, version, name, description, user_id"""

    # ── Assistant CRUD ────────────────────────────────────────────────────

    async def assistant_get(
        self, assistant_id: str, *, user_id: str | None = None
    ) -> dict | None:
        if user_id is not None:
            async with self._cursor() as cur:
                await cur.execute(
                    f"""SELECT {self.SELECT_COLUMNS}
                       FROM assistants WHERE assistant_id = %s AND user_id = %s""",
                    (assistant_id, user_id),
                )
                row = await cur.fetchone()
                return dict(row) if row else None
        async with self._cursor() as cur:
            await cur.execute(
                f"""SELECT {self.SELECT_COLUMNS}
                   FROM assistants WHERE assistant_id = %s""",
                (assistant_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def assistant_create(
        self,
        *,
        assistant_id: str,
        graph_id: str,
        config: dict | None = None,
        context: dict | None = None,
        metadata: dict | None = None,
        name: str | None = None,
        description: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        config = config or {}
        context = context or {}
        metadata = metadata or {}
        name = name or "Untitled"

        async with self._cursor() as cur:
            await cur.execute(
                f"""INSERT INTO assistants (assistant_id, graph_id, config, context, metadata, version, name, description, user_id, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, 1, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                   RETURNING {self.SELECT_COLUMNS}""",
                (
                    assistant_id,
                    graph_id,
                    Jsonb(config),
                    Jsonb(context),
                    Jsonb(metadata),
                    name,
                    description,
                    user_id,
                ),
            )
            row = await cur.fetchone()

            await cur.execute(
                """INSERT INTO assistant_versions (assistant_id, version, graph_id, config, context, metadata, name, description, created_at)
                   VALUES (%s, 1, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)""",
                (
                    assistant_id,
                    graph_id,
                    Jsonb(config),
                    Jsonb(context),
                    Jsonb(metadata),
                    name,
                    description,
                ),
            )

            if self.index_config and self.embeddings:
                await self._upsert_assistant_vectors(
                    cur, assistant_id, metadata
                )

            return dict(row)

    async def assistant_update(
        self,
        assistant_id: str,
        *,
        graph_id: str | None = None,
        config: dict | None = None,
        context: dict | None = None,
        metadata: dict | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> dict | None:
        set_parts: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = []

        if graph_id is not None:
            set_parts.append("graph_id = %s")
            params.append(graph_id)
        if config is not None:
            set_parts.append("config = %s")
            params.append(Jsonb(config))
        if context is not None:
            set_parts.append("context = %s")
            params.append(Jsonb(context))
        if metadata is not None:
            set_parts.append("metadata = %s")
            params.append(Jsonb(metadata))
        if name is not None:
            set_parts.append("name = %s")
            params.append(name)
        if description is not None:
            set_parts.append("description = %s")
            params.append(description)

        needs_version_bump = any(
            k is not None for k in [graph_id, config, context, metadata]
        )
        if needs_version_bump:
            set_parts.append("version = version + 1")

        params.append(assistant_id)

        async with self._cursor() as cur:
            await cur.execute(
                f"""UPDATE assistants SET {", ".join(set_parts)}
                    WHERE assistant_id = %s
                    RETURNING {self.SELECT_COLUMNS}""",
                params,
            )
            row = await cur.fetchone()
            if row is None:
                return None

            result = dict(row)

            if needs_version_bump:
                await cur.execute(
                    """INSERT INTO assistant_versions (assistant_id, version, graph_id, config, context, metadata, name, description, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)""",
                    (
                        assistant_id,
                        result["version"],
                        result["graph_id"],
                        Jsonb(result.get("config", {})),
                        Jsonb(result.get("context", {})),
                        Jsonb(result.get("metadata", {})),
                        result.get("name"),
                        result.get("description"),
                    ),
                )

            if self.index_config and self.embeddings and metadata is not None:
                await self._upsert_assistant_vectors(
                    cur, assistant_id, metadata
                )

            return result

    async def assistant_delete(
        self, assistant_id: str, *, user_id: str | None = None
    ) -> bool:
        if user_id is not None:
            async with self._cursor() as cur:
                await cur.execute(
                    "DELETE FROM assistants WHERE assistant_id = %s AND user_id = %s",
                    (assistant_id, user_id),
                )
                return cur.rowcount > 0
        async with self._cursor() as cur:
            await cur.execute(
                "DELETE FROM assistants WHERE assistant_id = %s",
                (assistant_id,),
            )
            return cur.rowcount > 0

    async def assistant_search(
        self,
        *,
        metadata_filter: dict[str, Any] | None = None,
        graph_id: str | None = None,
        name: str | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: str | None = None,
        sort_order: str | None = None,
        query: str | None = None,
        user_id: str | None = None,
    ) -> list[dict]:
        if query and self.index_config and self.embeddings:
            return await self._assistant_search_vector(
                metadata_filter=metadata_filter,
                graph_id=graph_id,
                name=name,
                limit=limit,
                offset=offset,
                query=query,
                user_id=user_id,
            )

        conditions: list[str] = []
        params: list[Any] = []

        if metadata_filter:
            self._build_jsonb_filter(conditions, params, "metadata", metadata_filter)
        if graph_id is not None:
            conditions.append("graph_id = %s")
            params.append(graph_id)
        if name is not None:
            conditions.append("name ILIKE %s")
            params.append(f"%{name}%")
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        sort_column = self._resolve_sort_column(sort_by)
        direction = "DESC" if sort_order and sort_order.lower() == "desc" else "ASC"

        sql = f"""SELECT {self.SELECT_COLUMNS}
                  FROM assistants
                  WHERE {where_clause}
                  ORDER BY {sort_column} {direction}
                  LIMIT %s OFFSET %s"""
        params.extend([limit, offset])

        async with self._cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def _assistant_search_vector(
        self,
        *,
        metadata_filter: dict[str, Any] | None,
        graph_id: str | None,
        name: str | None,
        limit: int,
        offset: int,
        query: str,
        user_id: str | None = None,
    ) -> list[dict]:
        score_operator, post_operator = get_distance_operator(self)
        ann_config = self.index_config.get("ann_index_config", {})
        vector_type = ann_config.get("vector_type", "vector")
        score_operator = score_operator % ("%s", vector_type)
        score_operator = score_operator.replace("sv", "av")
        post_operator = post_operator.replace("scored", "uniq")

        conditions: list[str] = []
        params: list[Any] = []

        if metadata_filter:
            self._build_jsonb_filter(conditions, params, "a.metadata", metadata_filter)
        if graph_id is not None:
            conditions.append("a.graph_id = %s")
            params.append(graph_id)
        if name is not None:
            conditions.append("a.name ILIKE %s")
            params.append(f"%{name}%")
        if user_id is not None:
            conditions.append("a.user_id = %s")
            params.append(user_id)

        extra_where = (" AND " + " AND ".join(conditions)) if conditions else ""

        vectors_per_doc_estimate = self.index_config.get("__estimated_num_vectors", 1)
        expanded_limit = (limit * vectors_per_doc_estimate * 2) + 1

        vector_search_cte = f"""
            SELECT a.assistant_id, a.graph_id, a.config, a.context, a.created_at, a.updated_at,
                   a.metadata, a.version, a.name, a.description, a.user_id,
                {score_operator} AS neg_score
            FROM assistants a
            JOIN assistant_vectors av ON a.assistant_id = av.assistant_id
            WHERE TRUE {extra_where}
            ORDER BY {score_operator} ASC
            LIMIT %s
        """

        search_results_sql = f"""
            WITH scored AS (
                {vector_search_cte}
            )
            SELECT uniq.assistant_id, uniq.graph_id, uniq.config, uniq.context, uniq.created_at, uniq.updated_at,
                   uniq.metadata, uniq.version, uniq.name, uniq.description, uniq.user_id,
                {post_operator} AS score
            FROM (
                SELECT DISTINCT ON (scored.assistant_id)
                    scored.*
                FROM scored
                ORDER BY scored.assistant_id, scored.neg_score ASC
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

    async def assistant_count(
        self,
        *,
        metadata_filter: dict[str, Any] | None = None,
        graph_id: str | None = None,
        name: str | None = None,
        user_id: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[Any] = []

        if metadata_filter:
            self._build_jsonb_filter(conditions, params, "metadata", metadata_filter)
        if graph_id is not None:
            conditions.append("graph_id = %s")
            params.append(graph_id)
        if name is not None:
            conditions.append("name ILIKE %s")
            params.append(f"%{name}%")
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        async with self._cursor() as cur:
            await cur.execute(
                f"SELECT COUNT(*) AS cnt FROM assistants WHERE {where_clause}",
                params,
            )
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    # ── Version operations ───────────────────────────────────────────────

    async def assistant_get_versions(
        self,
        assistant_id: str,
        *,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict]:
        conditions: list[str] = ["assistant_id = %s"]
        params: list[Any] = [assistant_id]

        if metadata_filter:
            self._build_jsonb_filter(conditions, params, "metadata", metadata_filter)

        where_clause = " AND ".join(conditions)
        params.extend([limit, offset])

        async with self._cursor() as cur:
            await cur.execute(
                f"""SELECT assistant_id, version, graph_id, config, context, metadata, name, description, created_at
                    FROM assistant_versions
                    WHERE {where_clause}
                    ORDER BY version DESC
                    LIMIT %s OFFSET %s""",
                params,
            )
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def assistant_set_latest(
        self, assistant_id: str, version: int
    ) -> dict | None:
        async with self._cursor() as cur:
            await cur.execute(
                """SELECT graph_id, config, context, metadata, name, description
                   FROM assistant_versions
                   WHERE assistant_id = %s AND version = %s""",
                (assistant_id, version),
            )
            version_row = await cur.fetchone()
            if version_row is None:
                return None

            await cur.execute(
                f"""UPDATE assistants
                    SET graph_id = %s, config = %s, context = %s, metadata = %s,
                        name = %s, description = %s, version = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE assistant_id = %s
                    RETURNING {self.SELECT_COLUMNS}""",
                (
                    version_row["graph_id"],
                    version_row["config"],
                    version_row["context"],
                    version_row["metadata"],
                    version_row["name"],
                    version_row["description"],
                    version,
                    assistant_id,
                ),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    # ── Vector helpers ───────────────────────────────────────────────────

    async def _upsert_assistant_vectors(
        self,
        cur: AsyncCursor[DictRow],
        assistant_id: str,
        metadata: dict,
    ) -> None:
        texts: list[str] = []
        field_names: list[str] = []
        if metadata:
            texts.append(orjson.dumps(metadata).decode("utf-8"))
            field_names.append("metadata")

        if not texts:
            return

        vectors = await self.embeddings.aembed_documents(texts)
        dims = int(self.index_config["dims"])
        ann_config = self.index_config.get("ann_index_config", {})
        vector_type = ann_config.get("vector_type", "vector")
        for field_name, vector in zip(field_names, vectors):
            await cur.execute(
                f"""INSERT INTO assistant_vectors (assistant_id, field_name, embedding, created_at, updated_at)
                    VALUES (%s, %s, %s::{vector_type}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (assistant_id, field_name) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP""",
                (assistant_id, field_name, vector),
            )

    # ── Static helpers ───────────────────────────────────────────────────

    @staticmethod
    def _build_jsonb_filter(
        conditions: list[str],
        params: list[Any],
        column: str,
        filter_dict: dict[str, Any],
    ) -> None:
        import json

        for key, value in filter_dict.items():
            if isinstance(value, dict):
                for op_name, val in value.items():
                    condition, p = AsyncPostgresAssistant._get_filter_condition(
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
        import json

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
        allowed = {
            "assistant_id": "assistant_id",
            "graph_id": "graph_id",
            "name": "name",
            "created_at": "created_at",
            "updated_at": "updated_at",
        }
        return allowed.get(sort_by, "updated_at")
