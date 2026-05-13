import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from psycopg import AsyncConnection, AsyncCursor
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from langgraph.checkpoint.postgres import _ainternal

from ._sql import split_sql_statements

logger = logging.getLogger(__name__)


class AsyncPostgresCron:
    """Async Postgres-backed cron job persistence.

    Manages CRUD for the `crons` table independently from thread/run storage.
    """

    MIGRATIONS: list[str] = [
        """\
CREATE TABLE IF NOT EXISTS crons (
    cron_id TEXT PRIMARY KEY,
    assistant_id TEXT NOT NULL,
    thread_id TEXT DEFAULT NULL,
    schedule TEXT NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    enabled BOOLEAN DEFAULT TRUE,
    on_run_completed TEXT DEFAULT NULL,
    payload JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    next_run_date TIMESTAMP WITH TIME ZONE,
    user_id TEXT DEFAULT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_crons_assistant_id ON crons(assistant_id);
CREATE INDEX IF NOT EXISTS idx_crons_thread_id ON crons(thread_id);
CREATE INDEX IF NOT EXISTS idx_crons_enabled ON crons(enabled);
CREATE INDEX IF NOT EXISTS idx_crons_user_id ON crons(user_id);
""",
    ]

    __slots__ = ("conn", "lock")

    def __init__(self, conn: _ainternal.Conn) -> None:
        self.conn = conn
        self.lock = asyncio.Lock()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
    ) -> AsyncIterator["AsyncPostgresCron"]:
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            yield cls(conn=conn)

    async def setup(self) -> None:
        """Create the crons table and indexes if they don't already exist."""

        async def _get_version(cur, table: str) -> int:
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
            version = await _get_version(cur, table="cron_migrations")
            for v, sql in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                try:
                    for stmt in split_sql_statements(sql):
                        await cur.execute(stmt)
                    await cur.execute(
                        "INSERT INTO cron_migrations (v) VALUES (%s)", (v,)
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to apply migration {v}.\nSql={sql}\nError={e}"
                    )
                    raise

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[AsyncCursor[DictRow]]:
        async with _ainternal.get_connection(self.conn) as conn:
            async with self.lock:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    # ── Cron operations ───────────────────────────────────────────────────

    async def cron_put(
        self,
        cron_id: str,
        *,
        assistant_id: str,
        thread_id: str | None = None,
        schedule: str,
        end_time: datetime | None = None,
        enabled: bool = True,
        on_run_completed: str | None = None,
        payload: dict | None = None,
        metadata: dict | None = None,
        next_run_date: datetime | None = None,
        user_id: str | None = None,
    ) -> dict:
        """Insert or update a cron record. Returns the full cron row."""
        payload = payload or {}
        metadata = metadata or {}

        async with self._cursor() as cur:
            await cur.execute(
                """INSERT INTO crons (cron_id, assistant_id, thread_id, schedule, end_time,
                                      enabled, on_run_completed, payload, metadata,
                                      next_run_date, user_id, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                   ON CONFLICT (cron_id) DO UPDATE
                   SET assistant_id = EXCLUDED.assistant_id,
                       thread_id = EXCLUDED.thread_id,
                       schedule = EXCLUDED.schedule,
                       end_time = EXCLUDED.end_time,
                       enabled = EXCLUDED.enabled,
                       on_run_completed = EXCLUDED.on_run_completed,
                       payload = EXCLUDED.payload,
                       metadata = EXCLUDED.metadata,
                       next_run_date = EXCLUDED.next_run_date,
                       user_id = EXCLUDED.user_id,
                       updated_at = CURRENT_TIMESTAMP
                   RETURNING cron_id, assistant_id, thread_id, schedule, end_time,
                             enabled, on_run_completed, payload, metadata,
                             next_run_date, user_id, created_at, updated_at""",
                (
                    cron_id, assistant_id, thread_id, schedule, end_time,
                    enabled, on_run_completed, Jsonb(payload), Jsonb(metadata),
                    next_run_date, user_id,
                ),
            )
            return dict(await cur.fetchone())

    async def cron_get(self, cron_id: str) -> dict | None:
        """Get a single cron by ID."""
        async with self._cursor() as cur:
            await cur.execute(
                """SELECT cron_id, assistant_id, thread_id, schedule, end_time,
                          enabled, on_run_completed, payload, metadata,
                          next_run_date, user_id, created_at, updated_at
                   FROM crons WHERE cron_id = %s""",
                (cron_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def cron_update(
        self,
        cron_id: str,
        *,
        schedule: str | None = None,
        end_time: datetime | None = None,
        enabled: bool | None = None,
        on_run_completed: str | None = None,
        payload: dict | None = None,
        metadata: dict | None = None,
        next_run_date: datetime | None = None,
    ) -> dict | None:
        """Update a cron's fields. Only non-None values are updated. Returns the full row."""
        set_parts = ["updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = []

        if schedule is not None:
            set_parts.append("schedule = %s")
            params.append(schedule)
        if end_time is not None:
            set_parts.append("end_time = %s")
            params.append(end_time)
        if enabled is not None:
            set_parts.append("enabled = %s")
            params.append(enabled)
        if on_run_completed is not None:
            set_parts.append("on_run_completed = %s")
            params.append(on_run_completed)
        if payload is not None:
            set_parts.append("payload = %s")
            params.append(Jsonb(payload))
        if metadata is not None:
            set_parts.append("metadata = %s")
            params.append(Jsonb(metadata))
        if next_run_date is not None:
            set_parts.append("next_run_date = %s")
            params.append(next_run_date)

        if len(set_parts) == 1:
            return await self.cron_get(cron_id)

        params.append(cron_id)

        async with self._cursor() as cur:
            await cur.execute(
                f"""UPDATE crons SET {", ".join(set_parts)}
                    WHERE cron_id = %s
                    RETURNING cron_id, assistant_id, thread_id, schedule, end_time,
                              enabled, on_run_completed, payload, metadata,
                              next_run_date, user_id, created_at, updated_at""",
                params,
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def cron_delete(self, cron_id: str) -> None:
        """Delete a cron by ID."""
        async with self._cursor() as cur:
            await cur.execute(
                "DELETE FROM crons WHERE cron_id = %s",
                (cron_id,),
            )

    async def cron_search(
        self,
        *,
        assistant_id: str | None = None,
        thread_id: str | None = None,
        enabled: bool | None = None,
        user_id: str | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> list[dict]:
        """Search crons with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if assistant_id is not None:
            conditions.append("assistant_id = %s")
            params.append(assistant_id)
        if thread_id is not None:
            conditions.append("thread_id = %s")
            params.append(thread_id)
        if enabled is not None:
            conditions.append("enabled = %s")
            params.append(enabled)
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        sort_column = self._resolve_cron_sort_column(sort_by)
        direction = "DESC" if sort_order and sort_order.lower() == "desc" else "ASC"

        params.extend([limit, offset])

        async with self._cursor() as cur:
            await cur.execute(
                f"""SELECT cron_id, assistant_id, thread_id, schedule, end_time,
                           enabled, on_run_completed, payload, metadata,
                           next_run_date, user_id, created_at, updated_at
                    FROM crons
                    WHERE {where_clause}
                    ORDER BY {sort_column} {direction}
                    LIMIT %s OFFSET %s""",
                params,
            )
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def cron_count(
        self,
        *,
        assistant_id: str | None = None,
        thread_id: str | None = None,
        enabled: bool | None = None,
        user_id: str | None = None,
    ) -> int:
        """Count crons matching filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if assistant_id is not None:
            conditions.append("assistant_id = %s")
            params.append(assistant_id)
        if thread_id is not None:
            conditions.append("thread_id = %s")
            params.append(thread_id)
        if enabled is not None:
            conditions.append("enabled = %s")
            params.append(enabled)
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        async with self._cursor() as cur:
            await cur.execute(
                f"SELECT COUNT(*) AS cnt FROM crons WHERE {where_clause}",
                params,
            )
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    @staticmethod
    def _resolve_cron_sort_column(sort_by: str | None) -> str:
        """Map cron sort field name to safe column reference."""
        allowed = {
            "cron_id": "cron_id",
            "assistant_id": "assistant_id",
            "thread_id": "thread_id",
            "created_at": "created_at",
            "updated_at": "updated_at",
            "next_run_date": "next_run_date",
            "end_time": "end_time",
        }
        return allowed.get(sort_by, "updated_at")
