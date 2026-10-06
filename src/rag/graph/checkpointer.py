from __future__ import annotations

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from rag.settings import settings

_PSYCOPG_URL = settings.database_url.replace("postgresql+psycopg://", "postgresql://")


async def get_checkpointer() -> tuple[AsyncPostgresSaver, AsyncConnectionPool]:
    pool = AsyncConnectionPool(
        conninfo=_PSYCOPG_URL,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()   
    return checkpointer, pool