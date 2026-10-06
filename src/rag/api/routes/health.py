from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from sqlalchemy import text

from rag.api.deps import get_qdrant_client
from rag.db.session import session_scope
from rag.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(qdrant=Depends(get_qdrant_client)) -> dict:
    checks: dict[str, dict] = {}

    try:
        def _qdrant_check() -> int:
            info = qdrant.get_collection(settings.qdrant_collection_leaves)
            return info.points_count

        points = await asyncio.to_thread(_qdrant_check)
        checks["qdrant"] = {"ok": True, "points": points}
    except Exception as e:  
        checks["qdrant"] = {"ok": False, "error": str(e)}

    try:
        def _pg_check() -> int:
            with session_scope() as s:
                return s.execute(text("SELECT 1")).scalar_one()

        await asyncio.to_thread(_pg_check)
        checks["postgres"] = {"ok": True}
    except Exception as e:  
        checks["postgres"] = {"ok": False, "error": str(e)}

    ok = all(c["ok"] for c in checks.values())
    return {"status": "ok" if ok else "degraded", "checks": checks}