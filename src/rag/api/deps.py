from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from rag.indexing.qdrant_store import get_client as _build_qdrant_client


def get_graph(request: Request):
    return request.app.state.graph


def get_pg_pool(request: Request):
    return request.app.state.pool


@lru_cache(maxsize=1)
def get_qdrant_client():
    return _build_qdrant_client()