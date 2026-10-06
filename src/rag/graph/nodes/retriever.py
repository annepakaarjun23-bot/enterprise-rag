from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from langgraph.config import get_stream_writer

from rag.graph.state import GraphState, RetrievedChunk
from rag.retrieval.retriever import RetrieverConfig, get_retriever

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_retriever(mode: str, top_k: int, automerge: bool):
    return get_retriever(
        RetrieverConfig(mode=mode, top_k=top_k, automerge=automerge)
    )


def _to_chunk(node_with_score) -> RetrievedChunk:
    m = node_with_score.node.metadata
    return {
        "chunk_id": m.get("chunk_id", node_with_score.node.node_id),
        "parent_chunk_id": m.get("parent_chunk_id", ""),
        "text": node_with_score.get_content(),
        "score": float(node_with_score.score or 0.0),
        "library": m.get("library", ""),
        "version": m.get("version", ""),
        "source_path": m.get("source_path", ""),
        "source_url": m.get("source_url", ""),
        "heading_path": m.get("heading_path", ""),
        "has_code": bool(m.get("has_code", False)),
    }


async def retriever_node(state: GraphState) -> dict:
    writer = get_stream_writer()
    query = state["user_query"]

    mode = state.get("retrieval_mode", "dense")
    top_k = state.get("retrieval_top_k", 10)
    automerge = state.get("use_automerge", True)

    try:
        writer({
            "node": "retriever",
            "status": "started",
            "message": f"Retrieving ({mode}, top_k={top_k}) for: {query}",
        })

        retriever = _get_retriever(mode, top_k, automerge)
        nodes = await asyncio.to_thread(retriever.retrieve, query)
        chunks = [_to_chunk(nws) for nws in nodes]

        for i, chunk in enumerate(chunks):
            writer({
                "node": "retriever",
                "type": "chunk",
                "index": i,
                "chunk_id": chunk["chunk_id"],
                "heading_path": chunk["heading_path"],
                "source_path": chunk["source_path"],
                "score": chunk["score"],
                "preview": chunk["text"][:200],
            })

        writer({
            "node": "retriever",
            "status": "completed",
            "count": len(chunks),
            "message": f"Retrieved {len(chunks)} chunks.",
        })
        return {"retrieved_chunks": chunks}

    except Exception as e:  # noqa: BLE001
        logger.exception("retriever_node failed")
        writer({"node": "retriever", "status": "error", "message": str(e)})
        return {"error": str(e), "retrieval_error": str(e)}