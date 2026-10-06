from __future__ import annotations

from typing import Any

from llama_index.core.vector_stores.types import VectorStoreQueryResult


def reciprocal_rank_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,     
    top_k: int = 10,
    k: int = 60,               
) -> VectorStoreQueryResult:
    dense_nodes = list(dense_result.nodes or [])
    sparse_nodes = list(sparse_result.nodes or [])

    # nothing to fuse -> return whichever side has data
    if not dense_nodes and not sparse_nodes:
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    if not sparse_nodes:
        trimmed = dense_nodes[:top_k]
        return VectorStoreQueryResult(
            nodes=trimmed,
            similarities=list(dense_result.similarities or [])[:top_k],
            ids=[n.node_id for n in trimmed],
        )
    if not dense_nodes:
        trimmed = sparse_nodes[:top_k]
        return VectorStoreQueryResult(
            nodes=trimmed,
            similarities=list(sparse_result.similarities or [])[:top_k],
            ids=[n.node_id for n in trimmed],
        )

    scores: dict[str, float] = {}
    nodes: dict[str, Any] = {}

    for rank, node in enumerate(dense_nodes):
        scores[node.node_id] = scores.get(node.node_id, 0.0) + 1.0 / (k + rank + 1)
        nodes[node.node_id] = node

    for rank, node in enumerate(sparse_nodes):
        scores[node.node_id] = scores.get(node.node_id, 0.0) + 1.0 / (k + rank + 1)
        nodes.setdefault(node.node_id, node)

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return VectorStoreQueryResult(
        nodes=[nodes[nid] for nid, _ in ordered],
        similarities=[float(s) for _, s in ordered],
        ids=[nid for nid, _ in ordered],
    )