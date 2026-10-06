from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.retrieval.reranker import get_reranker
from rag.retrieval.retriever import RetrieverConfig, get_retriever


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        # convenience: allow r["chunk_id"] as alias for r.doc_id
        if key == "chunk_id":
            return self.doc_id
        if key == "text":
            return self.text
        if key == "score":
            return self.score
        if key == "metadata":
            return self.metadata
        raise KeyError(key)


def retrieve(
    query: str,
    k: int = 10,
    *,
    mode: str = "dense",
    automerge: bool = True,
    rerank: bool = False,
    rerank_top_n: int = 5,
    filters: Any = None,
) -> list[RetrievedDoc]:
    retrieve_top_k = max(k * 3, 20) if rerank else k

    config = RetrieverConfig(
        mode=mode,
        top_k=retrieve_top_k,
        sparse_top_k=max(retrieve_top_k, 20),
        automerge=automerge,
        filters=filters,
    )
    retriever = get_retriever(config)
    results = retriever.retrieve(query)

    # optional reranking
    if rerank:
        reranker = get_reranker(top_n=rerank_top_n)
        results = reranker.postprocess_nodes(results, query_str=query)

    docs: list[RetrievedDoc] = []
    for nws in results:
        m = nws.node.metadata
        docs.append(
            RetrievedDoc(
                doc_id=m.get("chunk_id", nws.node.node_id),
                text=nws.node.text,
                score=float(nws.score or 0.0),
                metadata={
                    "level": m.get("level"),
                    "is_leaf": m.get("is_leaf"),
                    "library": m.get("library"),
                    "version": m.get("version"),
                    "heading_path": m.get("heading_path"),
                    "source_path": m.get("source_path"),
                    "has_code": m.get("has_code"),
                },
            )
        )
    return docs