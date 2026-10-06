from __future__ import annotations

from llama_index.core.query_engine import RetrieverQueryEngine

from rag.retrieval.reranker import get_reranker
from rag.retrieval.retriever import RetrieverConfig, get_retriever


def get_query_engine(
    config: RetrieverConfig | None = None,
    *,
    rerank: bool = True,
    rerank_top_n: int = 5,
    rerank_model: str | None = None,
) -> RetrieverQueryEngine:
    retriever = get_retriever(config)

    postprocessors = []
    if rerank:
        postprocessors.append(
            get_reranker(
                model=rerank_model or None,
                top_n=rerank_top_n,
            )
        )

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=postprocessors,
        response_mode="compact",
    )