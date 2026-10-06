from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.vector_stores.types import MetadataFilters

from rag.retrieval.index import get_index, get_storage_context


@dataclass
class RetrieverConfig:
    mode: str = "dense" 
    top_k: int = 10                 
    sparse_top_k: int = 10                  
    automerge: bool = True
    filters: MetadataFilters | None = None
    automerge_threshold: float = 0.5


def get_retriever(config: RetrieverConfig | None = None):
    config = config or RetrieverConfig()
    index = get_index()

    kwargs: dict = {"similarity_top_k": config.top_k}
    if config.filters is not None:
        kwargs["filters"] = config.filters

    if config.mode == "hybrid":
        kwargs["vector_store_query_mode"] = "hybrid"
        kwargs["sparse_top_k"] = config.sparse_top_k
    elif config.mode == "dense":
        kwargs["vector_store_query_mode"] = "default"
    else:
        raise ValueError(f"unknown mode: {config.mode!r}")

    base = index.as_retriever(**kwargs)

    if not config.automerge:
        return base

    return AutoMergingRetriever(
        vector_retriever=base,
        storage_context=get_storage_context(),
        simple_ratio_thresh=config.automerge_threshold,
        verbose=False,
    )