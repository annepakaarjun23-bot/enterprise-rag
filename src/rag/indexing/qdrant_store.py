from __future__ import annotations

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.settings import settings

DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse-new"


def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def get_vector_store(client: QdrantClient | None = None) -> QdrantVectorStore:
    client = client or get_client()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_leaves,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        hybrid_fusion_fn=reciprocal_rank_fusion,
    )


def collection_exists() -> bool:
    return get_client().collection_exists(settings.qdrant_collection_leaves)


def collection_summary() -> dict:
    client = get_client()
    if not client.collection_exists(settings.qdrant_collection_leaves):
        return {"exists": False}
    info = client.get_collection(settings.qdrant_collection_leaves)
    vectors = info.config.params.vectors
    sparse = getattr(info.config.params, "sparse_vectors", None)

    def _vec_names(cfg):
        if cfg is None:
            return []
        if isinstance(cfg, dict):
            return list(cfg.keys())
        return [getattr(cfg, "size", "single")]

    return {
        "exists": True,
        "points_count": info.points_count,
        "dense_vector_names": _vec_names(vectors),
        "sparse_vector_names": list(sparse.keys()) if isinstance(sparse, dict) else [],
    }


def drop_collection() -> None:
    client = get_client()
    if client.collection_exists(settings.qdrant_collection_leaves):
        client.delete_collection(settings.qdrant_collection_leaves)