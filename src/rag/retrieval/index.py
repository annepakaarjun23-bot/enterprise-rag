from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore

from rag.indexing.qdrant_store import get_vector_store
from rag.retrieval.embedding_model import AirouterEmbedding

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCSTORE_DIR = REPO_ROOT / "data" / "chunks" / "docstore"


@lru_cache(maxsize=1)
def get_embed_model() -> AirouterEmbedding:
    return AirouterEmbedding()


@lru_cache(maxsize=1)
def get_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(),
        embed_model=get_embed_model(),
    )


@lru_cache(maxsize=1)
def get_docstore() -> SimpleDocumentStore:
    if not DOCSTORE_DIR.exists():
        raise RuntimeError(
            f"docstore missing at {DOCSTORE_DIR}. Run chunk_all.py first."
        )
    return SimpleDocumentStore.from_persist_dir(str(DOCSTORE_DIR))


@lru_cache(maxsize=1)
def get_storage_context() -> StorageContext:
    return StorageContext.from_defaults(docstore=get_docstore())