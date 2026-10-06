from __future__ import annotations

from llama_index.core.embeddings import BaseEmbedding

from rag.ingestion.embedder import MODEL, embed_texts


class AirouterEmbedding(BaseEmbedding):
    model_name: str = MODEL
    embed_batch_size: int = 32

    @classmethod
    def class_name(cls) -> str:
        return "AirouterEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed_texts([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)