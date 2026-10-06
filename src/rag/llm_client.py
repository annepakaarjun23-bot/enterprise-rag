from __future__ import annotations

from openai import AsyncOpenAI

from rag.settings import settings

llm_client = AsyncOpenAI(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url,
)

DEFAULT_LLM_MODEL = "meta/llama-3.1-8b"