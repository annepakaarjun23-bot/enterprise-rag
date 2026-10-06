from __future__ import annotations

import time

from openai import APIError, OpenAI, RateLimitError

from rag.settings import settings

_client = OpenAI(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url,
)

MODEL = settings.embedding_model
DIMENSIONS = settings.embedding_dimensions
BATCH_SIZE = settings.embedding_batch_size


def embed_texts(texts: list[str], max_retries: int = 5) -> list[list[float]]:
    if not texts:
        return []

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = _client.embeddings.create(model=MODEL, input=texts)
            ordered = sorted(resp.data, key=lambda d: d.index)
            return [d.embedding for d in ordered]
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except APIError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_batch(
    texts: list[str], batch_size: int | None = None
) -> list[list[float]]:
    if not texts:
        return []
    bs = batch_size or BATCH_SIZE
    out: list[list[float]] = []
    for i in range(0, len(texts), bs):
        out.extend(embed_texts(texts[i : i + bs]))
    return out