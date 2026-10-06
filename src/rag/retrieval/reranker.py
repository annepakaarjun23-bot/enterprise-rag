from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_HF_CACHE = REPO_ROOT / "data" / "models" / "hf"
_HF_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(_HF_CACHE))

from llama_index.core.postprocessor import SentenceTransformerRerank  # noqa: E402

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=4)
def get_reranker(model: str = DEFAULT_RERANK_MODEL, top_n: int = 5) -> SentenceTransformerRerank:
    return SentenceTransformerRerank(
        model=model,
        top_n=top_n,
        device="cpu",
    )