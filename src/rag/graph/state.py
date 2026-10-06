from __future__ import annotations

from typing import Any, TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str
    parent_chunk_id: str
    text: str
    score: float
    library: str
    version: str
    source_path: str
    source_url: str
    heading_path: str
    has_code: bool


class GraphState(TypedDict, total=False):

    user_query: str

    retrieval_mode: str        
    retrieval_top_k: int
    use_automerge: bool

    retrieved_chunks: list[RetrievedChunk]
    retrieval_error: str

    answer: str
    generation_error: str

    error: str