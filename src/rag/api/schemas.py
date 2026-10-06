from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        description="Client-supplied session key. Pass the same value on "
        "subsequent turns to keep conversational memory via the checkpointer.",
    )
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    use_automerge: bool = True


class ChatChunk(BaseModel):
    chunk_id: str
    parent_chunk_id: str = ""
    text: str
    score: float
    library: str = ""
    version: str = ""
    source_path: str = ""
    source_url: str = ""
    heading_path: str = ""
    has_code: bool = False


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    chunks: list[ChatChunk]
    duration_ms: int


class HealthResponse(BaseModel):
    status: str
    checks: dict