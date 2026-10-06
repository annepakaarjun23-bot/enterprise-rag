from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from rag.api.deps import get_graph
from rag.api.schemas import ChatChunk, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def _build_inputs(req: ChatRequest) -> dict:
    return {
        "user_query": req.message,
        "retrieval_mode": req.retrieval_mode,
        "retrieval_top_k": req.top_k,
        "use_automerge": req.use_automerge,
        "error": None,
        "retrieval_error": None,
        "generation_error": None,
        "answer": "",
        "retrieved_chunks": [],
    }


def _extract_updates(data, answer: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Pull the final answer and retrieved chunks from an updates event."""
    if not isinstance(data, dict):
        return answer, chunks
    for node, delta in data.items():
        if not isinstance(delta, dict):
            continue
        if node == "generator" and delta.get("answer"):
            answer = delta["answer"]
        if node == "retriever" and delta.get("retrieved_chunks"):
            chunks = delta["retrieved_chunks"]
    return answer, chunks


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, graph=Depends(get_graph)) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    inputs = _build_inputs(req)

    t0 = time.perf_counter()
    answer = ""
    chunks: list[dict] = []

    async for part in graph.astream(
        inputs,
        config=config,
        stream_mode=["updates"],
        version="v2",
    ):
        if part["type"] == "updates":
            answer, chunks = _extract_updates(part["data"], answer, chunks)

    duration_ms = int((time.perf_counter() - t0) * 1000)

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        chunks=[ChatChunk(**c) for c in chunks],
        duration_ms=duration_ms,
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def chat_stream(req: ChatRequest, graph=Depends(get_graph)):
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    inputs = _build_inputs(req)

    async def event_gen():
        t0 = time.perf_counter()
        answer_parts: list[str] = []
        chunks: list[dict] = []

        yield _sse({"type": "session", "session_id": session_id})

        try:
            async for part in graph.astream(
                inputs,
                config=config,
                stream_mode=["updates", "custom"],
                version="v2",
            ):
                t = part["type"]
                data = part["data"]

                if t == "custom":
                    node = data.get("node")
                    kind = data.get("type")
                    status = data.get("status")

                    if kind == "token":
                        answer_parts.append(data["content"])
                        yield _sse({"type": "token", "text": data["content"]})
                    elif kind == "chunk":
                        # forward retrieval event to the client
                        yield _sse({"type": "chunk", "chunk": data})
                    elif status == "started":
                        yield _sse({
                            "type": "stage",
                            "node": node,
                            "message": data.get("message", ""),
                        })
                    elif status == "completed":
                        yield _sse({"type": "stage_done", "node": node})
                    elif status == "error":
                        yield _sse({
                            "type": "error",
                            "node": node,
                            "message": data.get("message", ""),
                        })

                elif t == "updates":
                    answer, chunks = _extract_updates(data, "", chunks)
                    if answer and not answer_parts:
                        answer_parts = [answer]

        except asyncio.CancelledError:
            # client disconnected; stop cleanly
            raise
        except Exception as e:  
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

        duration_ms = int((time.perf_counter() - t0) * 1000)
        yield _sse({
            "type": "done",
            "session_id": session_id,
            "answer": "".join(answer_parts),
            "chunks": chunks,
            "duration_ms": duration_ms,
        })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  
        },
    )