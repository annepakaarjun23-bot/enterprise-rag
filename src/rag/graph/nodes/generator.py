from __future__ import annotations

import logging
from pathlib import Path

from langgraph.config import get_stream_writer

from rag.llm_client import DEFAULT_LLM_MODEL, llm_client
from rag.graph.state import GraphState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generator_prompt.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with source headers so the model can cite."""
    if not chunks:
        return "(no context retrieved)"
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.get('library')}/{c.get('version')} :: {c.get('heading_path') or '(no heading)'}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)

def _render_prompt(template: str, context: str, question: str) -> str:
    return (
        template
        .replace("{context}", context)
        .replace("{question}", question)
    )

async def generator_node(state: GraphState) -> dict:
    writer = get_stream_writer()

    if state.get("error"):
        writer({"node": "generator", "status": "skipped", "reason": state["error"]})
        return {}

    query = state["user_query"]
    chunks = state.get("retrieved_chunks", [])
    logger.info("generator: formatting prompt (chunks=%d)", len(chunks))

    try:
        prompt = _render_prompt(
            _PROMPT_TEMPLATE,
            context=_format_context(chunks),
            question=query,
        )

        writer({
            "node": "generator",
            "status": "started",
            "message": "Generating answer...",
        })

        response = await llm_client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.2,
        )

        full = []
        async for event in response:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                full.append(token)
                writer({"node": "generator", "type": "token", "content": token})

        answer = "".join(full)
        writer({
            "node": "generator",
            "status": "completed",
            "message": "Generation complete.",
            "length": len(answer),
        })
        return {"answer": answer}

    except Exception as e:  # noqa: BLE001
        logger.exception("generator_node failed")
        writer({"node": "generator", "status": "error", "message": str(e)})
        return {"error": str(e), "generation_error": str(e)}