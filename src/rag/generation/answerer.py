from __future__ import annotations

from pathlib import Path

from rag.llm_client import DEFAULT_LLM_MODEL, llm_client

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "graph" / "prompts" / "generator_prompt.md"
)
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")
PROMPT_HEAD, _AFTER_CONTEXT = PROMPT_TEMPLATE.split("{context}", 1)
PROMPT_MIDDLE, PROMPT_TAIL = _AFTER_CONTEXT.split("{question}", 1)
TEMPERATURE = 0.2
NO_CONTEXT = "(no context retrieved)"


def format_context(chunks: list[str]) -> str:
    if not chunks:
        return NO_CONTEXT
    return "\n\n---\n\n".join(f"[S{i}]\n{text}" for i, text in enumerate(chunks, 1))


def render_prompt(query: str, context_chunks: list[str]) -> str:
    return (
        PROMPT_HEAD
        + format_context(context_chunks)
        + PROMPT_MIDDLE
        + query
        + PROMPT_TAIL
    )


async def generate_answer(query: str, context_chunks: list[str]) -> str:
    response = await llm_client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role": "user", "content": render_prompt(query, context_chunks)}],
        temperature=TEMPERATURE,
    )
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""