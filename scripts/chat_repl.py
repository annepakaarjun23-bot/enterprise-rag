from __future__ import annotations

import asyncio
import io
import sys
import uuid
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, io.UnsupportedOperation):
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag._compat import run_async, setup_windows_event_loop  # noqa: E402

setup_windows_event_loop()

from rag.graph.checkpointer import get_checkpointer  # noqa: E402
from rag.graph.graph_build import build_rag_graph

async def stream_one(graph, query: str, thread_id: str) -> None:
    inputs = {
        "user_query": query,
        "retrieval_mode": "dense",
        "retrieval_top_k": 10,
        "use_automerge": True,
        "error": None,
        "retrieval_error": None,
        "generation_error": None,
        "answer": "",
        "retrieved_chunks": [],
    }
    config = {"configurable": {"thread_id": thread_id}}

    print()
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
            status = data.get("status")
            kind = data.get("type")

            if status == "started":
                print(f"  [{node}] {data.get('message', '...')}")
            elif status == "completed":
                print(f"  [{node}] {data.get('message', 'done')}")
            elif status == "skipped":
                print(f"  [{node}] skipped: {data.get('reason', '')}")
            elif status == "error":
                print(f"  [{node}] ERROR: {data.get('message')}")
            elif kind == "chunk":
                print(
                    f"      chunk {data['index']:2d}  "
                    f"score={data['score']:.3f}  "
                    f"{data['heading_path'][:70]}"
                )
            elif kind == "token":
                sys.stdout.write(data["content"])
                sys.stdout.flush()

        elif t == "updates":
            # v2 shape: data is {node_name: delta_or_None}
            if not isinstance(data, dict):
                continue
            for node_name, state_delta in data.items():
                if not isinstance(state_delta, dict):
                    continue
                if node_name == "generator" and state_delta.get("answer"):
                    print()
                    print(f"  [generator] done ({len(state_delta['answer'])} chars)")

    print()


async def main() -> None:
    checkpointer, pool = await get_checkpointer()
    try:
        graph = build_rag_graph(checkpointer)
        # fresh thread per process -> no state carryover from prior sessions
        thread_id = f"cli-{uuid.uuid4().hex[:8]}"
        print(f"RAG REPL (thread={thread_id}) -- type 'quit' to exit.")
        while True:
            query = input("\n> ").strip()
            if not query or query.lower() in {"quit", "exit"}:
                break
            await stream_one(graph, query, thread_id)
    finally:
        await pool.close()

        

if __name__ == "__main__":
    run_async(main())  