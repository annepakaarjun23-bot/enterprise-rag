from __future__ import annotations

import io
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, io.UnsupportedOperation):
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llama_index.core.schema import NodeRelationship  # noqa: E402
from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from rag.retrieval.retriever import RetrieverConfig, get_retriever  # noqa: E402


PASS = 0
FAIL = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))


def print_top(results, n: int = 3) -> None:
    for i, nws in enumerate(results[:n]):
        m = nws.node.metadata
        path = m.get("heading_path", "")
        lib = m.get("library", "")
        ver = m.get("version", "")
        score = nws.score
        print(f"    {i+1}. [{score:.4f}] {lib}/{ver}  {path[:90]}")


# ------------------------------------------------------------------
# 1. Canary -- embedder wired correctly
# ------------------------------------------------------------------
print("=" * 78)
print("1) canary: 'how do I add memory to a LangGraph agent?'")
print("=" * 78)
retriever = get_retriever(RetrieverConfig(mode="hybrid", top_k=5, automerge=False))
results = retriever.retrieve("how to create workflow in langgraph")
print_top(results, 5)

hits = [
    r for r in results
    if any(k in (r.node.metadata.get("heading_path", "") or "").lower()
           for k in ("memory", "persistence", "checkpointer", "store"))
]
report("canary returns memory/persistence-related nodes", len(hits) >= 2,
       f"{len(hits)} of {len(results)} match")
print()


# # ------------------------------------------------------------------
# # 2. Filter -- library + version
# # ------------------------------------------------------------------
print("=" * 78)
print("2) filter: langgraph 1.x only")
print("=" * 78)
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="library", value="langgraph"),
        MetadataFilter(key="version", value="1.x"),
    ],
    condition=FilterCondition.AND,
)
retriever = get_retriever(
    RetrieverConfig(mode="dense", top_k=5, automerge=False, filters=filters)
)
results = retriever.retrieve("how do I add persistence?")
print_top(results, 5)

all_ok = all(
    r.node.metadata.get("library") == "langgraph"
    and r.node.metadata.get("version") == "1.x"
    for r in results
)
report("all returned nodes match library=langgraph, version=1.x", all_ok,
       f"{len(results)} results")
print()


# # ------------------------------------------------------------------
# # 3. Exact-term query
# # ------------------------------------------------------------------
print("=" * 78)
print("3) exact term: 'create_agent import statement'")
print("=" * 78)
retriever = get_retriever(RetrieverConfig(mode="dense", top_k=5, automerge=False))
results = retriever.retrieve("create_agent import statement")
print_top(results, 5)

contains = sum(1 for r in results if "create_agent" in r.node.text)
report("at least one result literally contains 'create_agent'", contains >= 1,
       f"{contains} of {len(results)}")
print()


# ------------------------------------------------------------------
# 4. Automerge -- parents show up when leaves cluster
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 4. Automerge -- force a cluster, verify a parent appears
# ----------------
# --------------------------------------------------
from rag.retrieval.index import get_docstore
print("=" * 78)
print("4) automerge on vs off (top_k=15, per-node diff)")
print("=" * 78)
q = "how do I add short-term memory in LangGraph?"

no_merge = get_retriever(RetrieverConfig(mode="dense", top_k=15, automerge=False))
off_results = no_merge.retrieve(q)

merge = get_retriever(RetrieverConfig(mode="dense", top_k=15, automerge=True))
on_results = merge.retrieve(q)

off_ids = [r.node.node_id for r in off_results]
on_ids  = [r.node.node_id for r in on_results]

print(f"  top_k=15")
print(f"  off: {len(off_results)} nodes, max_tokens={max((r.node.metadata.get('token_count',0) for r in off_results), default=0)}")
print(f"  on:  {len(on_results)} nodes, max_tokens={max((r.node.metadata.get('token_count',0) for r in on_results), default=0)}")

added   = [i for i in on_ids if i not in off_ids]
removed = [i for i in off_ids if i not in on_ids]
print(f"  nodes added by automerge:   {len(added)}")
print(f"  nodes removed by automerge: {len(removed)}")

if added:
    print("  newly introduced nodes (these should be parents):")
    for i in on_ids:
        if i in added:
            m = next(r.node.metadata for r in on_results if r.node.node_id == i)
            print(f"    - tokens={m.get('token_count')}  {m.get('heading_path','')[:80]}")

report("automerge changed the result set (at least one node swapped)",
       len(added) > 0 or len(removed) > 0,
       f"added={len(added)} removed={len(removed)}")
print()

# diagnostic: are PARENT relationships present on retrieved nodes at all?
print("  --- diagnostic: PARENT relationship presence ---")
from llama_index.core.schema import NodeRelationship
first = off_results[0].node if off_results else None
if first is not None:
    has_parent = NodeRelationship.PARENT in first.relationships
    parent_id = first.relationships[NodeRelationship.PARENT].node_id if has_parent else None
    print(f"  node {first.node_id} has PARENT: {has_parent}")
    print(f"  parent_id (UUID):  {parent_id}")

    if has_parent:
        ds = get_docstore()
        resolved = ds.get_node(parent_id)
        print(f"  resolves in docstore: {resolved is not None}")
        if resolved:
            print(f"  parent tokens:  {resolved.metadata.get('token_count')}")
            print(f"  parent heading: {resolved.metadata.get('heading_path','')[:80]}")
print()

# With automerge on, at least one returned node should be a parent-sized node
# (bigger token_count than what a single leaf would have).
off_max = max((r.node.metadata.get("token_count", 0) for r in off_results), default=0)
on_max = max((r.node.metadata.get("token_count", 0) for r in on_results), default=0)
report("automerge returns at least one larger node than dense-only would",
       on_max >= off_max, f"on_max={on_max} off_max={off_max}")
print()


# # ------------------------------------------------------------------
# # 5. Empty / OOD query doesn't crash
# # ------------------------------------------------------------------
print("=" * 78)
print("5) out-of-distribution query: 'asdfghjkl'")
print("=" * 78)
try:
    results = retriever.retrieve("asdfghjkl qwertyuiop")
    print_top(results, 3)
    report("does not crash on OOD input", True, f"{len(results)} results")
except Exception as e:  # noqa: BLE001
    report("does not crash on OOD input", False, f"{type(e).__name__}: {e}")
print()


# # ------------------------------------------------------------------
# # summary
# # ------------------------------------------------------------------
print("=" * 78)
print(f"SUMMARY: {PASS} passed, {FAIL} failed")
print("=" * 78)
sys.exit(0 if FAIL == 0 else 1)