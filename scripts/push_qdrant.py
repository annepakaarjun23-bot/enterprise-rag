from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, io.UnsupportedOperation):
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from llama_index.core.schema import TextNode  # noqa: E402
from sqlalchemy import func, select, text  # noqa: E402

from rag.db.models import Chunk, Document, IndexStatus  # noqa: E402
from rag.db.session import session_scope  # noqa: E402
from rag.indexing.qdrant_store import (  # noqa: E402
    collection_summary,
    get_vector_store,
)

from llama_index.core.schema import (
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from rag.ingestion.ids import hex_to_uuid

from rag.settings import settings  # noqa: E402

CHUNK_PARTS = REPO_ROOT / "data" / "chunks" / "parts"
EMB_PARTS = REPO_ROOT / "data" / "embeddings" / "parts"


def hex_to_uuid(h: str) -> str:
    """chunk_id is 32 hex chars -> format as UUID for Qdrant point ID.
    Original chunk_id stays in the payload, so the conversion is lossless."""
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def build_node(leaf: dict, chunk_part: dict, vector: list[float]) -> TextNode:
    m = leaf["metadata"]
    node = TextNode(
        id_=hex_to_uuid(leaf["chunk_id"]),
        text=leaf["text"],
        metadata={
            "chunk_id": leaf["chunk_id"],
            "doc_id": chunk_part["doc_id"],
            "parent_chunk_id": leaf["parent_chunk_id"] or "",
            "library": chunk_part["library"],
            "version": chunk_part["version"],
            "doc_type": m.get("doc_type", ""),
            "source_path": chunk_part["source_path"],
            "source_url": m.get("source_url", ""),
            "heading_path": m.get("heading_path", ""),
            "section_title": m.get("section_title") or "",
            "has_code": bool(m.get("has_code", False)),
            "code_languages": list(m.get("code_languages", [])),
            "token_count": int(m.get("token_count", 0)),
            "short_leaf": bool(m.get("short_leaf", False)),
        },
    )
    node.embedding = vector

    if leaf.get("parent_chunk_id"):
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
            node_id=hex_to_uuid(leaf["parent_chunk_id"])
        )
    return node

def load_doc_id_lookup() -> dict[tuple[str, str, str], int]:
    with session_scope() as session:
        rows = session.execute(
            select(Document.id, Document.library, Document.version, Document.source_path)
        ).all()
    return {(r.library, r.version, r.source_path): r.id for r in rows}


def load_fully_indexed_set() -> set[int]:
    """Document integer IDs whose leaves are all indexed."""
    sql = text(
        """
        SELECT document_id
        FROM chunks
        WHERE is_leaf = true
        GROUP BY document_id
        HAVING bool_and(index_status = 'indexed')
        """
    )
    with session_scope() as session:
        rows = session.execute(sql).all()
    return {r[0] for r in rows}


def mark_indexed(chunk_ids: list[str]) -> int:
    if not chunk_ids:
        return 0
    with session_scope() as session:
        result = session.execute(
            Chunk.__table__.update()
            .where(Chunk.chunk_id.in_(chunk_ids))
            .values(
                index_status=IndexStatus.INDEXED,
                indexed_at=func.now(),
                embedding_model=settings.embedding_model,
            )
        )
        return result.rowcount or 0


def main() -> None:
    if not settings.embedding_api_key:
        raise SystemExit("EMBEDDING_API_KEY is empty; not needed for push, but "
                         "check .env to avoid mixups.")

    emb_parts = sorted(EMB_PARTS.glob("*.npz"))
    if not emb_parts:
        raise SystemExit(f"No embedding parts in {EMB_PARTS}. Run embed_all.py first.")

    print(f"[push] {len(emb_parts)} docs to consider")
    print(f"  collection: {settings.qdrant_collection_leaves}")
    print(f"  qdrant:     {settings.qdrant_url}")
    print(f"  model tag:  {settings.embedding_model}")
    print()

    before = collection_summary()
    print(f"[qdrant] pre-existing: {before}")
    print()

    doc_id_lookup = load_doc_id_lookup()
    fully_indexed = load_fully_indexed_set()
    print(f"[db] {len(doc_id_lookup)} documents, {len(fully_indexed)} fully indexed")
    print()

    vector_store = get_vector_store()

    pushed_docs = 0
    skipped_docs = 0
    pushed_leaves = 0
    failed: list[tuple[str, str]] = []

    for i, emb_path in enumerate(emb_parts, 1):
        doc_hash = emb_path.stem
        chunk_path = CHUNK_PARTS / f"{doc_hash}.json"
        if not chunk_path.exists():
            failed.append((doc_hash, "chunk part missing"))
            continue

        chunk_part = json.loads(chunk_path.read_text(encoding="utf-8"))
        key = (chunk_part["library"], chunk_part["version"], chunk_part["source_path"])
        doc_int = doc_id_lookup.get(key)
        if doc_int is None:
            failed.append((chunk_part["source_path"], "document not in Postgres"))
            continue

        if doc_int in fully_indexed:
            skipped_docs += 1
            if i % 25 == 0 or i == len(emb_parts):
                print(f"  [{i:3d}/{len(emb_parts)}]  pushed={pushed_docs}  "
                      f"skipped={skipped_docs}  failed={len(failed)}")
            continue

        print(f"  [{i:3d}/{len(emb_parts)}]  {chunk_part['source_path']}")
        try:
            d = np.load(emb_path, allow_pickle=False)
            vectors = d["vectors"]
            chunk_ids = d["chunk_ids"]

            leaf_lookup = {l["chunk_id"]: l for l in chunk_part["leaves"]}
            nodes: list[TextNode] = []
            for j, cid in enumerate(chunk_ids):
                leaf = leaf_lookup.get(str(cid))
                if leaf is None:
                    continue
                nodes.append(build_node(leaf, chunk_part, vectors[j].tolist()))

            if nodes:
                vector_store.add(nodes)
                mark_indexed([n.id_.replace("-", "") for n in nodes])
                pushed_leaves += len(nodes)

            pushed_docs += 1
        except Exception as e:  # noqa: BLE001
            failed.append((chunk_part["source_path"], f"{type(e).__name__}: {e}"))
            print(f"    FAIL: {type(e).__name__}: {e}")

        if i % 25 == 0 or i == len(emb_parts):
            print(f"  [{i:3d}/{len(emb_parts)}]  pushed={pushed_docs}  "
                  f"skipped={skipped_docs}  failed={len(failed)}")

    after = collection_summary()

    print()
    print("=" * 78)
    print("PUSHED TO QDRANT")
    print("=" * 78)
    print(f"  docs pushed:          {pushed_docs}")
    print(f"  docs skipped:         {skipped_docs}")
    print(f"  docs failed:          {len(failed)}")
    print(f"  leaves pushed:        {pushed_leaves}")
    print(f"  collection points:    {after.get('points_count')}")
    print(f"  dense vector names:   {after.get('dense_vector_names')}")
    print(f"  sparse vector names:  {after.get('sparse_vector_names')}")

    if failed:
        print()
        for p, e in failed[:10]:
            print(f"  {p}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()