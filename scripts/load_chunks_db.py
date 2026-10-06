from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

# force UTF-8 console on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, io.UnsupportedOperation):
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from rag.db.models import (  # noqa: E402
    Chunk,
    DocStatus,
    DocType,
    Document,
    IndexStatus,
)
from rag.db.session import session_scope  # noqa: E402
from rag.settings import settings  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
PARTS_DIR = CHUNKS_DIR / "parts"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_enum(enum_cls, value: str, default):
    try:
        return enum_cls(value)
    except ValueError:
        return default


def load_one_doc(session, part: dict) -> str:
    """Load one doc's parents + leaves. Returns 'loaded' or 'skipped'."""
    lib = part["library"]
    ver = part["version"]
    src_path = part["source_path"]

    # content hash of the cleaned source file (matches documents.content_hash)
    cleaned_rel = part.get("cleaned_path")
    if cleaned_rel is None:
        # derive from source_path: data/processed/<lib>/<ver>/<path>.md
        stem = Path(src_path).with_suffix(".md").as_posix()
        cleaned_rel = f"data/processed/{lib}/{ver}/{stem}"
    cleaned_fp = REPO_ROOT / cleaned_rel
    cleaned_text = cleaned_fp.read_text(encoding="utf-8", errors="replace")
    doc_hash = sha256_text(cleaned_text)
    first_leaf = part["leaves"][0] if part["leaves"] else None
    doc_type_str = (first_leaf["metadata"].get("doc_type") if first_leaf else None) or "other"

    # ---------- 1. upsert documents row ----------
    stmt = (
        pg_insert(Document)
        .values(
            library=lib,
            version=ver,
            source_path=src_path,
            source_url=None,
            title=None,
            doc_type=_coerce_enum(DocType, doc_type_str, DocType.OTHER),
            content_hash=doc_hash,
            status=DocStatus.ACTIVE,
        )
        .on_conflict_do_nothing(
            index_elements=["library", "version", "source_path"]
        )
        .returning(Document.id)
    )
    doc_id = session.execute(stmt).scalar_one_or_none()

    if doc_id is None:
        # already exists -> fetch
        doc_id = session.execute(
            select(Document.id).where(
                Document.library == lib,
                Document.version == ver,
                Document.source_path == src_path,
            )
        ).scalar_one()

    # ---------- 2. skip if hash matches and chunks exist ----------
    existing = session.get(Document, doc_id)
    chunk_count = session.execute(
        select(func.count(Chunk.chunk_id)).where(Chunk.document_id == doc_id)
    ).scalar_one()

    if existing.content_hash == doc_hash and chunk_count > 0:
        return "skipped"

    # ---------- 3. delete existing chunks (hash changed) ----------
    if chunk_count > 0:
        session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
        session.flush()

    # update document's hash + ingested_at
    existing.content_hash = doc_hash
    existing.ingested_at = func.now()

    parents = part["parents"]
    leaves = part["leaves"]

    # ---------- 4. insert parents ----------
    parent_rows = []
    for idx, p in enumerate(parents):
        m = p["metadata"]
        parent_rows.append(
            {
                "chunk_id": p["chunk_id"],
                "document_id": doc_id,
                "strategy": m["strategy"],
                "level": 0,
                "parent_chunk_id": None,
                "is_leaf": False,
                "chunk_index": idx,
                "heading_path": m.get("heading_path", ""),
                "heading_path_list": m.get("heading_path_list", []),
                "section_title": m.get("section_title") or None,
                "text": p["text"],
                "token_count": m.get("token_count", 0),
                "content_hash": sha256_text(p["text"]),
                "has_code": bool(m.get("has_code", False)),
                "code_languages": m.get("code_languages", []),
                "index_status": IndexStatus.NOT_INDEXED,
                "embedding_model": None,
                "indexed_at": None,
                "index_error": None,
            }
        )
    if parent_rows:
        session.execute(pg_insert(Chunk), parent_rows)
        session.flush()

    # ---------- 5. insert leaves ----------
    leaf_rows = []
    for idx, l in enumerate(leaves):
        m = l["metadata"]
        leaf_rows.append(
            {
                "chunk_id": l["chunk_id"],
                "document_id": doc_id,
                "strategy": m["strategy"],
                "level": 1,
                "parent_chunk_id": l["parent_chunk_id"],
                "is_leaf": True,
                "chunk_index": idx,
                "heading_path": m.get("heading_path", ""),
                "heading_path_list": m.get("heading_path_list", []),
                "section_title": m.get("section_title") or None,
                "text": l["text"],
                "token_count": m.get("token_count", 0),
                "content_hash": sha256_text(l["text"]),
                "has_code": bool(m.get("has_code", False)),
                "code_languages": m.get("code_languages", []),
                "index_status": IndexStatus.PENDING,
                "embedding_model": None,
                "indexed_at": None,
                "index_error": None,
            }
        )
    if leaf_rows:
        session.execute(pg_insert(Chunk), leaf_rows)

    return "loaded"


def main() -> None:
    parts = sorted(PARTS_DIR.glob("*.json"))
    if not parts:
        raise SystemExit(f"No part files in {PARTS_DIR}. Run chunk_all.py first.")

    print(f"[load] {len(parts)} part files")

    loaded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for i, part_path in enumerate(parts, 1):
        data = json.loads(part_path.read_text(encoding="utf-8"))
        try:
            with session_scope() as session:
                result = load_one_doc(session, data)
            if result == "loaded":
                loaded += 1
            else:
                skipped += 1
        except Exception as e:  # noqa: BLE001
            failed.append((data.get("source_path", part_path.name), f"{type(e).__name__}: {e}"))
            print(f"  FAIL: {data.get('source_path')} -> {type(e).__name__}: {e}")

        if i % 25 == 0 or i == len(parts):
            print(f"  [{i:3d}/{len(parts)}]  loaded={loaded}  skipped={skipped}  failed={len(failed)}")

    # ---------- final counts from the DB ----------
    with session_scope() as session:
        docs_n = session.execute(select(func.count(Document.id))).scalar_one()
        chunks_n = session.execute(select(func.count(Chunk.chunk_id))).scalar_one()
        parents_n = session.execute(
            select(func.count(Chunk.chunk_id)).where(Chunk.is_leaf.is_(False))
        ).scalar_one()
        leaves_n = session.execute(
            select(func.count(Chunk.chunk_id)).where(Chunk.is_leaf.is_(True))
        ).scalar_one()

    print()
    print("=" * 78)
    print("LOADED INTO POSTGRES")
    print("=" * 78)
    print(f"  docs:    {docs_n}")
    print(f"  chunks:  {chunks_n}   (parents={parents_n}  leaves={leaves_n})")
    print(f"  loaded this run:  {loaded}")
    print(f"  skipped:          {skipped}")
    print(f"  failed:           {len(failed)}")

    if failed:
        print()
        for path, err in failed[:10]:
            print(f"  {path}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()