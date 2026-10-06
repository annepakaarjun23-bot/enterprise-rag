from __future__ import annotations

import io
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, io.UnsupportedOperation):
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import rag.ingestion.chunkers as ch
from rag.ingestion.chunkers import chunk_document
from rag.ingestion.ids import make_doc_id
from llama_index.core.schema import (
    Document,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.storage.docstore import SimpleDocumentStore

EXPECTED_FENCE = r"^[ ]{0,6}(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$"
EXPECTED_HEADING = r"^[ ]{0,3}(#{1,6})\s+(.*?)\s*$"

if ch._FENCE_RE.pattern != EXPECTED_FENCE:
    raise SystemExit(
        f"chunkers.py stale:\n  found:  {ch._FENCE_RE.pattern}\n"
        f"  expect: {EXPECTED_FENCE}"
    )
if ch._HEADING_RE.pattern != EXPECTED_HEADING:
    raise SystemExit(
        f"chunkers.py stale heading regex:\n  found:  {ch._HEADING_RE.pattern}\n"
        f"  expect: {EXPECTED_HEADING}"
    )
print("[pre-flight] regexes OK")

PROCESSED = REPO_ROOT / "data" / "processed"
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
PARTS_DIR = CHUNKS_DIR / "parts"
STRATEGY = "md_codesafe_v1"
SHORT_LEAF_THRESHOLD = 20

LANG_MAP = {
    "py": "python",
    "python3": "python",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "txt": "plaintext",
    "text": "plaintext",
    "con": "console",
}


def normalise_lang(lang: str) -> str:
    return LANG_MAP.get(lang.lower(), lang.lower())


CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
PARTS_DIR.mkdir(parents=True, exist_ok=True)
(CHUNKS_DIR / "docstore").mkdir(parents=True, exist_ok=True)


manifest = [
    json.loads(l)
    for l in (PROCESSED / "cleaned_manifest.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if l.strip()
]
print(f"[load] {len(manifest)} files in manifest")


def doc_id_for(row: dict) -> str:
    return make_doc_id(row["library"], row["version"], row["source_path"])


def part_path_for(row: dict) -> Path:
    return PARTS_DIR / f"{doc_id_for(row)}.json"


def chunk_one(row: dict) -> dict:
    """Chunk one cleaned file. Returns a dict with parents and leaves as plain rows."""
    src = REPO_ROOT / row["cleaned_path"]
    text = src.read_text(encoding="utf-8", errors="replace")

    doc_id = doc_id_for(row)
    doc = Document(
        text=text,
        metadata={
            "library": row["library"],
            "version": row["version"],
            "source_path": row["source_path"],
            "doc_type": row.get("doc_type", "other"),
            "source_url": row.get("source_url", ""),
        },
    )
    doc.doc_id = doc_id

    parents, leaves = chunk_document(doc, strategy=STRATEGY)

    for node in parents + leaves:
        raw = node.metadata.get("code_languages", []) or []
        node.metadata["code_languages"] = sorted({normalise_lang(l) for l in raw})
    for leaf in leaves:
        leaf.metadata["short_leaf"] = (
            leaf.metadata["token_count"] < SHORT_LEAF_THRESHOLD
        )

    def to_row(node):
        rel = node.relationships
        src_id = (
            rel[NodeRelationship.SOURCE].node_id
            if NodeRelationship.SOURCE in rel
            else None
        )
        return {
            "chunk_id": node.id_,
            "doc_id": src_id,
            "text": node.text,
            "metadata": node.metadata,
            "parent_chunk_id": (
                rel[NodeRelationship.PARENT].node_id
                if NodeRelationship.PARENT in rel
                else None
            ),
            "prev_chunk_id": (
                rel[NodeRelationship.PREVIOUS].node_id
                if NodeRelationship.PREVIOUS in rel
                else None
            ),
            "next_chunk_id": (
                rel[NodeRelationship.NEXT].node_id
                if NodeRelationship.NEXT in rel
                else None
            ),
        }

    return {
        "doc_id": doc_id,
        "library": row["library"],
        "version": row["version"],
        "source_path": row["source_path"],
        "parents": [to_row(p) for p in parents],
        "leaves": [to_row(l) for l in leaves],
    }


skipped = 0
processed = 0
failures: list[dict] = []

for i, row in enumerate(manifest, 1):
    part = part_path_for(row)
    if part.exists():
        skipped += 1
        if i % 25 == 0 or i == len(manifest):
            print(
                f"  [{i:3d}/{len(manifest)}]  skipped={skipped}  processed={processed}"
            )
        continue

    try:
        result = chunk_one(row)
        tmp = part.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
        tmp.rename(part)
        processed += 1
    except Exception as e:  # noqa: BLE001
        failures.append(
            {
                "path": row["cleaned_path"],
                "error": f"{type(e).__name__}: {e}",
            }
        )
        print(f"  FAIL: {row['cleaned_path']}  ->  {type(e).__name__}: {e}")

    if i % 25 == 0 or i == len(manifest):
        print(
            f"  [{i:3d}/{len(manifest)}]  skipped={skipped}  processed={processed}"
        )


part_files = sorted(PARTS_DIR.glob("*.json"))
print(f"\n[combine] reading {len(part_files)} part files")

all_parents: list[dict] = []
all_leaves: list[dict] = []
lang_counter: Counter[str] = Counter()
short_leaf_count = 0
per_doc: list[dict] = []

for part in part_files:
    data = json.loads(part.read_text(encoding="utf-8"))
    all_parents.extend(data["parents"])
    all_leaves.extend(data["leaves"])
    for leaf in data["leaves"]:
        for lang in leaf["metadata"].get("code_languages", []):
            lang_counter[lang] += 1
        if leaf["metadata"].get("short_leaf"):
            short_leaf_count += 1
    per_doc.append(
        {
            "doc_id": data["doc_id"],
            "library": data["library"],
            "version": data["version"],
            "source_path": data["source_path"],
            "parents": len(data["parents"]),
            "leaves": len(data["leaves"]),
        }
    )

# write combined JSONL
parents_path = CHUNKS_DIR / "parents.jsonl"
leaves_path = CHUNKS_DIR / "leaves.jsonl"
with parents_path.open("w", encoding="utf-8") as f:
    for row in all_parents:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with leaves_path.open("w", encoding="utf-8") as f:
    for row in all_leaves:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

docstore_dir = CHUNKS_DIR / "docstore"
docstore_dir.mkdir(parents=True, exist_ok=True)
docstore_file = docstore_dir / "docstore.json"

from rag.ingestion.ids import hex_to_uuid 

docstore = SimpleDocumentStore()
for row in all_parents:
    node = TextNode(
        id_=hex_to_uuid(row["chunk_id"]),   
        text=row["text"],
        metadata={
            **row["metadata"],
            "chunk_id": row["chunk_id"],        
        },
    )
    if row["doc_id"]:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=row["doc_id"]
        )
    docstore.add_documents([node])

docstore.persist(str(docstore_file))

summary = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy": STRATEGY,
    "short_leaf_threshold": SHORT_LEAF_THRESHOLD,
    "files_total": len(manifest),
    "files_processed_this_run": processed,
    "files_skipped_this_run": skipped,
    "files_failed": len(failures),
    "parents": len(all_parents),
    "leaves": len(all_leaves),
    "short_leaves": short_leaf_count,
    "code_languages": dict(lang_counter.most_common()),
    "failures": failures,
    "per_doc": per_doc,
    "outputs": {
        "parents": str(parents_path.relative_to(REPO_ROOT)),
        "leaves": str(leaves_path.relative_to(REPO_ROOT)),
        "docstore": str(docstore_dir.relative_to(REPO_ROOT)),
        "parts": str(PARTS_DIR.relative_to(REPO_ROOT)),
    },
}
(CHUNKS_DIR / "chunk_run.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print()
print("=" * 78)
print("CHUNKED")
print("=" * 78)
print(f"  files in manifest:      {len(manifest)}")
print(f"  processed this run:     {processed}")
print(f"  skipped (already done): {skipped}")
print(f"  failed this run:        {len(failures)}")
print()
print(f"  parents:                {len(all_parents)}")
print(f"  leaves:                 {len(all_leaves)}")
print(f"  short leaves (<{SHORT_LEAF_THRESHOLD}):    {short_leaf_count}")
print()
print("  code languages (canonical):")
for lang, count in lang_counter.most_common():
    print(f"    {lang:>15s}  {count}")
print()
print("  outputs:")
for label, path in summary["outputs"].items():
    print(f"    {label:>10s}: {path}")

if failures:
    print()
    print("  failures:")
    for f in failures[:10]:
        print(f"    {f['path']}: {f['error']}")
    print()
    print("  rerun `python scripts/chunk_all.py` to retry only the failed files.")
    raise SystemExit(1)