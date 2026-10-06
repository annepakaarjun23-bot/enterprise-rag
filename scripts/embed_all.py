from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, io.UnsupportedOperation):
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.ingestion.embedder import DIMENSIONS, MODEL, embed_batch  # noqa: E402
from rag.settings import settings  # noqa: E402

CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
PARTS_DIR = CHUNKS_DIR / "parts"
EMB_DIR = REPO_ROOT / "data" / "embeddings"
EMB_PARTS = EMB_DIR / "parts"
EMB_PARTS.mkdir(parents=True, exist_ok=True)

if not settings.embedding_api_key:
    raise SystemExit(
        "EMBEDDING_API_KEY is empty. Set it in .env to your sk-air-v1-... key."
    )


def embed_one_doc(part: dict) -> dict:
    """Embed every leaf in a doc. Returns arrays for the .npz part."""
    leaves = part["leaves"]
    if not leaves:
        return {
            "chunk_ids": np.array([], dtype="U40"),
            "vectors": np.zeros((0, DIMENSIONS), dtype=np.float32),
            "token_count": 0,
        }

    chunk_ids = [l["chunk_id"] for l in leaves]
    texts = [l["text"] for l in leaves]
    vectors = embed_batch(texts)
    token_count = sum(l["metadata"]["token_count"] for l in leaves)

    return {
        "chunk_ids": np.array(chunk_ids, dtype="U40"),
        "vectors": np.asarray(vectors, dtype=np.float32),
        "token_count": token_count,
    }


def main() -> None:
    part_files = sorted(PARTS_DIR.glob("*.json"))
    if not part_files:
        raise SystemExit(f"No chunk parts in {PARTS_DIR}. Run chunk_all.py first.")

    print(f"[embed] {len(part_files)} docs")
    print(f"  model: {MODEL}")
    print(f"  base:  {settings.embedding_base_url}")
    print(f"  dims:  {DIMENSIONS}")
    print()

    processed = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    total_tokens = 0

    for i, part_path in enumerate(part_files, 1):
        doc_id = part_path.stem
        out_path = EMB_PARTS / f"{doc_id}.npz"

        if out_path.exists():
            skipped += 1
            if i % 25 == 0 or i == len(part_files):
                print(f"  [{i:3d}/{len(part_files)}]  skipped={skipped}  processed={processed}")
            continue

        data = json.loads(part_path.read_text(encoding="utf-8"))
        print(f"  [{i:3d}/{len(part_files)}]  {data['source_path']}")
        try:
            result = embed_one_doc(data)
            tmp = out_path.with_suffix(".tmp.npz")
            np.savez(
                tmp,
                chunk_ids=result["chunk_ids"],
                vectors=result["vectors"],
                token_count=np.array([result["token_count"]]),
            )
            tmp.rename(out_path)
            processed += 1
            total_tokens += result["token_count"]
        except Exception as e:  # noqa: BLE001
            failed.append((data.get("source_path", doc_id), f"{type(e).__name__}: {e}"))
            print(f"    FAIL: {type(e).__name__}: {e}")

        if i % 25 == 0 or i == len(part_files):
            print(f"  [{i:3d}/{len(part_files)}]  skipped={skipped}  processed={processed}")

    # combine
    print(f"\n[combine] reading {len(list(EMB_PARTS.glob('*.npz')))} part files")
    ids_list: list[np.ndarray] = []
    vec_list: list[np.ndarray] = []
    for p in sorted(EMB_PARTS.glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        ids_list.append(d["chunk_ids"])
        vec_list.append(d["vectors"])

    combined_ids = np.concatenate(ids_list, axis=0) if ids_list else np.array([], dtype="U40")
    combined_vecs = np.concatenate(vec_list, axis=0) if vec_list else np.zeros((0, DIMENSIONS), dtype=np.float32)

    combined_path = EMB_DIR / "leaves_vectors.npz"
    np.savez(combined_path, chunk_ids=combined_ids, vectors=combined_vecs)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "base_url": settings.embedding_base_url,
        "dimensions": DIMENSIONS,
        "docs_total": len(part_files),
        "docs_processed_this_run": processed,
        "docs_skipped_this_run": skipped,
        "docs_failed": len(failed),
        "leaves_embedded_total": int(combined_ids.shape[0]),
        "tokens_embedded_total": total_tokens,
        "failures": [{"path": p, "error": e} for p, e in failed],
        "outputs": {
            "parts": str(EMB_PARTS.relative_to(REPO_ROOT)),
            "combined": str(combined_path.relative_to(REPO_ROOT)),
        },
    }
    (EMB_DIR / "embed_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("EMBEDDED")
    print("=" * 78)
    print(f"  model:                   {MODEL}")
    print(f"  docs processed:          {processed}")
    print(f"  docs skipped:            {skipped}")
    print(f"  docs failed:             {len(failed)}")
    print(f"  leaves embedded (total): {combined_ids.shape[0]}")
    print(f"  tokens this run:         {total_tokens:,}")
    print(f"  vector shape:            {combined_vecs.shape}")
    print()
    print(f"  combined: {combined_path.relative_to(REPO_ROOT)}")

    if failed:
        print()
        for p, e in failed[:10]:
            print(f"  {p}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()