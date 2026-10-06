from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import rag.ingestion.chunkers as ch
from rag.ingestion.chunkers import chunk_document
from rag.ingestion.ids import make_doc_id
from llama_index.core.schema import Document, NodeRelationship

PROCESSED = REPO_ROOT / "data" / "processed"
N = 4
SEED = 42

EXPECTED_FENCE = r"^[ ]{0,6}(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$"
EXPECTED_HEADING = r"^[ ]{0,3}(#{1,6})\s+(.*?)\s*$"

if ch._FENCE_RE.pattern != EXPECTED_FENCE:
    raise SystemExit(
        f"chunkers.py is stale.\n  found:  {ch._FENCE_RE.pattern}\n"
        f"  expect: {EXPECTED_FENCE}\nRe-save chunkers.py and restart your shell."
    )
if ch._HEADING_RE.pattern != EXPECTED_HEADING:
    raise SystemExit("chunkers.py heading regex is stale; re-save the file.")

print("[pre-flight] regexes OK")

_FENCE_LINE = re.compile(r"^[ ]{0,6}(`{3,}|~{3,})(.*)$")

def fence_balanced(text: str) -> bool:
    depth, open_char, open_len = 0, "", 0
    for line in text.splitlines():
        m = _FENCE_LINE.match(line)
        if not m:
            continue
        run, trailing = m.group(1), m.group(2).strip()
        if depth == 0:
            open_char, open_len, depth = run[0], len(run), 1
        elif run[0] == open_char and len(run) >= open_len and trailing == "":
            depth = 0
    return depth == 0

manifest = [
    json.loads(l)
    for l in (PROCESSED / "cleaned_manifest.jsonl").read_text().splitlines()
    if l.strip()
]

groups: dict[tuple[str, str], list[dict]] = {}
for r in manifest:
    groups.setdefault((r["library"], r["version"]), []).append(r)

random.seed(SEED)
picked = [random.choice(v) for v in groups.values()]
if len(picked) < N:
    seen = {id(r) for r in picked}
    pool = [r for v in groups.values() for r in v if id(r) not in seen]
    picked += random.sample(pool, min(N - len(picked), len(pool)))

# --- 4. run and assert ---
_COLON = re.compile(r"(?m)^:::[a-zA-Z]")
_IMPORT = re.compile(r"(?m)^\s*import\s+.*from\s+")
_JSX = re.compile(r"<[A-Z][A-Za-z0-9_]*[\s/>]")

failures = 0
for r in picked:
    src = REPO_ROOT / r["cleaned_path"]
    text = src.read_text(encoding="utf-8", errors="replace")

    noise = {
        "colon": len(_COLON.findall(text)),
        "import": len(_IMPORT.findall(text)),
        "jsx": len(_JSX.findall(text)),
    }

    doc = Document(
        text=text,
        metadata={
            "library": r["library"],
            "version": r["version"],
            "source_path": r["source_path"],
        },
    )
    doc.doc_id = make_doc_id(r["library"], r["version"], r["source_path"])

    parents, leaves = chunk_document(doc)

    bad = [l.id_ for l in leaves if not fence_balanced(l.text)]
    orphans = [l.id_ for l in leaves if NodeRelationship.PARENT not in l.relationships]

    print("=" * 78)
    print(f"{r['library']}/{r['version']}/{r['source_path']}  [{r['format']}]")
    print(f"  noise: {noise}")
    print(f"  parents={len(parents)}  leaves={len(leaves)}")
    if leaves:
        toks = [l.metadata["token_count"] for l in leaves]
        print(f"  leaf tokens:    min={min(toks)}  max={max(toks)}")
        print(f"  leaves w/ code: {sum(1 for l in leaves if l.metadata['has_code'])}")
        langs = sorted({lang for l in leaves for lang in l.metadata["code_languages"]})
        print(f"  code languages: {langs}")
    if bad:
        print(f"  FAIL fences: {bad[:3]}")
        failures += 1
    if orphans:
        print(f"  FAIL parent missing on: {orphans[:3]}")
        failures += 1

print()
if failures:
    raise SystemExit(f"{failures} file(s) failed. See above.")
print("all checks passed")