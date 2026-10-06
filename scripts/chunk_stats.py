from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import rag.ingestion.chunkers as ch
from rag.ingestion.chunkers import chunk_document
from rag.ingestion.ids import make_doc_id
from llama_index.core.schema import Document, NodeRelationship

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


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def bucketed(values: list[int], edges: list[int]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    prev: int | None = None
    for e in edges:
        if prev is None:
            out.append((f"<{e}", sum(1 for v in values if v < e)))
        else:
            out.append((f"{prev}-{e}", sum(1 for v in values if prev <= v < e)))
        prev = e
    out.append((f">={edges[-1]}", sum(1 for v in values if v >= edges[-1])))
    return out


def bar(pct: float, width: int = 40) -> str:
    return "#" * int(round(pct / 100 * width))


manifest = [
    json.loads(l)
    for l in (PROCESSED / "cleaned_manifest.jsonl").read_text().splitlines()
    if l.strip()
]

doc_records: list[dict] = []
failures: list[dict] = []

leaf_tokens: list[int] = []
parent_tokens: list[int] = []
leaves_per_parent: list[int] = []

code_langs: Counter[str] = Counter()
per_group: dict[tuple[str, str], dict] = defaultdict(
    lambda: {"docs": 0, "parents": 0, "leaves": 0, "leaf_tokens": []}
)

total_parents = 0
total_leaves = 0
orphan_leaves = 0
bad_fence_leaves = 0
small_leaves = 0
empty_leaves = 0

for r in manifest:
    try:
        src = REPO_ROOT / r["cleaned_path"]
        text = src.read_text(encoding="utf-8", errors="replace")

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

        parent_to_children: Counter[str] = Counter()

        for p in parents:
            parent_tokens.append(p.metadata["token_count"])

        for l in leaves:
            tok = l.metadata["token_count"]
            leaf_tokens.append(tok)
            for lang in l.metadata.get("code_languages", []):
                code_langs[lang] += 1
            if not fence_balanced(l.text):
                bad_fence_leaves += 1
            if tok == 0:
                empty_leaves += 1
            elif tok < 30:
                small_leaves += 1
            if NodeRelationship.PARENT not in l.relationships:
                orphan_leaves += 1
            else:
                parent_to_children[
                    l.relationships[NodeRelationship.PARENT].node_id
                ] += 1

        for count in parent_to_children.values():
            leaves_per_parent.append(count)

        g = per_group[(r["library"], r["version"])]
        g["docs"] += 1
        g["parents"] += len(parents)
        g["leaves"] += len(leaves)
        g["leaf_tokens"].extend(l.metadata["token_count"] for l in leaves)

        total_parents += len(parents)
        total_leaves += len(leaves)

        doc_records.append(
            {
                "path": r["cleaned_path"],
                "library": r["library"],
                "version": r["version"],
                "doc_type": r.get("doc_type", ""),
                "parents": len(parents),
                "leaves": len(leaves),
                "leaf_tokens_min": min(
                    (l.metadata["token_count"] for l in leaves), default=0
                ),
                "leaf_tokens_max": max(
                    (l.metadata["token_count"] for l in leaves), default=0
                ),
                "leaves_with_code": sum(
                    1 for l in leaves if l.metadata["has_code"]
                ),
            }
        )

    except Exception as e:
        failures.append(
            {"path": r["cleaned_path"], "error": f"{type(e).__name__}: {e}"}
        )

ppd = [d["parents"] for d in doc_records]
lpd = [d["leaves"] for d in doc_records]

print("=" * 78)
print(f"CHUNKER DRY RUN: {len(manifest)} files")
print("=" * 78)
print()
print(f"processed: {len(manifest) - len(failures)}")
print(f"failed:    {len(failures)}")
for f in failures[:10]:
    print(f"  {f['path']}: {f['error']}")
print()
print(f"total parents: {total_parents}")
print(f"total leaves:  {total_leaves}")
print(f"avg leaves/parent: {total_leaves / max(total_parents, 1):.2f}")
print()

print("parents per doc:")
print(
    f"  min={min(ppd)}  median={statistics.median(ppd)}  "
    f"p95={percentile(ppd, 95)}  max={max(ppd)}"
)
print()

print("leaves per doc:")
print(
    f"  min={min(lpd)}  median={statistics.median(lpd)}  "
    f"p95={percentile(lpd, 95)}  max={max(lpd)}"
)
print()

print("leaves per parent:")
if leaves_per_parent:
    print(
        f"  min={min(leaves_per_parent)}  median={statistics.median(leaves_per_parent)}  "
        f"p95={percentile(leaves_per_parent, 95)}  max={max(leaves_per_parent)}"
    )
    single = sum(1 for x in leaves_per_parent if x == 1)
    multi = sum(1 for x in leaves_per_parent if x >= 2)
    print(
        f"  parents with 1 leaf:    {single:4d} "
        f"({100 * single / len(leaves_per_parent):5.1f}%)  "
        "-- automerge no-op"
    )
    print(
        f"  parents with 2+ leaves: {multi:4d} "
        f"({100 * multi / len(leaves_per_parent):5.1f}%)  "
        "-- automerge can help"
    )
print()

print("leaf token distribution:")
for label, count in bucketed(leaf_tokens, [50, 100, 200, 300, 500, 1000]):
    pct = 100 * count / max(len(leaf_tokens), 1)
    print(f"  {label:>12s}  {count:5d}  ({pct:5.1f}%)  {bar(pct)}")
print()

print("parent token distribution:")
for label, count in bucketed(parent_tokens, [100, 300, 600, 1200, 2400]):
    pct = 100 * count / max(len(parent_tokens), 1)
    print(f"  {label:>12s}  {count:5d}  ({pct:5.1f}%)  {bar(pct)}")
print()

print("quality flags:")
print(f"  leaves with unbalanced fences: {bad_fence_leaves}")
print(
    f"  leaves with <30 tokens:        {small_leaves} "
    f"({100 * small_leaves / max(len(leaf_tokens), 1):.1f}%)"
)
print(f"  leaves with 0 tokens:          {empty_leaves}")
print(f"  orphan leaves (no parent):     {orphan_leaves}")
print()

print("code languages (leaves containing each):")
for lang, count in code_langs.most_common():
    print(f"  {lang:>15s}  {count}")
print()

print("per (library, version):")
print(
    f"  {'group':<24s} {'docs':>5s} {'parents':>8s} {'leaves':>7s} "
    f"{'avg L/D':>8s} {'med leaf tok':>13s}"
)
for (lib, ver), g in sorted(per_group.items()):
    med = int(statistics.median(g["leaf_tokens"])) if g["leaf_tokens"] else 0
    print(
        f"  {lib + '/' + ver:<24s} {g['docs']:>5d} {g['parents']:>8d} "
        f"{g['leaves']:>7d} {g['leaves'] / max(g['docs'], 1):>8.1f} {med:>13d}"
    )
print()

print("top 5 docs by leaves:")
for d in sorted(doc_records, key=lambda x: -x["leaves"])[:5]:
    print(
        f"  {d['leaves']:4d} leaves / {d['parents']:3d} parents  "
        f"{d['path'].replace(chr(92), '/')}"
    )
print()

print("bottom 5 docs by leaves:")
for d in sorted(doc_records, key=lambda x: x["leaves"])[:5]:
    print(
        f"  {d['leaves']:4d} leaves / {d['parents']:3d} parents  "
        f"{d['path'].replace(chr(92), '/')}"
    )
print()

print("docs where leaves == parents (no automerge benefit):")
flat = [d for d in doc_records if d["leaves"] == d["parents"] and d["parents"] > 1]
print(f"  {len(flat)} of {len(doc_records)} docs")
for d in flat[:10]:
    print(f"    {d['parents']:3d} parents  {d['path'].replace(chr(92), '/')}")
print()

report = {
    "totals": {
        "files": len(manifest),
        "processed": len(manifest) - len(failures),
        "failed": len(failures),
        "parents": total_parents,
        "leaves": total_leaves,
        "avg_leaves_per_parent": total_leaves / max(total_parents, 1),
    },
    "quality": {
        "bad_fence_leaves": bad_fence_leaves,
        "small_leaves_under_30_tok": small_leaves,
        "empty_leaves": empty_leaves,
        "orphan_leaves": orphan_leaves,
    },
    "distributions": {
        "parents_per_doc": {
            "min": min(ppd) if ppd else 0,
            "median": statistics.median(ppd) if ppd else 0,
            "p95": percentile(ppd, 95),
            "max": max(ppd) if ppd else 0,
        },
        "leaves_per_doc": {
            "min": min(lpd) if lpd else 0,
            "median": statistics.median(lpd) if lpd else 0,
            "p95": percentile(lpd, 95),
            "max": max(lpd) if lpd else 0,
        },
        "leaves_per_parent": {
            "min": min(leaves_per_parent) if leaves_per_parent else 0,
            "median": statistics.median(leaves_per_parent) if leaves_per_parent else 0,
            "p95": percentile(leaves_per_parent, 95),
            "max": max(leaves_per_parent) if leaves_per_parent else 0,
        },
        "leaf_token_buckets": dict(bucketed(leaf_tokens, [50, 100, 200, 300, 500, 1000])),
        "parent_token_buckets": dict(bucketed(parent_tokens, [100, 300, 600, 1200, 2400])),
    },
    "code_languages": dict(code_langs),
    "per_group": {
        f"{k[0]}/{k[1]}": {
            "docs": v["docs"],
            "parents": v["parents"],
            "leaves": v["leaves"],
        }
        for k, v in per_group.items()
    },
    "docs": doc_records,
    "failures": failures,
}

out = PROCESSED / "chunk_stats.json"
out.write_text(json.dumps(report, indent=2))
print(f"detailed report -> {out.relative_to(REPO_ROOT)}")