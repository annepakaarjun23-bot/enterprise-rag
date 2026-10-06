from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DOCS_REPO = "https://github.com/langchain-ai/docs.git"
LG_REPO = "https://github.com/langchain-ai/langgraph.git"
LG_OLD_BRANCH = "0.6"

CURRENT_VERSION = "1.x"
OLD_LG_VERSION = "0.6"

DOC_EXTS = {".md", ".mdx"}

CURRENT_SOURCES = [
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
    ("concepts", "langchain"),
    ("python/migrate", None),        
    ("python/releases", None),
]
CURRENT_EXCLUDE = re.compile(
    r"(^|/)(changelog-(js|py)\.mdx?|academy\.mdx|case-studies\.mdx|get-help\.mdx)$"  # stubs / marketing
    r"|(^|/)frontend/"                                                                
)

OLD_INCLUDE_DIRS = ["concepts", "how-tos", "agents", "troubleshooting"]
OLD_EXCLUDE = re.compile(
    r"langgraph_(cloud|platform|server|control|data|self|standalone|cli|studio|components)"
    r"|deployment_options|(^|/)plans\.md|template_applications|scalability_and_resilience"
    r"|(^|/)sdk\.md|server-mcp|assistants|double_texting|(^|/)auth|(^|/)faq\.md|/http/|/auth/"
)


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def shallow_clone(repo: str, dest: Path, branch: str | None = None, sparse: list[str] | None = None) -> str:
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    if sparse:
        cmd += ["--filter=blob:none", "--sparse"]
    run(cmd + [repo, str(dest)])
    if sparse:
        run(["git", "sparse-checkout", "set", *sparse], cwd=dest)
    return run(["git", "rev-parse", "HEAD"], cwd=dest)


def guess_doc_type(rel: str, source: str) -> str:
    r = rel.lower()
    if "/errors/" in r or r.startswith("troubleshooting/") or "/troubleshooting/" in r:
        return "other"
    if "migrate" in r or "releases" in r or "changelog" in r:
        return "other"
    if source == "old":
        if r.startswith("concepts/"):
            return "concept"
        if r.startswith("how-tos/"):
            return "how_to"
        if r.startswith("agents/"):
            return "tutorial"
        return "other"
    name = Path(r).stem
    if name.startswith("use-") or "how-to" in name:
        return "how_to"
    if name in {"quickstart", "install", "agentic-rag", "sql-agent", "voice-agent", "knowledge-base",
                "thinking-in-langgraph", "deep-agent-from-scratch"}:
        return "tutorial"
    if r.startswith("concepts/") or name in {
        "overview", "philosophy", "persistence", "pregel", "graph-api", "functional-api", "checkpointers",
        "stores", "interrupts", "streaming", "runtime", "context-engineering", "component-architecture",
        "choosing-apis", "application-structure", "short-term-memory", "long-term-memory", "messages",
        "models", "tools", "agents",
    }:
        return "concept"
    return "other"


def lib_from_filename(name: str) -> str:
    return "langgraph" if "langgraph" in name else "langchain"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect_current(root: Path, sha: str):
    base = root / "src" / "oss"
    for sub, lib in CURRENT_SOURCES:
        for p in sorted((base / sub).rglob("*")):
            if not p.is_file() or p.suffix not in DOC_EXTS:
                continue
            rel_to_oss = p.relative_to(base).as_posix()
            if CURRENT_EXCLUDE.search(rel_to_oss):
                continue
            library = lib or lib_from_filename(p.name)
            yield {
                "library": library,
                "version": CURRENT_VERSION,
                "src": p,
                "source_path": rel_to_oss,
                "source_url": f"https://github.com/langchain-ai/docs/blob/{sha}/src/oss/{rel_to_oss}",
                "doc_type": guess_doc_type(rel_to_oss, "current"),
                "repo": "langchain-ai/docs",
                "commit": sha,
            }


def collect_old(root: Path, sha: str):
    base = root / "docs" / "docs"
    for d in OLD_INCLUDE_DIRS:
        for p in sorted((base / d).rglob("*")):
            if not p.is_file() or p.suffix not in DOC_EXTS:
                continue
            rel = p.relative_to(base).as_posix()
            if OLD_EXCLUDE.search(rel):
                continue
            yield {
                "library": "langgraph",
                "version": OLD_LG_VERSION,
                "src": p,
                "source_path": rel,
                "source_url": f"https://github.com/langchain-ai/langgraph/blob/{sha}/docs/docs/{rel}",
                "doc_type": guess_doc_type(rel, "old"),
                "repo": f"langchain-ai/langgraph@{LG_OLD_BRANCH}",
                "commit": sha,
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--min-bytes", type=int, default=400, help="skip stub pages")
    args = ap.parse_args()

    out = Path(args.out)
    tmp = Path(tempfile.mkdtemp(prefix="rag_docs_"))
    try:
        docs_sha = shallow_clone(DOCS_REPO, tmp / "docs")
        lg_sha = shallow_clone(LG_REPO, tmp / "lg_old", branch=LG_OLD_BRANCH, sparse=["docs"])

        candidates = list(collect_current(tmp / "docs", docs_sha)) + list(collect_old(tmp / "lg_old", lg_sha))
        candidates = [c for c in candidates if c["src"].stat().st_size >= args.min_bytes]

        # priority: current docs first, then older ones, until the cap is reached
        selected = candidates[: args.max_docs]

        if out.exists():
            shutil.rmtree(out)
        manifest = []
        for c in selected:
            dest = out / c["library"] / c["version"] / c["source_path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c["src"], dest)
            manifest.append({
                **{k: v for k, v in c.items() if k != "src"},
                "local_path": dest.as_posix(),
                "bytes": dest.stat().st_size,
                "sha256": sha256(dest),
            })

        (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
        print(f"candidates={len(candidates)} selected={len(selected)} -> {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()