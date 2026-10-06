from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "evals" / "datasets" / "golden_retriever.jsonl"

if not SRC.exists():
    raise SystemExit(f"not found: {SRC}")

backup = SRC.with_suffix(".jsonl.bak")
shutil.copy2(SRC, backup)
print(f"backup: {backup.name}")

raw = SRC.read_text(encoding="utf-8")
lines = raw.splitlines()

clean = [ln for ln in lines if not ln.lstrip().startswith("//")]

text = "\n".join(clean)
text = re.sub(r",(\s*[\]}])", r"\1", text)

# 3. parse
try:
    data = json.loads(text)
except json.JSONDecodeError as e:
    print(f"still broken at line {e.lineno}, col {e.colno}: {e.msg}")
    ctx_line = text.splitlines()[e.lineno - 1] if e.lineno - 1 < len(text.splitlines()) else ""
    print(f"  context: {ctx_line!r}")
    raise SystemExit(
        "manual fix needed — the line above shows where. Then rerun this script."
    )

if not isinstance(data, list):
    raise SystemExit(f"expected a JSON array, got {type(data).__name__}")

# 4. write as JSONL
with SRC.open("w", encoding="utf-8") as f:
    for row in data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"wrote {len(data)} rows as JSONL to {SRC.name}")
print("columns now: one object per line. Delete a line to remove a row.")