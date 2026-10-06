from __future__ import annotations

import io
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, io.UnsupportedOperation):
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "models" / "hf"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main() -> None:
    print(f"[download] cache: {CACHE_DIR}")
    print(f"[download] model: {MODEL}")

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(MODEL, cache_folder=str(CACHE_DIR))
    print(f"[download] loaded OK. Model dir: {CACHE_DIR / 'hub'}")

    # smoke test
    scores = model.predict([("how do I add memory", "use a checkpointer")])
    print(f"[download] smoke test score: {float(scores[0]):.4f}")


if __name__ == "__main__":
    main()