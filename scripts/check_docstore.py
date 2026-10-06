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

from rag.retrieval.index import get_docstore


def main() -> None:
    ds = get_docstore()
    print(f"docstore path: {REPO_ROOT / 'data' / 'chunks' / 'docstore' / 'docstore.json'}")
    print(f"nodes:         {len(ds.docs)}")
    print()

    # show first 3 parents with their text
    for i, (node_id, node) in enumerate(ds.docs.items()):
        if i >= 3:
            break
        m = node.metadata
        print("=" * 78)
        print(f"id:            {node_id}")
        print(f"heading_path:  {m.get('heading_path')}")
        print(f"section_title: {m.get('section_title')}")
        print(f"tokens:        {m.get('token_count')}")
        print(f"is_leaf:       {m.get('is_leaf')}  level={m.get('level')}")
        print(f"has_code:      {m.get('has_code')}  langs={m.get('code_languages')}")
        print(f"text length:   {len(node.text)} chars")
        print("-" * 78)
        print(node.text[:400])
        print()


if __name__ == "__main__":
    main()