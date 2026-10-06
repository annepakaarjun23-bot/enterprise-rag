"""Tests for chunkers.build_leaves."""
from __future__ import annotations

from rag.ingestion.chunkers import (
    Block,
    Section,
    build_leaves,
    parse_blocks,
    build_parents,
)


def _section_from_markdown(text: str, max_parent_tokens: int = 100000) -> Section:
    blocks = parse_blocks(text)
    sections = build_parents(blocks, min_tokens=0, max_tokens=max_parent_tokens)
    assert sections, "no sections produced"
    return sections[0]


def test_code_block_stays_intact():
    text = """\
# Root

## Section

Some prose.

```python
def foo():
    return 42
    
More prose.
"""