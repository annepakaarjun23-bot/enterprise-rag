from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def _check_chunker_regex():
    import rag.ingestion.chunkers as ch
    expected_fence = r"^[ ]{0,6}(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$"
    expected_heading = r"^[ ]{0,3}(#{1,6})\s+(.*?)\s*$"
    assert ch._FENCE_RE.pattern == expected_fence, (
        f"chunkers.py fence regex is stale:\n"
        f"  found:  {ch._FENCE_RE.pattern}\n"
        f"  expect: {expected_fence}"
    )
    assert ch._HEADING_RE.pattern == expected_heading, (
        f"chunkers.py heading regex is stale:\n"
        f"  found:  {ch._HEADING_RE.pattern}\n"
        f"  expect: {expected_heading}"
    )



@pytest.fixture
def plain_markdown():
    return """\
# Title

Intro paragraph.

## Section A

Some content in section A.

### Subsection A.1

More content here.
"""


@pytest.fixture
def markdown_with_plain_fence():
    return """\
## Setup

Install the package:

```bash
pip install foo
Then import it.
"""

