from __future__ import annotations

from rag.ingestion.chunkers import parse_blocks


def test_plain_headings(plain_markdown):
    blocks = parse_blocks(plain_markdown)
    headings = [b for b in blocks if b.kind == "heading"]
    assert [h.level for h in headings] == [1, 2, 3]
    assert headings[0].text == "# Title"
    assert headings[1].text == "## Section A"
    assert headings[2].text == "### Subsection A.1"


def test_simple_code_fence(markdown_with_plain_fence):
    blocks = parse_blocks(markdown_with_plain_fence)
    code = [b for b in blocks if b.kind == "code"]
    assert len(code) == 1
    assert code[0].lang == "bash"
    assert "pip install foo" in code[0].text


def test_display_name_info_string(markdown_with_display_name_fence):
    """'python OpenAI' should parse as lang=python, not fall through to paragraph."""
    blocks = parse_blocks(markdown_with_display_name_fence)
    code = [b for b in blocks if b.kind == "code"]
    assert len(code) == 1, "fence with display name was not recognized as code"
    assert code[0].lang == "python"
    assert "create_agent" in code[0].text


def test_indented_fence_in_list(markdown_with_indented_fence):
    """Fence indented 4 spaces inside a list item should still be recognized."""
    blocks = parse_blocks(markdown_with_indented_fence)
    code = [b for b in blocks if b.kind == "code"]
    assert len(code) == 1, "indented fence not recognized as code"
    assert code[0].lang == "bash"
    assert "API_KEY" in code[0].text


def test_nested_backticks(markdown_with_nested_fence):
    """A ```` block containing ``` should be one block, not two."""
    blocks = parse_blocks(markdown_with_nested_fence)
    code = [b for b in blocks if b.kind == "code"]
    assert len(code) == 1, f"expected 1 code block, got {len(code)}"
    assert "Not a real heading" in code[0].text
    # the inner ``` should NOT close the outer ````
    assert code[0].text.count("````") == 2  # opening + closing
    assert "print(" in code[0].text


def test_7_hashes_is_not_a_heading():
    text = "####### Not a heading\n\nparagraph\n"
    blocks = parse_blocks(text)
    headings = [b for b in blocks if b.kind == "heading"]
    assert len(headings) == 0, "7+ hashes should not be a heading"


def test_heading_with_trailing_whitespace():
    text = "## Title   \n\ncontent\n"
    blocks = parse_blocks(text)
    headings = [b for b in blocks if b.kind == "heading"]
    assert len(headings) == 1
    assert headings[0].text.rstrip() == "## Title"


def test_empty_input():
    assert parse_blocks("") == []


def test_paragraph_collected_until_blank():
    text = "line one\nline two\nline three\n\nnext paragraph\n"
    blocks = parse_blocks(text)
    paras = [b for b in blocks if b.kind == "para"]
    assert len(paras) == 2
    assert "line three" in paras[0].text
    assert paras[1].text == "next paragraph"