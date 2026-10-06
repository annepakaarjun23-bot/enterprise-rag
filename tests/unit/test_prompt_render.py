from __future__ import annotations

from rag.generation.answerer import render_prompt


def test_basic_substitution():
    out = render_prompt(
        query="What is X?",
        context_chunks=["X is a thing.", "X is useful."],
    )
    assert "What is X?" in out
    assert "X is a thing." in out
    assert "X is useful." in out


def test_braces_in_context_do_not_crash():
    ctx = 'config = {"model": "gpt-4", "temperature": 0.2}'
    out = render_prompt(query="how?", context_chunks=[ctx])
    assert '{"model": "gpt-4", "temperature": 0.2}' in out


def test_json_in_context_survives():
    ctx = '{"users": [{"id": 1, "name": "alice"}]}'
    out = render_prompt(query="who?", context_chunks=[ctx])
    assert '{"users":' in out


def test_braces_around_placeholder_name():
    ctx = 'print("{context}")'
    out = render_prompt(query="q", context_chunks=[ctx])
    assert 'print("{context}")' in out


def test_empty_context():
    out = render_prompt(query="anything?", context_chunks=[])
    assert "no context retrieved" in out
    assert "anything?" in out


def test_multiple_chunks_numbered():
    out = render_prompt(
        query="q",
        context_chunks=["first", "second", "third"],
    )
    assert "[S1]" in out
    assert "[S2]" in out
    assert "[S3]" in out

def test_chunk_labels_match_citation_contract():
    out = render_prompt(
        query="anything",
        context_chunks=["alpha", "beta"],
    )
    assert "[S1]" in out
    assert "[S2]" in out
    assert out.index("[S1]") < out.index("alpha")
    assert out.index("[S2]") < out.index("beta")