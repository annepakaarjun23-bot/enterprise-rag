from __future__ import annotations

import pytest

from rag.retrieval.retriever import RetrieverConfig


def test_defaults():
    c = RetrieverConfig()
    assert c.mode == "dense"
    assert c.top_k == 10
    assert c.automerge is True


def test_invalid_mode_rejected():
    """The factory should raise on unknown modes."""
    from rag.retrieval.retriever import get_retriever
    with pytest.raises(ValueError, match="unknown mode"):
        get_retriever(RetrieverConfig(mode="nonsense"))