from __future__ import annotations

import pytest

from rag.retrieval.retriever import RetrieverConfig, get_retriever

pytestmark = pytest.mark.integration


def test_dense_retriever_returns_results():
    r = get_retriever(RetrieverConfig(mode="dense", top_k=5, automerge=False))
    results = r.retrieve("how do I add memory to a LangGraph agent?")
    assert len(results) > 0
    assert all(r.node.text for r in results)
    assert all(r.node.metadata.get("library") for r in results)


def test_hybrid_retriever_returns_results():
    r = get_retriever(
        RetrieverConfig(mode="hybrid", top_k=5, sparse_top_k=20, automerge=False)
    )
    results = r.retrieve("how do I add memory to a LangGraph agent?")
    assert len(results) > 0


def test_metadata_filter_restricts_library():
    from llama_index.core.vector_stores.types import (
        FilterCondition,
        MetadataFilter,
        MetadataFilters,
    )
    filters = MetadataFilters(
        filters=[MetadataFilter(key="library", value="langgraph")],
        condition=FilterCondition.AND,
    )
    r = get_retriever(
        RetrieverConfig(mode="dense", top_k=5, automerge=False, filters=filters)
    )
    results = r.retrieve("persistence")
    assert len(results) > 0
    for nws in results:
        assert nws.node.metadata.get("library") == "langgraph"


def test_automerge_resolves_parent():
    from rag.retrieval.index import get_docstore
    r = get_retriever(RetrieverConfig(mode="dense", top_k=15, automerge=False))
    results = r.retrieve("memory")
    assert results, "no results"
    ds = get_docstore()
    first = results[0].node
    from llama_index.core.schema import NodeRelationship
    if NodeRelationship.PARENT in first.relationships:
        parent_id = first.relationships[NodeRelationship.PARENT].node_id
        parent = ds.get_node(parent_id)
        assert parent is not None, f"parent {parent_id} not in docstore"