from __future__ import annotations

from unittest.mock import MagicMock

from llama_index.core.vector_stores.types import VectorStoreQueryResult

from rag.retrieval.fusion import reciprocal_rank_fusion


def _node(node_id: str):
    m = MagicMock()
    m.node_id = node_id
    return m


def _result(ids: list[str]) -> VectorStoreQueryResult:
    nodes = [_node(i) for i in ids]
    return VectorStoreQueryResult(
        nodes=nodes,
        similarities=[0.5] * len(nodes),
        ids=ids,
    )


def test_rrf_identical_lists_preserve_order():
    dense = _result(["a", "b", "c"])
    sparse = _result(["a", "b", "c"])
    fused = reciprocal_rank_fusion(dense, sparse, top_k=None)
    assert [n.node_id for n in fused.nodes] == ["a", "b", "c"]


def test_rrf_disjoint_lists_interleave():
    dense = _result(["a", "b"])
    sparse = _result(["c", "d"])
    fused = reciprocal_rank_fusion(dense, sparse, top_k=None)
    ids = [n.node_id for n in fused.nodes]
    # all present
    assert set(ids) == {"a", "b", "c", "d"}
    assert ids.index("a") < ids.index("b")
    assert ids.index("c") < ids.index("d")


def test_rrf_doc_in_both_lists_outranks_doc_in_one():
    dense = _result(["x", "a"])
    sparse = _result(["x", "b"])
    fused = reciprocal_rank_fusion(dense, sparse, top_k=None)
    ids = [n.node_id for n in fused.nodes]
    assert ids[0] == "x"


def test_rrf_empty_dense_returns_sparse():
    dense = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    sparse = _result(["a", "b"])
    fused = reciprocal_rank_fusion(dense, sparse, top_k=None)
    assert [n.node_id for n in fused.nodes] == ["a", "b"]


def test_rrf_empty_sparse_returns_dense():
    dense = _result(["a", "b"])
    sparse = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    fused = reciprocal_rank_fusion(dense, sparse, top_k=None)
    assert [n.node_id for n in fused.nodes] == ["a", "b"]


def test_rrf_both_empty():
    empty = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    fused = reciprocal_rank_fusion(empty, empty, top_k=None)
    assert fused.nodes == []


def test_rrf_respects_top_k():
    dense = _result([f"a{i}" for i in range(10)])
    sparse = _result([f"b{i}" for i in range(10)])
    fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
    assert len(fused.nodes) == 5


def test_rrf_no_top_k_returns_all():
    dense = _result([f"a{i}" for i in range(15)])
    sparse = _result([f"b{i}" for i in range(15)])
    fused = reciprocal_rank_fusion(dense, sparse, top_k=None)
    assert len(fused.nodes) == 30