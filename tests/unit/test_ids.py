from __future__ import annotations

from rag.ingestion.ids import hex_to_uuid, make_chunk_id, make_doc_id


def test_chunk_id_is_deterministic():
    a = make_chunk_id("lg", "1.x", "p.md", "md_codesafe_v1", 1, "A > B", 0)
    b = make_chunk_id("lg", "1.x", "p.md", "md_codesafe_v1", 1, "A > B", 0)
    assert a == b
    assert len(a) == 32


def test_chunk_id_differs_on_any_input_change():
    base = dict(
        library="lg", version="1.x", source_path="p.md",
        strategy="md_codesafe_v1", level=1, heading_path="A > B", ordinal_in_section=0,
    )
    baseline = make_chunk_id(**base)
    assert make_chunk_id(**{**base, "library": "lc"}) != baseline
    assert make_chunk_id(**{**base, "version": "0.6"}) != baseline
    assert make_chunk_id(**{**base, "source_path": "q.md"}) != baseline
    assert make_chunk_id(**{**base, "strategy": "other"}) != baseline
    assert make_chunk_id(**{**base, "level": 0}) != baseline
    assert make_chunk_id(**{**base, "heading_path": "A > C"}) != baseline
    assert make_chunk_id(**{**base, "ordinal_in_section": 1}) != baseline


def test_leaf_id_differs_by_parent():
    common = dict(
        library="lg", version="1.x", source_path="p.md",
        strategy="md_codesafe_v1", level=1, heading_path="A > B", ordinal_in_section=0,
    )
    leaf1 = make_chunk_id(**common, parent_chunk_id="parent_aaa")
    leaf2 = make_chunk_id(**common, parent_chunk_id="parent_bbb")
    assert leaf1 != leaf2


def test_parent_id_unaffected_by_adding_parent_kwarg():
    no_kwarg = make_chunk_id("lg", "1.x", "p.md", "md_codesafe_v1", 0, "A", 0)
    with_none = make_chunk_id("lg", "1.x", "p.md", "md_codesafe_v1", 0, "A", 0, None)
    assert no_kwarg == with_none


def test_doc_id_is_24_chars():
    d = make_doc_id("lg", "1.x", "p.md")
    assert len(d) == 24


def test_doc_id_stable():
    assert make_doc_id("lg", "1.x", "p.md") == make_doc_id("lg", "1.x", "p.md")


def test_doc_id_differs_on_input():
    assert make_doc_id("lg", "1.x", "p.md") != make_doc_id("lg", "0.6", "p.md")


def test_hex_to_uuid_roundtrip():
    h = "e20a0b71e8a351b54b469e814a9af734"
    u = hex_to_uuid(h)
    assert u == "e20a0b71-e8a3-51b5-4b46-9e814a9af734"
    assert u.replace("-", "") == h


def test_hex_to_uuid_rejects_wrong_length():
    h = "e20a0b71e8a351b54b469e814a9af7"
    u = hex_to_uuid(h)
    assert len(u) != 36  # not a valid UUID string