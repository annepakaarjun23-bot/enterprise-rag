from __future__ import annotations

import pytest
from sqlalchemy import func, select

from rag.db.models import Chunk, Document
from rag.db.session import session_scope

pytestmark = pytest.mark.integration


def test_documents_table_populated():
    with session_scope() as s:
        n = s.execute(select(func.count(Document.id))).scalar_one()
    assert n > 0, "documents table is empty -- run scripts/load_chunks_db.py"


def test_parents_and_leaves_counts():
    with session_scope() as s:
        parents = s.execute(
            select(func.count(Chunk.chunk_id)).where(Chunk.is_leaf.is_(False))
        ).scalar_one()
        leaves = s.execute(
            select(func.count(Chunk.chunk_id)).where(Chunk.is_leaf.is_(True))
        ).scalar_one()
    assert parents > 0
    assert leaves > 0
    assert 1000 < parents < 3000
    assert 2000 < leaves < 4000


def test_every_leaf_has_a_parent():
    with session_scope() as s:
        orphans = s.execute(
            select(func.count(Chunk.chunk_id)).where(
                Chunk.is_leaf.is_(True),
                Chunk.parent_chunk_id.is_(None),
            )
        ).scalar_one()
    assert orphans == 0


def test_index_status_values():
    with session_scope() as s:
        leaf_statuses = {
            row.index_status
            for row in s.execute(
                select(Chunk.index_status).where(Chunk.is_leaf.is_(True))
            ).all()
        }
    # allowed leaf states
    assert leaf_statuses.issubset({"pending", "indexed", "failed", "stale"})