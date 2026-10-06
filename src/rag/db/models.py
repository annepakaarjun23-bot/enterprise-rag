from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)



def _enum(e: type[enum.Enum], length: int = 24) -> Enum:
    return Enum(
        e,
        native_enum=False,
        length=length,
        values_callable=lambda x: [m.value for m in x],
        validate_strings=True,
    )


class DocType(str, enum.Enum):
    TUTORIAL = "tutorial"
    HOW_TO = "how_to"
    CONCEPT = "concept"
    API_REFERENCE = "api_reference"
    OTHER = "other"


class DocStatus(str, enum.Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded" 
    DELETED = "deleted"      


class IndexStatus(str, enum.Enum):
    PENDING = "pending" 
    INDEXED = "indexed"
    FAILED = "failed"
    STALE = "stale"    
    NOT_INDEXED = "not_indexed"  


class RunStatus(str, enum.Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )



class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("library", "version", "source_path", name="library_version_path"),
        Index("ix_documents_filter", "library", "version", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    library: Mapped[str] = mapped_column(String(50), nullable=False)   
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)    
    source_url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    doc_type: Mapped[DocType] = mapped_column(
        _enum(DocType), nullable=False, default=DocType.OTHER
    )

    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)  
    status: Mapped[DocStatus] = mapped_column(
        _enum(DocStatus), nullable=False, default=DocStatus.ACTIVE
    )
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "strategy", "level", "chunk_index", name="doc_strategy_level_idx"
        ),
        CheckConstraint("level >= 0", name="level_nonneg"),
        CheckConstraint("token_count >= 0", name="tokens_nonneg"),
        Index("ix_chunks_doc_strategy", "document_id", "strategy"),
        Index("ix_chunks_sync", "strategy", "is_leaf", "index_status"),
        Index("ix_chunks_parent", "parent_chunk_id"),
    )

    chunk_id: Mapped[str] = mapped_column(String(40), primary_key=True)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    strategy: Mapped[str] = mapped_column(String(64), nullable=False)

    level: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    parent_chunk_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunks.chunk_id", ondelete="CASCADE")
    )
    is_leaf: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True) 

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    heading_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    heading_path_list: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    section_title: Mapped[str | None] = mapped_column(Text)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    has_code: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    code_languages: Mapped[list[str]] = mapped_column(ARRAY(String(32)), nullable=False, default=list)

    index_status: Mapped[IndexStatus] = mapped_column(
        _enum(IndexStatus), nullable=False, default=IndexStatus.PENDING
    )
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    index_error: Mapped[str | None] = mapped_column(Text)

    document: Mapped["Document"] = relationship(back_populates="chunks")
    parent: Mapped["Chunk | None"] = relationship(
        back_populates="children", remote_side="Chunk.chunk_id"
    )
    children: Mapped[list["Chunk"]] = relationship(back_populates="parent")

