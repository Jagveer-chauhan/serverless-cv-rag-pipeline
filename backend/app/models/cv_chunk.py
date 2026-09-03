"""CV Chunk database model for text embeddings and RAG vector search."""
import uuid
from datetime import datetime, timezone
from typing import Optional, Any, List
from sqlalchemy import String, Integer, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base
from backend.app.core.config import settings

# Track whether the native pgvector SQLAlchemy type is available.
# When True  → embedding column uses Vector(dim); asyncpg handles the wire type.
# When False → embedding column uses Text; we store the '[f1,f2,...]' string directly.
PGVECTOR_AVAILABLE: bool = False
try:
    from pgvector.sqlalchemy import Vector
    EMBEDDING_TYPE = Vector(settings.EMBEDDING_DIM)
    PGVECTOR_AVAILABLE = True
except Exception:
    # Fallback for SQLite / environments without pgvector installed.
    # Text stores the canonical '[f1,f2,...]' string that PostgreSQL can cast.
    EMBEDDING_TYPE = Text


class CVChunk(Base):
    __tablename__ = "cv_chunks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("cv_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # 384-dimensional vector embedding
    embedding: Mapped[Optional[Any]] = mapped_column(EMBEDDING_TYPE, nullable=True)
    metadata_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    document: Mapped["CVDocument"] = relationship("CVDocument", back_populates="chunks", lazy="selectin")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "section_name": self.section_name,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
