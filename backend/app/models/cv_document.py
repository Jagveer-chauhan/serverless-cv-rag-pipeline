"""CV Document database model."""
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Any
from sqlalchemy import String, Integer, Float, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base


class CVDocument(Base):
    __tablename__ = "cv_documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), default="application/pdf")
    
    # Status progression: queued -> extracting -> indexing -> rag_ready (or failed)
    status: Mapped[str] = mapped_column(String(50), default="queued", index=True)
    
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    total_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    chunks: Mapped[List["CVChunk"]] = relationship(
        "CVChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="CVChunk.chunk_index",
        lazy="selectin"
    )
    traces: Mapped[List["CVProcessingTrace"]] = relationship(
        "CVProcessingTrace",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="CVProcessingTrace.start_time",
        lazy="selectin"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "status": self.status,
            "error_message": self.error_message,
            "total_duration_ms": self.total_duration_ms,
            "parsed_json": self.parsed_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
