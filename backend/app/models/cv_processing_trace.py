"""CV Processing Trace model for millisecond-level pipeline stage observability."""
import uuid
from datetime import datetime, timezone
from typing import Optional, Any
from sqlalchemy import String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base


class CVProcessingTrace(Base):
    __tablename__ = "cv_processing_traces"

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
    stage: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True
    )  # text_extraction, chunking, llm_extraction, validation, merge, embedding, vector_upsert, rag_verification, total
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="success")  # success, failed
    metadata_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    document: Mapped["CVDocument"] = relationship("CVDocument", back_populates="traces", lazy="selectin")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "stage": self.stage,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata_json,
        }
