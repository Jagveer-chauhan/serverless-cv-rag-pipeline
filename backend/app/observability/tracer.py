"""PipelineTracer context manager for high-resolution millisecond-level stage observability."""
import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.models.cv_processing_trace import CVProcessingTrace
from backend.app.models.cv_document import CVDocument

logger = logging.getLogger("cv_rag_pipeline.tracer")

# Required pipeline stages for strict SLA benchmarking
REQUIRED_STAGES = [
    "text_extraction",
    "chunking",
    "llm_extraction",
    "validation",
    "merge",
    "embedding",
    "vector_upsert",
    "rag_verification",
]


@dataclass
class StageTrace:
    stage: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    status: str = "success"  # "success" or "failed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


class PipelineTracer:
    """Tracks and measures execution duration across all pipeline stages."""

    def __init__(self, document_id: Optional[str] = None):
        self.document_id = document_id
        self.traces: Dict[str, StageTrace] = {}
        self.start_perf_time: float = time.perf_counter()
        self.start_wall_time: datetime = datetime.now(timezone.utc)
        self.total_duration_ms: float = 0.0

    @asynccontextmanager
    async def trace_stage(self, stage: str, metadata: Optional[Dict[str, Any]] = None):
        """Asynchronous context manager to time a stage execution."""
        t_start = time.perf_counter()
        wall_start = datetime.now(timezone.utc)
        stage_trace = StageTrace(
            stage=stage,
            start_time=wall_start,
            metadata=metadata or {},
        )
        try:
            yield stage_trace
            stage_trace.status = "success"
        except Exception as e:
            stage_trace.status = "failed"
            stage_trace.error_message = str(e)
            logger.error(f"Stage '{stage}' failed: {e}", exc_info=True)
            raise
        finally:
            t_end = time.perf_counter()
            stage_trace.end_time = datetime.now(timezone.utc)
            stage_trace.duration_ms = round((t_end - t_start) * 1000, 3)
            self.traces[stage] = stage_trace
            logger.info(
                f"⏱️ [PipelineTracer] Stage '{stage}' completed in {stage_trace.duration_ms:.2f}ms "
                f"[{stage_trace.status}]"
            )

    @contextmanager
    def trace_stage_sync(self, stage: str, metadata: Optional[Dict[str, Any]] = None):
        """Synchronous context manager to time a stage execution."""
        t_start = time.perf_counter()
        wall_start = datetime.now(timezone.utc)
        stage_trace = StageTrace(
            stage=stage,
            start_time=wall_start,
            metadata=metadata or {},
        )
        try:
            yield stage_trace
            stage_trace.status = "success"
        except Exception as e:
            stage_trace.status = "failed"
            stage_trace.error_message = str(e)
            logger.error(f"Stage '{stage}' failed: {e}", exc_info=True)
            raise
        finally:
            t_end = time.perf_counter()
            stage_trace.end_time = datetime.now(timezone.utc)
            stage_trace.duration_ms = round((t_end - t_start) * 1000, 3)
            self.traces[stage] = stage_trace
            logger.info(
                f"⏱️ [PipelineTracer] Stage '{stage}' completed in {stage_trace.duration_ms:.2f}ms "
                f"[{stage_trace.status}]"
            )

    def record_stage(
        self,
        stage: str,
        duration_ms: float,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> StageTrace:
        """Manually records a stage trace."""
        now = datetime.now(timezone.utc)
        trace = StageTrace(
            stage=stage,
            start_time=now,
            end_time=now,
            duration_ms=round(duration_ms, 3),
            status=status,
            metadata=metadata or {},
            error_message=error_message
        )
        self.traces[stage] = trace
        return trace

    def compute_total(self) -> float:
        """Computes the overall elapsed time from tracer initialization."""
        self.total_duration_ms = round((time.perf_counter() - self.start_perf_time) * 1000, 2)
        return self.total_duration_ms

    def get_summary(self) -> Dict[str, Any]:
        """Returns a consolidated SLA breakdown dictionary."""
        total_ms = self.compute_total()
        sla_target = settings.SLA_TARGET_MS
        within_sla = total_ms <= sla_target
        
        stages_dict = {k: v.to_dict() for k, v in self.traces.items()}
        
        # Check completeness against required stages
        missing_stages = [s for s in REQUIRED_STAGES if s not in self.traces]

        return {
            "document_id": self.document_id,
            "total_duration_ms": total_ms,
            "sla_target_ms": sla_target,
            "within_sla": within_sla,
            "stages_count": len(self.traces),
            "stages": stages_dict,
            "missing_stages": missing_stages,
            "start_time": self.start_wall_time.isoformat(),
        }

    async def persist(self, db_session: AsyncSession, document_id: Optional[str] = None) -> List[CVProcessingTrace]:
        """Persists all recorded traces to the cv_processing_traces database table."""
        doc_id = document_id or self.document_id
        if not doc_id:
            logger.warning("Cannot persist traces: No document_id specified.")
            return []

        total_ms = self.compute_total()
        db_traces: List[CVProcessingTrace] = []

        for stage_name, trace in self.traces.items():
            db_trace = CVProcessingTrace(
                document_id=doc_id,
                stage=stage_name,
                duration_ms=trace.duration_ms,
                start_time=trace.start_time,
                end_time=trace.end_time or trace.start_time,
                status=trace.status,
                metadata_json=trace.metadata,
            )
            db_session.add(db_trace)
            db_traces.append(db_trace)

        # Also persist the total trace
        total_trace = CVProcessingTrace(
            document_id=doc_id,
            stage="total",
            duration_ms=total_ms,
            start_time=self.start_wall_time,
            end_time=datetime.now(timezone.utc),
            status="success" if all(t.status == "success" for t in self.traces.values()) else "failed",
            metadata_json={"within_sla": total_ms <= settings.SLA_TARGET_MS, "sla_target_ms": settings.SLA_TARGET_MS},
        )
        db_session.add(total_trace)
        db_traces.append(total_trace)

        # Update document total_duration_ms
        doc = await db_session.get(CVDocument, doc_id)
        if doc:
            doc.total_duration_ms = total_ms

        await db_session.commit()
        logger.info(f"Persisted {len(db_traces)} trace records for document {doc_id} (Total: {total_ms}ms)")
        return db_traces
