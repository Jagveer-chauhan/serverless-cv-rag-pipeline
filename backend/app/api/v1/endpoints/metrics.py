"""Metrics and SLA observability endpoint answering assignment benchmarks."""
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.app.db.session import get_db
from backend.app.models.cv_document import CVDocument
from backend.app.models.cv_processing_trace import CVProcessingTrace
from backend.app.core.config import settings
from backend.app.core.state import app_state

logger = logging.getLogger("cv_rag_pipeline.metrics")

router = APIRouter()


@router.get(
    "",
    summary="Observability & SLA Benchmark Metrics",
    description="Provides aggregate statistics on warm-path SLA compliance (p50/p95/p99), dominant stage bottleneck, cold starts, and OCR occurrences."
)
async def get_metrics(db: AsyncSession = Depends(get_db)):
    # Query all CVs
    cv_result = await db.execute(select(CVDocument))
    cvs = cv_result.scalars().all()
    total_cvs = len(cvs)

    if total_cvs == 0:
        return {
            "total_cvs_processed": 0,
            "sla_target_ms": settings.SLA_TARGET_MS,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "percentage_within_5s_sla": 100.0,
            "dominant_bottleneck_stage": "none",
            "cold_starts_count": 0 if app_state.is_warm else 1,
            "is_worker_warm": app_state.is_warm,
            "stage_averages_ms": {},
            "status_breakdown": {}
        }

    durations = [c.total_duration_ms for c in cvs if c.total_duration_ms is not None]
    durations.sort()

    def _percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        k = (len(data) - 1) * p
        f = int(k)
        c = min(f + 1, len(data) - 1)
        d = k - f
        return round(data[f] + d * (data[c] - data[f]), 2)

    p50 = _percentile(durations, 0.50)
    p95 = _percentile(durations, 0.95)
    p99 = _percentile(durations, 0.99)

    within_sla_count = sum(1 for d in durations if d <= settings.SLA_TARGET_MS)
    pct_within_sla = round((within_sla_count / max(len(durations), 1)) * 100, 1)

    # Query per-stage trace averages
    trace_result = await db.execute(
        select(
            CVProcessingTrace.stage,
            func.avg(CVProcessingTrace.duration_ms).label("avg_duration"),
            func.count(CVProcessingTrace.id).label("count")
        ).where(CVProcessingTrace.stage != "total").group_by(CVProcessingTrace.stage)
    )
    stage_rows = trace_result.fetchall()

    stage_averages = {r[0]: round(float(r[1]), 2) for r in stage_rows}

    # Find dominant bottleneck stage
    dominant_stage = "none"
    if stage_averages:
        dominant_stage = max(stage_averages.items(), key=lambda x: x[1])[0]

    # Status distribution
    status_counts = {}
    for c in cvs:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1

    return {
        "total_cvs_processed": total_cvs,
        "sla_target_ms": settings.SLA_TARGET_MS,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "percentage_within_5s_sla": pct_within_sla,
        "dominant_bottleneck_stage": dominant_stage,
        "stage_averages_ms": stage_averages,
        "status_breakdown": status_counts,
        "cold_starts_count": 0 if app_state.is_warm else 1,
        "is_worker_warm": app_state.is_warm,
        "uptime_seconds": round(app_state.uptime_seconds, 1)
    }
