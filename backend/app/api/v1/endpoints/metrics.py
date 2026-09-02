"""Metrics and SLA observability endpoint — answers all 5 observability questions from spec §3.6."""
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
    description=(
        "Provides aggregate statistics answering all 5 spec observability questions: "
        "stage durations, p50/p95/p99 SLA compliance, dominant bottleneck, "
        "cold-start count, and OCR/retry/failure rate."
    )
)
async def get_metrics(db: AsyncSession = Depends(get_db)):
    # Query all CVs
    cv_result = await db.execute(select(CVDocument))
    cvs = cv_result.scalars().all()
    total_cvs = len(cvs)

    empty_response = {
        "total_cvs_processed": 0,
        "sla_target_ms": settings.SLA_TARGET_MS,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "min_latency_ms": 0.0,
        "max_latency_ms": 0.0,
        "percentage_within_5s_sla": 100.0,
        "dominant_bottleneck_stage": "none",
        # Cold-start tracking (spec §3.6)
        "cold_start_occurred": app_state.cold_start_occurred,
        "cold_start_ms": app_state.cold_start_ms,
        "first_inference_ms": app_state.first_inference_ms,
        "warm_inference_ms": app_state.warm_inference_ms,
        "llm_call_count": app_state.llm_call_count,
        "is_worker_warm": app_state.is_warm,
        "uptime_seconds": round(app_state.uptime_seconds, 1),
        "stage_averages_ms": {},
        "status_breakdown": {},
        # Observability answers (spec §3.6 — 5 required questions)
        "observability": {
            "q1_stage_durations": "No data yet",
            "q2_pct_within_5s": "100% (no CVs processed)",
            "q3_dominant_bottleneck": "none",
            "q4_cold_starts": "0 cold starts observed",
            "q5_ocr_retry_failures": "No data yet"
        }
    }

    if total_cvs == 0:
        return empty_response

    durations = sorted([c.total_duration_ms for c in cvs if c.total_duration_ms is not None])

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

    # Dominant bottleneck stage (Q3)
    dominant_stage = "none"
    if stage_averages:
        dominant_stage = max(stage_averages.items(), key=lambda x: x[1])[0]

    # Status distribution
    status_counts: Dict[str, int] = {}
    for c in cvs:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1

    failed_count = status_counts.get("failed", 0) + status_counts.get("degraded", 0)
    failure_rate = round((failed_count / total_cvs) * 100, 1) if total_cvs > 0 else 0.0

    return {
        "total_cvs_processed": total_cvs,
        "sla_target_ms": settings.SLA_TARGET_MS,
        # Latency percentiles
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "min_latency_ms": round(min(durations), 2) if durations else 0.0,
        "max_latency_ms": round(max(durations), 2) if durations else 0.0,
        "percentage_within_5s_sla": pct_within_sla,
        "dominant_bottleneck_stage": dominant_stage,
        # Per-stage averages
        "stage_averages_ms": stage_averages,
        # Status breakdown
        "status_breakdown": status_counts,
        "failure_rate_pct": failure_rate,
        # Cold-start tracking (spec §3.6)
        "cold_start_occurred": app_state.cold_start_occurred,
        "cold_start_ms": app_state.cold_start_ms,
        "first_inference_ms": app_state.first_inference_ms,
        "warm_inference_ms": app_state.warm_inference_ms,
        "llm_call_count": app_state.llm_call_count,
        "is_worker_warm": app_state.is_warm,
        "uptime_seconds": round(app_state.uptime_seconds, 1),
        # Structured observability answers (spec §3.6 — 5 required questions)
        "observability": {
            "q1_stage_durations": stage_averages,
            "q2_pct_within_5s": f"{pct_within_sla}% of {total_cvs} CVs met p95 ≤ 5.0s",
            "q3_dominant_bottleneck": dominant_stage,
            "q4_cold_starts": (
                f"1 cold start observed: {app_state.cold_start_ms}ms boot → "
                f"first inference {app_state.first_inference_ms}ms"
                if app_state.cold_start_occurred
                else "No cold start observed in this session"
            ),
            "q5_ocr_retry_failures": (
                f"{failure_rate}% failure rate; "
                f"{status_counts.get('degraded', 0)} degraded; "
                f"{status_counts.get('failed', 0)} failed"
            )
        }
    }
