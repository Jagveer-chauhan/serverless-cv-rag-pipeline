"""Keepalive and health check endpoints to prevent Render idle spin-down."""
import os
import time
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field
import psutil

from backend.app.core.config import settings
from backend.app.core.state import app_state

router = APIRouter()


class KeepaliveResponse(BaseModel):
    status: str = Field(default="warm", description="Health and warmth status")
    timestamp: str = Field(..., description="Current UTC ISO timestamp")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    memory_usage_mb: float = Field(..., description="Resident set memory usage in MB")
    models_prewarmed: bool = Field(..., description="Whether embedding models are prewarmed in memory")
    sla_target_ms: float = Field(..., description="Target SLA threshold in milliseconds")
    environment: str = Field(..., description="Runtime environment")
    message: str = Field(..., description="Status summary message")


def _generate_keepalive_response() -> KeepaliveResponse:
    app_state.last_keepalive = time.time()
    
    # Measure memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_mb = round(mem_info.rss / (1024 * 1024), 2)
    
    return KeepaliveResponse(
        status="warm" if app_state.is_warm else "warming_up",
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=round(app_state.uptime_seconds, 2),
        memory_usage_mb=memory_mb,
        models_prewarmed=app_state.is_warm,
        sla_target_ms=settings.SLA_TARGET_MS,
        environment=settings.ENVIRONMENT,
        message="Render worker is active and warm. Ready for sub-5.0s SLA CV processing."
    )


@router.get(
    "/keepalive",
    response_model=KeepaliveResponse,
    summary="Keepalive health check",
    description="Lightweight endpoint pinged by external cron/uptime monitors to prevent Render 15-minute idle spin-down."
)
async def keepalive_get():
    return _generate_keepalive_response()


@router.post(
    "/keepalive",
    response_model=KeepaliveResponse,
    summary="Keepalive webhook trigger",
    description="POST webhook endpoint for scheduled cron triggers to refresh warm state."
)
async def keepalive_post(
    x_keepalive_token: Optional[str] = Header(None, alias="X-Keepalive-Token")
):
    if settings.KEEPALIVE_SECRET and settings.KEEPALIVE_SECRET.strip() and x_keepalive_token != settings.KEEPALIVE_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Keepalive-Token header"
        )
    return _generate_keepalive_response()
