"""Main FastAPI Application Entrypoint."""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.core.state import app_state
from backend.app.api.v1.router import api_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cv_rag_pipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown tasks."""
    logger.info("Starting up Serverless CV Parsing & RAG Pipeline...")
    app_state.start_time = time.time()
    
    # Warmup routines (pre-load lightweight models or test connections if configured)
    try:
        # Mark warm once initialized
        app_state.is_warm = True
        logger.info(f"Pipeline warmed up successfully. SLA Target: {settings.SLA_TARGET_MS}ms")
    except Exception as e:
        logger.error(f"Error during warmup: {e}", exc_info=True)
        app_state.is_warm = False
        
    yield
    
    logger.info("Shutting down Serverless CV Parsing & RAG Pipeline...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="High-performance Serverless CV Parsing, Extraction, and RAG Pipeline with warm-path p95 SLA <= 5.0s",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Global timing middleware recording millisecond duration per HTTP request."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    return response


# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.api_route("/health", methods=["GET", "HEAD"], summary="Basic Health Check")
async def health():
    return {
        "status": "healthy",
        "warm": app_state.is_warm,
        "uptime_seconds": round(app_state.uptime_seconds, 2),
        "llm_model": settings.HF_MODEL_NAME,
        "embedding_model": settings.EMBEDDING_MODEL_NAME
    }


@app.api_route("/", methods=["GET", "HEAD"], summary="Root Health Status")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "status": "online",
        "warm": app_state.is_warm,
        "sla_target_ms": settings.SLA_TARGET_MS,
        "health": "/health",
        "docs": "/docs",
        "keepalive": f"{settings.API_V1_STR}/keepalive"
    }
