"""API v1 router registry."""
from fastapi import APIRouter
from backend.app.api.v1.endpoints import keepalive, cvs, chat, index_chunks, metrics

api_router = APIRouter()

# Keepalive and health routes
api_router.include_router(keepalive.router, tags=["Health & Keepalive"])

# CV Upload and Management routes
api_router.include_router(cvs.router, prefix="/cvs", tags=["CV Ingestion & Management"])

# RAG Chat SSE routes
api_router.include_router(chat.router, prefix="/chat", tags=["RAG Chat"])

# Vector Indexing routes
api_router.include_router(index_chunks.router, prefix="/index", tags=["Vector Indexing"])

# Observability Metrics routes
api_router.include_router(metrics.router, prefix="/metrics", tags=["Metrics & Observability"])
