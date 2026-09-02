"""Vector Indexing API endpoint — POST /api/v1/index."""
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.app.db.session import get_db
from backend.app.models.cv_document import CVDocument
from backend.app.services.embedder import generate_embeddings

logger = logging.getLogger("cv_rag_pipeline.api.index")

router = APIRouter()


class IndexChunkItem(BaseModel):
    chunk_index: int
    section_name: str
    content: str
    token_count: Optional[int] = 0
    metadata: Optional[Dict[str, Any]] = None


class IndexRequest(BaseModel):
    document_id: str
    chunks: Optional[List[IndexChunkItem]] = None


@router.post(
    "",
    summary="Index CV Chunks in Vector DB",
    description="Generates embeddings for CV chunks and upserts them into Supabase pgvector table via raw SQL CAST."
)
async def index_cv_chunks(
    request: IndexRequest,
    db: AsyncSession = Depends(get_db)
):
    doc = await db.get(CVDocument, request.document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CV document '{request.document_id}' not found."
        )

    # Determine chunks to index
    chunks_to_index: List[Dict[str, Any]] = []
    if request.chunks:
        chunks_to_index = [c.model_dump() for c in request.chunks]
    else:
        chunks_to_index = [
            {
                "chunk_index": c.chunk_index,
                "section_name": c.section_name,
                "content": c.content,
                "token_count": c.token_count,
                "metadata": c.metadata_json,
            }
            for c in doc.chunks
        ]

    if not chunks_to_index:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks available to index."
        )

    # Generate embeddings
    texts = [c["content"] for c in chunks_to_index]
    embeddings = generate_embeddings(texts)

    # Upsert via raw SQL CAST(:embedding AS vector)
    # This bypasses asyncpg codec registration, which is unreliable on free-tier pooled connections.
    now = datetime.now(timezone.utc)
    created_count = 0

    for c_data, emb in zip(chunks_to_index, embeddings):
        emb_data = emb
        if isinstance(emb_data, str):
            try:
                emb_data = json.loads(emb_data)
            except Exception:
                emb_data = []

        clean_floats = [float(x) for x in emb_data] if emb_data else []
        embedding_str = (
            f"[{','.join(str(v) for v in clean_floats)}]" if clean_floats else None
        )

        chunk_meta = c_data.get("metadata") or {}
        if isinstance(chunk_meta, str):
            try:
                chunk_meta = json.loads(chunk_meta)
            except Exception:
                chunk_meta = {}

        chunk_id = str(uuid.uuid4())

        if embedding_str:
            await db.execute(
                text("""
                    INSERT INTO cv_chunks
                        (id, document_id, chunk_index, section_name,
                         content, token_count, embedding, metadata_json, created_at)
                    VALUES
                        (:id, :document_id, :chunk_index, :section_name,
                         :content, :token_count, CAST(:embedding AS vector),
                         CAST(:metadata_json AS json), :created_at)
                    ON CONFLICT (id) DO NOTHING
                """),
                {
                    "id": chunk_id,
                    "document_id": doc.id,
                    "chunk_index": c_data["chunk_index"],
                    "section_name": c_data["section_name"],
                    "content": c_data["content"],
                    "token_count": c_data.get("token_count", 0),
                    "embedding": embedding_str,
                    "metadata_json": json.dumps(chunk_meta),
                    "created_at": now,
                }
            )
        else:
            await db.execute(
                text("""
                    INSERT INTO cv_chunks
                        (id, document_id, chunk_index, section_name,
                         content, token_count, metadata_json, created_at)
                    VALUES
                        (:id, :document_id, :chunk_index, :section_name,
                         :content, :token_count, CAST(:metadata_json AS json), :created_at)
                    ON CONFLICT (id) DO NOTHING
                """),
                {
                    "id": chunk_id,
                    "document_id": doc.id,
                    "chunk_index": c_data["chunk_index"],
                    "section_name": c_data["section_name"],
                    "content": c_data["content"],
                    "token_count": c_data.get("token_count", 0),
                    "metadata_json": json.dumps(chunk_meta),
                    "created_at": now,
                }
            )
        created_count += 1

    doc.status = "rag_ready"
    await db.commit()

    return {
        "document_id": doc.id,
        "status": "rag_ready",
        "chunks_indexed": created_count,
        "embedding_dim": 384,
        "message": f"Successfully indexed {created_count} chunks for document {doc.id} in pgvector."
    }
