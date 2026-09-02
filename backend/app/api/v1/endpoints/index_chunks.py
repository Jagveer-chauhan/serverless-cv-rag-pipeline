"""Vector Indexing API endpoint."""
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.app.db.session import get_db
from backend.app.models.cv_document import CVDocument
from backend.app.models.cv_chunk import CVChunk
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
    description="Generates embeddings for CV chunks and upserts them into Supabase pgvector table."
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

    # Upsert chunks into database
    created_chunks = []
    for c_data, emb in zip(chunks_to_index, embeddings):
        chunk_obj = CVChunk(
            document_id=doc.id,
            chunk_index=c_data["chunk_index"],
            section_name=c_data["section_name"],
            content=c_data["content"],
            token_count=c_data.get("token_count", 0),
            embedding=emb,
            metadata_json=c_data.get("metadata")
        )
        db.add(chunk_obj)
        created_chunks.append(chunk_obj)

    doc.status = "rag_ready"
    await db.commit()

    return {
        "document_id": doc.id,
        "status": "rag_ready",
        "chunks_indexed": len(created_chunks),
        "embedding_dim": len(embeddings[0]) if embeddings else 384,
        "message": f"Successfully indexed {len(created_chunks)} chunks for document {doc.id} in pgvector."
    }
