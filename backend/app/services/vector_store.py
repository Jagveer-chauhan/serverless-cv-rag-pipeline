"""Vector store and similarity search service with pgvector and fallback support."""
import logging
import json
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.cv_chunk import CVChunk
from backend.app.services.embedder import generate_query_embedding

logger = logging.getLogger("cv_rag_pipeline.vector_store")


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two normalized vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _is_postgres(db: AsyncSession) -> bool:
    """Safely detect if the session is backed by PostgreSQL (SQLAlchemy 2.0 compatible)."""
    try:
        # SQLAlchemy 2.0: session.bind is deprecated; reach the engine via get_bind()
        engine = db.get_bind()
        return "postgresql" in str(engine.url)
    except Exception:
        pass
    try:
        # Fallback for older SQLAlchemy / custom setups
        bind = db.bind  # type: ignore[attr-defined]
        return bool(bind and "postgresql" in str(bind.url))
    except Exception:
        return False


async def search_similar_chunks(
    db: AsyncSession,
    query_text: str,
    document_id: Optional[str] = None,
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """Retrieves top-k most relevant chunks using vector similarity search.

    Strategy:
    1. Try native pgvector similarity search (fast, server-side ranking).
    2. On any SQL failure, ROLLBACK to clear the aborted transaction, then
       fall back to in-process cosine similarity over ORM-fetched chunks.
    """
    query_vector = generate_query_embedding(query_text)
    vec_str = f"[{','.join(str(v) for v in query_vector)}]"

    if _is_postgres(db):
        # Build the similarity query using CAST(:vec AS vector) instead of :vec::vector.
        # SQLAlchemy's text() tokenizer can mis-parse the :: cast syntax when a bind
        # parameter name immediately precedes it (e.g. :vec::vector).
        where_clause = "WHERE document_id = :doc_id" if document_id else ""
        sql_query = f"""
            SELECT id, document_id, chunk_index, section_name, content, metadata_json,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM cv_chunks
            {where_clause}
            ORDER BY embedding <=> CAST(:vec AS vector) ASC
            LIMIT :top_k
        """
        params: Dict[str, Any] = {"vec": vec_str, "top_k": top_k}
        if document_id:
            params["doc_id"] = document_id

        try:
            result = await db.execute(text(sql_query), params)
            rows = result.fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "document_id": r[1],
                    "chunk_index": r[2],
                    "section_name": r[3],
                    "content": r[4],
                    "metadata": r[5],
                    "similarity": round(float(r[6]), 4),
                }
                for r in rows
            ]
        except Exception as pg_err:
            logger.warning(
                f"pgvector native search failed — rolling back and using in-process fallback: {pg_err}"
            )
            # CRITICAL: the failed SQL statement aborted the PostgreSQL transaction.
            # We MUST rollback before issuing any further statements, otherwise every
            # subsequent query will fail with InFailedSQLTransactionError.
            try:
                await db.rollback()
            except Exception as rb_err:
                logger.warning(f"Rollback after pgvector failure also failed: {rb_err}")

    # -------------------------------------------------------------------------
    # Fallback: fetch all chunks for this document and rank in Python.
    # Works on any backend (SQLite, PostgreSQL without pgvector, etc.)
    # -------------------------------------------------------------------------
    try:
        stmt = select(CVChunk)
        if document_id:
            stmt = stmt.where(CVChunk.document_id == document_id)

        result = await db.execute(stmt)
        chunks = result.scalars().all()
    except Exception as fetch_err:
        logger.error(f"Fallback chunk fetch also failed: {fetch_err}")
        return []

    scored_chunks = []
    for c in chunks:
        if c.embedding:
            emb = list(c.embedding) if hasattr(c.embedding, "__iter__") else c.embedding
            if isinstance(emb, list):
                sim = compute_cosine_similarity(query_vector, emb)
                scored_chunks.append({
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "section_name": c.section_name,
                    "content": c.content,
                    "metadata": c.metadata_json,
                    "similarity": round(sim, 4),
                })

    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:top_k]
