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


async def search_similar_chunks(
    db: AsyncSession,
    query_text: str,
    document_id: Optional[str] = None,
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """Retrieves top-k most relevant chunks using vector similarity search."""
    query_vector = generate_query_embedding(query_text)
    
    # Check if database is PostgreSQL with pgvector
    bind = db.bind
    is_postgres = bind and "postgresql" in str(bind.url)

    if is_postgres:
        vec_str = f"[{','.join(map(str, query_vector))}]"
        sql_query = """
            SELECT id, document_id, chunk_index, section_name, content, metadata_json,
                   1 - (embedding <=> :vec::vector) AS similarity
            FROM cv_chunks
        """
        params: Dict[str, Any] = {"vec": vec_str}
        if document_id:
            sql_query += " WHERE document_id = :doc_id"
            params["doc_id"] = document_id
        sql_query += " ORDER BY embedding <=> :vec::vector ASC LIMIT :top_k"
        params["top_k"] = top_k

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
            logger.warning(f"PostgreSQL pgvector native search failed, falling back to python calculation: {pg_err}")

    # Fallback / In-Memory Similarity for test environments or SQLite
    stmt = select(CVChunk)
    if document_id:
        stmt = stmt.where(CVChunk.document_id == document_id)
    
    result = await db.execute(stmt)
    chunks = result.scalars().all()

    scored_chunks = []
    for c in chunks:
        if c.embedding:
            # Stored embedding could be list or pgvector Vector object
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
