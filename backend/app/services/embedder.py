"""Local embedding service using sentence-transformers/all-MiniLM-L6-v2."""
import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings

logger = logging.getLogger("cv_rag_pipeline.embedder")

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Singleton getter for the prewarmed local embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading local embedding model '{settings.EMBEDDING_MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info(f"Embedding model '{settings.EMBEDDING_MODEL_NAME}' loaded successfully (dim={settings.EMBEDDING_DIM}).")
    return _embedding_model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates normalized 384-dimensional vector embeddings for a list of texts."""
    if not texts:
        return []
    model = get_embedding_model()
    # Batch encode with normalization for direct cosine similarity via dot product
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def generate_query_embedding(query: str) -> List[float]:
    """Generates normalized vector embedding for a single search query."""
    model = get_embedding_model()
    embedding: np.ndarray = model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embedding.tolist()
