"""Hybrid Serverless & Local Embedding service for sentence-transformers/all-MiniLM-L6-v2."""
import logging
import gc
import json
import urllib.request
from typing import List, Optional
import numpy as np

from backend.app.core.config import settings

logger = logging.getLogger("cv_rag_pipeline.embedder")

_embedding_model = None


def _call_hf_embeddings_api(texts: List[str]) -> Optional[List[List[float]]]:
    """Attempts to call Hugging Face Serverless Inference API for zero-RAM embeddings."""
    if not settings.HF_API_KEY or not texts:
        return None

    url = f"https://api-inference.huggingface.co/models/{settings.EMBEDDING_MODEL_NAME}"
    headers = {
        "Authorization": f"Bearer {settings.HF_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"inputs": texts, "options": {"wait_for_model": True}}).encode("utf-8"),
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    # Check if returned 2D or 3D tensor
                    if isinstance(data[0][0], list):
                        # Mean pool if 3D token embeddings returned
                        arr = np.array(data)
                        pooled = np.mean(arr, axis=1)
                        # Normalize
                        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        return (pooled / norms).tolist()
                    else:
                        arr = np.array(data, dtype=float)
                        norms = np.linalg.norm(arr, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        return (arr / norms).tolist()
    except Exception as e:
        logger.debug(f"HF Serverless embedding API skipped/fallback: {e}")
    return None


def get_embedding_model():
    """Lazy getter for the local embedding model with minimal memory footprint."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading local embedding model '{settings.EMBEDDING_MODEL_NAME}'...")
        try:
            import torch
            torch.set_grad_enabled(False)
            torch.set_num_threads(1)
        except Exception:
            pass

        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
        logger.info(f"Embedding model '{settings.EMBEDDING_MODEL_NAME}' loaded (dim={settings.EMBEDDING_DIM}).")
    return _embedding_model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates normalized 384-dimensional vector embeddings for a list of texts."""
    if not texts:
        return []

    # 1. Try serverless HF API first (0 MB RAM overhead)
    api_result = _call_hf_embeddings_api(texts)
    if api_result is not None:
        return api_result

    # 2. Local SentenceTransformer fallback
    model = get_embedding_model()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    result = embeddings.tolist()
    gc.collect()
    return result


def generate_query_embedding(query: str) -> List[float]:
    """Generates normalized vector embedding for a single search query."""
    res = generate_embeddings([query])
    if res and len(res) > 0:
        return res[0]
    return [0.0] * settings.EMBEDDING_DIM
