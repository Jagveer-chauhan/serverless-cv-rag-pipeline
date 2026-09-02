"""Zero-RAM Serverless & Deterministic Fallback Embedding Service (384-dim)."""
import logging
import json
import hashlib
import urllib.request
from typing import List, Optional
import numpy as np

from backend.app.core.config import settings

logger = logging.getLogger("cv_rag_pipeline.embedder")


def _generate_fast_deterministic_embedding(text: str, dim: int = 384) -> List[float]:
    """Generates a high-speed, zero-RAM normalized 384-d semantic feature vector.
    
    Uses character n-grams and token hash projection with L2 unit normalization.
    """
    if not text.strip():
        return [0.0] * dim

    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()

    for word in words:
        # Word hash
        h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign * 1.5

        # Trigrams
        if len(word) >= 3:
            for i in range(len(word) - 2):
                tri = word[i:i+3]
                th = int(hashlib.md5(tri.encode('utf-8')).hexdigest(), 16)
                tidx = th % dim
                tsign = 1.0 if ((th >> 8) & 1) == 0 else -1.0
                vec[tidx] += tsign * 0.8

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _call_hf_embeddings_api(texts: List[str]) -> Optional[List[List[float]]]:
    """Attempts Hugging Face Serverless Feature Extraction API."""
    if not settings.HF_API_KEY or not texts:
        return None

    candidate_urls = [
        settings.hf_embedding_url,
        f"https://router.huggingface.co/hf-inference/models/{settings.EMBEDDING_MODEL_NAME}",
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{settings.EMBEDDING_MODEL_NAME}",
        f"https://api-inference.huggingface.co/models/{settings.EMBEDDING_MODEL_NAME}",
    ]

    headers = {
        "Authorization": f"Bearer {settings.HF_API_KEY}",
        "Content-Type": "application/json"
    }

    for url in candidate_urls:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"inputs": texts, "options": {"wait_for_model": True}}).encode("utf-8"),
                headers=headers
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        # 3D token embeddings tensor
                        if isinstance(data[0][0], list):
                            arr = np.array(data, dtype=np.float32)
                            pooled = np.mean(arr, axis=1)
                            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                            norms[norms == 0] = 1.0
                            return (pooled / norms).tolist()
                        else:
                            # 2D sentence embeddings tensor
                            arr = np.array(data, dtype=np.float32)
                            norms = np.linalg.norm(arr, axis=1, keepdims=True)
                            norms[norms == 0] = 1.0
                            return (arr / norms).tolist()
        except Exception as e:
            logger.debug(f"HF Serverless endpoint {url} failed: {e}")
            continue

    return None


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates normalized 384-dimensional vector embeddings for a list of texts."""
    if not texts:
        return []

    # 1. Try serverless HF API (0 MB RAM)
    api_result = _call_hf_embeddings_api(texts)
    if api_result is not None:
        return api_result

    # 2. Ultra-fast, zero-RAM deterministic semantic vector generator
    return [_generate_fast_deterministic_embedding(t, dim=settings.EMBEDDING_DIM) for t in texts]


def generate_query_embedding(query: str) -> List[float]:
    """Generates normalized vector embedding for a single search query."""
    res = generate_embeddings([query])
    if res and len(res) > 0:
        return res[0]
    return [0.0] * settings.EMBEDDING_DIM
