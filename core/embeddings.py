"""
core/embeddings.py — Sentence embedding model, loaded once and shared.

Both task_a and task_b call get_embedding_model() and get the same
cached instance — no double download, no double memory usage.
"""
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

_model = None  


def get_embedding_model(model_name: str = None):
    """
    Return (and lazily initialise) the sentence-transformer model.
    Subsequent calls return the cached instance immediately.

    """
    global _model
    if _model is not None:
        return _model

    if model_name is None:
        from core.config import EMBEDDING_MODEL
        model_name = EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded: {model_name}")
    except Exception as exc:
        logger.warning(f"Embedding model unavailable ({model_name}): {exc}")
        _model = None

    return _model


def cosine_retrieve(
    query_text: str,
    texts: List[str],
    model=None,
    k: int = 3
) -> List[int]:
    """
    Encode query_text and all texts, return indices of the top-k
    most similar texts by cosine similarity.

    Returns an empty list if the model is unavailable or texts is empty.
    """
    if model is None:
        model = get_embedding_model()
    if model is None or not texts:
        return []

    try:
        q_emb   = model.encode(query_text)
        t_embs  = model.encode(texts)
        sims    = _cosine_similarity(np.asarray([q_emb]), np.asarray(t_embs))[0]
        top_k   = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return top_k
    except Exception as exc:
        logger.warning(f"cosine_retrieve failed: {exc}")
        return []


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy-only replacement for sklearn.metrics.pairwise.cosine_similarity."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    # Guard against zero-division on empty embeddings
    a_norm = np.where(a_norm == 0, 1, a_norm)
    b_norm = np.where(b_norm == 0, 1, b_norm)
    return np.dot(a, b.T) / (a_norm * b_norm.T)
