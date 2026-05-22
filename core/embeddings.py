"""
core/embeddings.py — Sentence embedding model, loaded once and shared.

Both task_a and task_b call get_embedding_model() and get the same
cached instance — no double download, no double memory usage.
"""
import logging
from typing import List, Optional

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
        from sklearn.metrics.pairwise import cosine_similarity
        q_emb   = model.encode(query_text)
        t_embs  = model.encode(texts)
        sims    = cosine_similarity([q_emb], t_embs)[0]
        top_k   = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return top_k
    except Exception as exc:
        logger.warning(f"cosine_retrieve failed: {exc}")
        return []
