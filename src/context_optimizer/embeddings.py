"""Semantic embeddings for relevance scoring.

Optional and fail-safe: if sentence-transformers (or the model) is unavailable,
every function degrades to returning ``None`` and the caller falls back to the
lexical TF-IDF path. This is what lets natural-language queries
(e.g. "real-time conversation with the AI") match code that uses different
words (``WebSocket``, ``streamResponse``, ``transcript``) — something pure
keyword matching cannot do.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get(
    "CONTEXT_OPTIMIZER_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

_model = None
_unavailable = False


def _get_model():
    """Lazily load the embedding model; cache the result. Returns None if unavailable."""
    global _model, _unavailable
    if _model is not None:
        return _model
    if _unavailable:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info("sentence-transformers not installed; semantic scoring disabled")
        _unavailable = True
        return None
    try:
        _model = SentenceTransformer(_DEFAULT_MODEL)
        logger.info("Loaded embedding model %s", _DEFAULT_MODEL)
    except Exception:
        logger.warning("Failed to load embedding model %s; semantic scoring disabled",
                       _DEFAULT_MODEL, exc_info=True)
        _unavailable = True
        return None
    return _model


def is_available() -> bool:
    """True if semantic scoring can run."""
    return _get_model() is not None


def semantic_scores(nodes: list, query: str) -> Optional[dict[str, float]]:
    """Cosine similarity of each node to the query, in [0, 1]-ish (raw cosine).

    Returns None if embeddings are unavailable (caller should fall back to TF-IDF).
    Scores are raw cosine similarities — meaningful as an *absolute* confidence
    signal, unlike min-max-normalized lexical scores.
    """
    if not nodes:
        return {}
    model = _get_model()
    if model is None:
        return None
    # Represent each node by its name + content so both signal sources count.
    texts = [query] + [f"{n.name}\n{n.content or ''}" for n in nodes]
    try:
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        logger.warning("Embedding encode failed; falling back to lexical scoring", exc_info=True)
        return None
    query_vec = embs[0]
    sims = embs[1:] @ query_vec  # normalized vectors → dot product == cosine
    return {n.id: float(sims[i]) for i, n in enumerate(nodes)}
