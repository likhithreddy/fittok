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
    "FITTOK_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

_model = None
_unavailable = False
# Capped FIFO (oldest evicted when full): a long-lived server parsing many
# codebases would otherwise grow this forever. L2 (diskcache) persists, so
# evicting L1 only costs a disk hit.
import collections as _collections
_EMB_CACHE: "_collections.OrderedDict[str, object]" = _collections.OrderedDict()
_EMB_CACHE_MAX = 4096


def _emb_cache_put(k: str, v) -> None:
    """Insert into the L1 embedding cache, evicting oldest (FIFO) when full."""
    _EMB_CACHE[k] = v
    while len(_EMB_CACHE) > _EMB_CACHE_MAX:
        _EMB_CACHE.popitem(last=False)


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
    node_texts = [f"{n.name}\n{n.content or ''}" for n in nodes]
    try:
        node_embs = _embed_cached(model, node_texts)        # reused across queries
        query_vec = model.encode([query], normalize_embeddings=True,
                                 show_progress_bar=False)[0]
    except Exception:
        logger.warning("Embedding encode failed; falling back to lexical scoring", exc_info=True)
        return None
    sims = node_embs @ query_vec  # normalized vectors → dot product == cosine
    return {n.id: float(sims[i]) for i, n in enumerate(nodes)}


def _embed_cached(model, texts: list[str]):
    """Encode texts, reusing a per-process cache keyed by content.

    Node content is stable across queries for a given graph, so without this the
    server re-embedded thousands of nodes on every call (and N times for an
    N-query batch) — seconds of wasted compute each time.
    """
    import hashlib
    import numpy as np
    from . import cache as _cache

    keys = [hashlib.sha256(t.encode("utf-8", "ignore")).hexdigest() for t in texts]
    out: list = [None] * len(texts)
    missing_i, missing_t = [], []
    for i, k in enumerate(keys):
        # L1: in-process cache
        vec = _EMB_CACHE.get(k)
        if vec is None:
            # L2: persistent disk cache (survives restarts; incremental by content)
            vec = _cache.get_cached_embedding(k)
            if vec is not None:
                _emb_cache_put(k, vec)
        if vec is None:
            missing_i.append(i)
            missing_t.append(texts[i])
        else:
            out[i] = vec
    if missing_t:
        enc = model.encode(missing_t, normalize_embeddings=True, show_progress_bar=False)
        for j, i in enumerate(missing_i):
            _emb_cache_put(keys[i], enc[j])
            _cache.set_cached_embedding(keys[i], enc[j])
            out[i] = enc[j]
    return np.array(out)
