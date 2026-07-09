"""3-level caching layer for fittok.

Levels:
  1. Graph cache — keyed by (root_path, file_mtimes_hash)
  2. Query cache — keyed by (graph_path, query_hash, token_budget)
  3. Compression cache — keyed by (context_hash, question_hash, target_tokens)

Uses diskcache for persistent storage. Cache dir: ~/.cache/fittok/
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get(
    "FITTOK_CACHE_DIR",
    os.path.expanduser("~/.cache/fittok"),
)
MAX_CACHE_SIZE = int(os.environ.get("FITTOK_CACHE_MAX_MB", "500")) * 1024 * 1024

# Stats tracking
_stats: dict[str, int] = {"graph_hits": 0, "graph_misses": 0,
                          "query_hits": 0, "query_misses": 0,
                          "compression_hits": 0, "compression_misses": 0}

_cache = None


def _get_cache():
    """Lazily initialize diskcache."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        import diskcache
        os.makedirs(CACHE_DIR, exist_ok=True)
        _cache = diskcache.Cache(CACHE_DIR, size_limit=MAX_CACHE_SIZE)
        return _cache
    except ImportError:
        logger.warning("diskcache not installed — caching disabled. Install with: pip install diskcache")
        return None


# ── Hashing helpers ───────────────────────────────────────────────────────────

def _hash_str(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# Debounce: the mtime walk is O(files) and runs on every get/set_cached_graph.
# Reuse the last hash for a root within a short window — edits are batched by
# the watcher's 2s tick anyway, so a sub-second-stale hash is fine.
# Note: with autowatch on (default), _resolve_graph prefers the watcher's live
# graph and bypasses this cache, so the stale window only bites when autowatch
# is off or the watcher has stopped.
_MTIME_HASH_TTL = 2.0
_mtime_hash_cache: dict[str, tuple[float, str]] = {}


def _hash_mtimes(root_path: str) -> str:
    """Hash of all code-file mtimes under root_path.

    Reuses the parser's skip-set and generated-file filter (single source of
    truth in graphify) so the key reflects exactly what gets parsed — otherwise
    churn in build/coverage/node_modules dirs would cause spurious cache misses.
    """
    import time as _time
    now = _time.monotonic()
    hit = _mtime_hash_cache.get(root_path)
    if hit and (now - hit[0]) < _MTIME_HASH_TTL:
        return hit[1]

    from .graphify import _EXT_TO_LANG, _SKIP_DIRS, _is_generated_file
    mtimes: list[str] = []
    root = Path(root_path).resolve()
    if not root.is_dir():
        h = _hash_str(root_path)
        _mtime_hash_cache[root_path] = (now, h)
        return h
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.suffix in _EXT_TO_LANG and not _is_generated_file(p):
                try:
                    mtimes.append(f"{p.relative_to(root)}:{p.stat().st_mtime}")
                except OSError:
                    pass
    h = _hash_str("|".join(mtimes))
    _mtime_hash_cache[root_path] = (now, h)
    return h


def graph_output_path(resolved_path) -> str:
    """On-disk path for a codebase's graph.json — under the cache dir, NEVER in
    the user's repo (writing graph.json into their codebase pollutes it and can
    get committed). Shared by the server tools AND the watcher so the watcher's
    incremental write-back lands exactly where the file-based tools read.
    """
    graphs_dir = os.path.join(CACHE_DIR, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    digest = hashlib.sha256(str(resolved_path).encode()).hexdigest()[:16]
    name = Path(str(resolved_path)).name or "graph"
    return os.path.join(graphs_dir, f"{name}-{digest}.json")


# ── Graph cache ───────────────────────────────────────────────────────────────

def get_cached_graph(root_path: str) -> Optional[Any]:
    """Look up a cached graph by root path + file mtimes."""
    cache = _get_cache()
    if cache is None:
        return None
    key = f"graph:{root_path}:{_hash_mtimes(root_path)}"
    result = cache.get(key)
    if result is not None:
        _stats["graph_hits"] += 1
        logger.debug("Graph cache HIT: %s", root_path)
        from .models import KnowledgeGraph
        return KnowledgeGraph.model_validate(result)
    _stats["graph_misses"] += 1
    return None


def set_cached_graph(root_path: str, graph: Any) -> None:
    """Store a graph in cache."""
    cache = _get_cache()
    if cache is None:
        return
    key = f"graph:{root_path}:{_hash_mtimes(root_path)}"
    cache.set(key, graph.model_dump(), expire=3600 * 24)  # 24h TTL


# ── Query cache ──────────────────────────────────────────────────────────────

def get_cached_query(graph_path: str, query: str, token_budget: int, graph_version: str = "") -> Optional[dict]:
    """Look up a cached query result. graph_version ties the entry to a specific
    graph revision so stale subgraphs aren't returned after the graph changes."""
    cache = _get_cache()
    if cache is None:
        return None
    key = f"query:{graph_path}:{_hash_str(query)}:{token_budget}:{graph_version}"
    result = cache.get(key)
    if result is not None:
        _stats["query_hits"] += 1
        logger.debug("Query cache HIT: %s", query[:40])
        return result  # type: ignore[return-value]
    _stats["query_misses"] += 1
    return None


def set_cached_query(graph_path: str, query: str, token_budget: int, result: dict, graph_version: str = "") -> None:
    """Store a query result in cache."""
    cache = _get_cache()
    if cache is None:
        return
    key = f"query:{graph_path}:{_hash_str(query)}:{token_budget}:{graph_version}"
    cache.set(key, result, expire=3600)  # 1h TTL


# ── Compression cache ────────────────────────────────────────────────────────

def get_cached_compression(context: str, question: str, target_tokens: int, rate: float | None = None) -> Optional[dict]:
    """Look up a cached compression result."""
    cache = _get_cache()
    if cache is None:
        return None
    key = f"compress:{_hash_str(context)}:{_hash_str(question)}:{target_tokens}:{rate}"
    result = cache.get(key)
    if result is not None:
        _stats["compression_hits"] += 1
        logger.debug("Compression cache HIT")
        return result  # type: ignore[return-value]
    _stats["compression_misses"] += 1
    return None


def set_cached_compression(context: str, question: str, target_tokens: int, result: dict, rate: float | None = None) -> None:
    """Store a compression result in cache."""
    cache = _get_cache()
    if cache is None:
        return
    key = f"compress:{_hash_str(context)}:{_hash_str(question)}:{target_tokens}:{rate}"
    cache.set(key, result, expire=3600)


# ── Embedding cache (persistent, content-keyed) ──────────────────────────────
# Content-keyed means this is incremental by nature: an unchanged function keeps
# its embedding across restarts and edits; only new/changed code is re-embedded.

def get_cached_embedding(content_hash: str):
    """Return a persisted embedding vector for a content hash, or None."""
    cache = _get_cache()
    if cache is None:
        return None
    return cache.get(f"emb:{content_hash}")


def set_cached_embedding(content_hash: str, vector) -> None:
    """Persist an embedding vector (30-day TTL)."""
    cache = _get_cache()
    if cache is None:
        return
    cache.set(f"emb:{content_hash}", vector, expire=3600 * 24 * 30)


# ── Cache management ─────────────────────────────────────────────────────────

def clear_cache(scope: str = "all") -> dict:
    """Clear cache by scope: all, graph, query, compression."""
    cache = _get_cache()
    if cache is None:
        return {"error": "Cache not available (diskcache not installed)"}

    removed = 0
    for key in list(cache.iterkeys()):
        if scope == "all":
            cache.delete(key)
            removed += 1
        elif scope == "graph" and str(key).startswith("graph:"):
            cache.delete(key)
            removed += 1
        elif scope == "query" and str(key).startswith("query:"):
            cache.delete(key)
            removed += 1
        elif scope == "compression" and str(key).startswith("compress:"):
            cache.delete(key)
            removed += 1

    return {"cleared": removed, "scope": scope}


def cache_stats() -> dict:
    """Return cache hit/miss stats and size."""
    cache = _get_cache()
    size = 0
    count = 0
    if cache is not None:
        try:
            count = len(cache)  # type: ignore[arg-type]
            size = cache.volume()
        except Exception:
            pass

    return {
        "stats": dict(_stats),
        "entries": count,
        "size_bytes": size,
        "cache_dir": CACHE_DIR,
        "available": cache is not None,
    }
