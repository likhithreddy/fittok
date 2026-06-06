"""3-level caching layer for context-optimizer.

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
    "CONTEXT_OPTIMIZER_CACHE_DIR",
    os.path.expanduser("~/.cache/fittok"),
)
MAX_CACHE_SIZE = int(os.environ.get("CONTEXT_OPTIMIZER_CACHE_MAX_MB", "500")) * 1024 * 1024

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


def _hash_mtimes(root_path: str) -> str:
    """Hash of all code file mtimes under root_path."""
    from .graphify import _EXT_TO_LANG
    mtimes: list[str] = []
    root = Path(root_path).resolve()
    if not root.is_dir():
        return _hash_str(root_path)
    skip_dirs = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".next", ".nuxt", "target", "vendor", ".gradle",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.suffix in _EXT_TO_LANG:
                try:
                    mtimes.append(f"{p.relative_to(root)}:{p.stat().st_mtime}")
                except OSError:
                    pass
    return _hash_str("|".join(mtimes))


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

def get_cached_query(graph_path: str, query: str, token_budget: int) -> Optional[dict]:
    """Look up a cached query result."""
    cache = _get_cache()
    if cache is None:
        return None
    key = f"query:{graph_path}:{_hash_str(query)}:{token_budget}"
    result = cache.get(key)
    if result is not None:
        _stats["query_hits"] += 1
        logger.debug("Query cache HIT: %s", query[:40])
        return result  # type: ignore[return-value]
    _stats["query_misses"] += 1
    return None


def set_cached_query(graph_path: str, query: str, token_budget: int, result: dict) -> None:
    """Store a query result in cache."""
    cache = _get_cache()
    if cache is None:
        return
    key = f"query:{graph_path}:{_hash_str(query)}:{token_budget}"
    cache.set(key, result, expire=3600)  # 1h TTL


# ── Compression cache ────────────────────────────────────────────────────────

def get_cached_compression(context: str, question: str, target_tokens: int) -> Optional[dict]:
    """Look up a cached compression result."""
    cache = _get_cache()
    if cache is None:
        return None
    key = f"compress:{_hash_str(context)}:{_hash_str(question)}:{target_tokens}"
    result = cache.get(key)
    if result is not None:
        _stats["compression_hits"] += 1
        logger.debug("Compression cache HIT")
        return result  # type: ignore[return-value]
    _stats["compression_misses"] += 1
    return None


def set_cached_compression(context: str, question: str, target_tokens: int, result: dict) -> None:
    """Store a compression result in cache."""
    cache = _get_cache()
    if cache is None:
        return
    key = f"compress:{_hash_str(context)}:{_hash_str(question)}:{target_tokens}"
    cache.set(key, result, expire=3600)


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
