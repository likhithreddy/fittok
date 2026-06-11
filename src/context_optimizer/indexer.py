"""Pre-index a codebase before using the MCP.

Builds + caches the knowledge graph and warms/persists embeddings, so the first
MCP query is instant instead of paying the one-time parse (~1s) + cold embedding
(~10s+) cost. Re-running after code changes is incremental: only new/changed
functions are re-embedded (embeddings are content-keyed and persisted on disk),
unchanged code is reused.

Usage:
    python -m context_optimizer.indexer /path/to/repo
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from . import embeddings
from .cache import set_cached_graph
from .graphify import parse_codebase
from .models import NodeType

logger = logging.getLogger(__name__)


def index_codebase(path: str) -> dict:
    """Parse + cache the graph and warm/persist embeddings for *path*."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {path}")

    t = time.perf_counter()
    graph = parse_codebase(str(root))
    set_cached_graph(str(root), graph)
    parse_s = time.perf_counter() - t

    content = [
        n for n in graph.nodes
        if n.type != NodeType.FILE or (n.content and n.content.strip())
    ]

    embedded = 0
    t = time.perf_counter()
    if embeddings.is_available() and content:
        from .embeddings import _embed_cached, _get_model
        model = _get_model()
        texts = [f"{n.name}\n{n.content or ''}" for n in content]
        _embed_cached(model, texts)  # fills the in-process + persistent caches
        embedded = len(texts)
    embed_s = time.perf_counter() - t

    return {
        "root": str(root),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "embedded": embedded,
        "parse_s": round(parse_s, 1),
        "embed_s": round(embed_s, 1),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m context_optimizer.indexer <codebase_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    print(f"Indexing {path} ...")
    try:
        r = index_codebase(path)
    except Exception as e:
        print(f"Index failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(
        f"Done: {r['nodes']} nodes / {r['edges']} edges, {r['embedded']} embeddings "
        f"(parse {r['parse_s']}s, embed {r['embed_s']}s).\n"
        f"Cached — the first MCP query against this repo will now be instant. "
        f"Re-run after code changes; only changed functions re-embed."
    )


if __name__ == "__main__":
    main()
