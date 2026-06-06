"""MCP server for context optimization — v0.2.0.

Tools:
  v0.1.0:
    - parse_codebase: Parse code into a knowledge graph
    - query_graph: Query the graph for relevant subgraph
    - compress_context: Compress text using LLMLingua
    - optimize_context: Full pipeline in one call
  v0.2.0:
    - optimize_context_stream: Streaming pipeline with progress
    - optimize_context_batch: Multi-query batching
    - optimize_context_structured: JSON structured output
    - parse_codebase_stream: Chunked parsing with progress
    - watch_start / watch_stop / get_graph_stats / reset_graph: Watch mode
    - diff_graph: Compare two graphs
    - scrub_text / scrub_file / list_pii_patterns / add_pii_pattern: PII scrubbing
    - clear_cache / cache_stats: Cache management
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from .cache import (
    get_cached_graph,
    set_cached_graph,
    get_cached_query,
    set_cached_query,
    get_cached_compression,
    set_cached_compression,
    clear_cache as _clear_cache,
    cache_stats as _cache_stats,
)
from .diff import diff_graphs
from .graphify import (
    load_graph,
    parse_codebase,
    parse_codebase_stream,
    save_graph,
)
from .llmlingua_wrapper import compress_context as _compress
from .pii_scrubber import (
    scrub_text as _scrub_text,
    scrub_file as _scrub_file,
    list_pii_patterns as _list_pii_patterns,
    add_pii_pattern as _add_pii_pattern,
    scrub_graph_content,
)
from .slurp import query_graph as _query_graph
from .watcher import start_watch, stop_watch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "context-optimizer",
    instructions="Filter and compress LLM context using graph-based analysis + LLMLingua. v0.2.0",
)

SCRUB_ENABLED = os.environ.get("CONTEXT_OPTIMIZER_SCRUB", "false").lower() in ("true", "1", "yes")


# ── v0.1.0 Tools ─────────────────────────────────────────────────────────────

@mcp.tool()
def parse_codebase_tool(path: str) -> dict:
    """Parse all code files in a directory into a knowledge graph."""
    logger.info("parse_codebase: %s", path)
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {path}"}

    # Check cache
    cached = get_cached_graph(str(resolved))
    if cached is not None:
        output_path = str(resolved / "graph.json")
        save_graph(cached, output_path)
        return {
            "graph_json_path": output_path,
            "total_nodes": cached.metadata.total_nodes,
            "total_edges": cached.metadata.total_edges,
            "cached": True,
        }

    graph = parse_codebase(str(resolved))
    if SCRUB_ENABLED:
        scrub_graph_content(graph)
    output_path = str(resolved / "graph.json")
    graph_json_path = save_graph(graph, output_path)
    set_cached_graph(str(resolved), graph)

    return {
        "graph_json_path": graph_json_path,
        "total_nodes": graph.metadata.total_nodes,
        "total_edges": graph.metadata.total_edges,
    }


@mcp.tool()
def query_graph_tool(
    graph_path: str,
    query: str,
    token_budget: int = 4000,
) -> dict:
    """Query a knowledge graph for the most relevant subgraph within a token budget."""
    logger.info("query_graph: %s (query=%r, budget=%d)", graph_path, query[:80], token_budget)

    if not os.path.isfile(graph_path):
        return {"error": f"Graph file not found: {graph_path}"}

    # Check cache
    cached = get_cached_query(graph_path, query, token_budget)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        graph = load_graph(graph_path)
    except Exception as e:
        return {"error": f"Failed to load graph: {e}"}

    markdown, node_count, tokens_used = _query_graph(graph, query, token_budget)
    result = {
        "subgraph_markdown": markdown,
        "selected_node_count": node_count,
        "tokens_used": tokens_used,
    }
    set_cached_query(graph_path, query, token_budget, result)
    return result


@mcp.tool()
def compress_context_tool(
    context: str,
    question: str,
    target_tokens: int = 500,
    rate: float | None = None,
) -> dict:
    """Compress text context using LLMLingua with a local model."""
    logger.info("compress_context: %d chars -> %d tokens target", len(context), target_tokens)

    # Check cache
    cached = get_cached_compression(context, question, target_tokens)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        result = _compress(context=context, question=question, target_tokens=target_tokens, rate=rate)
        set_cached_compression(context, question, target_tokens, result)
        return result
    except Exception as e:
        return {"error": f"Compression failed: {e}"}


@mcp.tool()
def optimize_context_tool(
    codebase_path: str,
    query: str,
    token_budget: int = 500,
) -> dict:
    """Full pipeline: parse → query → compress in one call."""
    logger.info("optimize_context: %s (budget=%d)", codebase_path, token_budget)

    resolved = Path(codebase_path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {codebase_path}"}

    # Stage 1: Parse (with cache)
    graph = get_cached_graph(str(resolved))
    if graph is not None:
        graph_stats = {"total_nodes": graph.metadata.total_nodes,
                       "total_edges": graph.metadata.total_edges, "cached": True}
    else:
        try:
            graph = parse_codebase(str(resolved))
            if SCRUB_ENABLED:
                scrub_graph_content(graph)
            save_graph(graph, str(resolved / "graph.json"))
            graph_stats = {"total_nodes": graph.metadata.total_nodes,
                           "total_edges": graph.metadata.total_edges}
            set_cached_graph(str(resolved), graph)
        except Exception as e:
            return {"error": f"Parse failed: {e}"}

    # Stage 2: Query
    slurp_budget = max(token_budget * 8, 4000)
    try:
        subgraph_md, node_count, tokens_used = _query_graph(graph, query, slurp_budget)
        slurp_stats = {"selected_nodes": node_count, "tokens_used": tokens_used, "budget": slurp_budget}
    except Exception as e:
        return {"error": f"Query failed: {e}", "graph_stats": graph_stats}

    # Stage 3: Compress
    try:
        compression = _compress(context=subgraph_md, question=query, target_tokens=token_budget)
        compression_stats = {
            "original_tokens": compression["original_tokens"],
            "compressed_tokens": compression["compressed_tokens"],
            "compression_ratio": compression["compression_ratio"],
        }
    except Exception as e:
        return {"error": f"Compression failed: {e}", "graph_stats": graph_stats,
                "slurp_stats": slurp_stats, "optimized_context": subgraph_md}

    return {
        "optimized_context": compression["compressed"],
        "graph_stats": graph_stats,
        "slurp_stats": slurp_stats,
        "compression_stats": compression_stats,
    }


# ── v0.2.0: Streaming ────────────────────────────────────────────────────────

@mcp.tool()
async def optimize_context_stream(
    codebase_path: str,
    query: str,
    token_budget: int = 500,
) -> list[dict]:
    """Streaming pipeline: yields stage-by-stage progress events.

    Returns a list of event dicts in order:
      [{"stage": "parsing", "status": "started"}, ...]
    """
    events: list[dict] = []
    resolved = Path(codebase_path).resolve()
    if not resolved.is_dir():
        return [{"error": f"Not a directory: {codebase_path}"}]

    # Stage 1: Parse (streaming)
    events.append({"stage": "parsing", "status": "started"})
    graph = get_cached_graph(str(resolved))
    if graph is not None:
        events.append({"stage": "parsing", "status": "done", "cached": True,
                        "total_nodes": graph.metadata.total_nodes})
    else:
        try:
            all_events = []
            graph = None
            async for event in parse_codebase_stream(str(resolved)):
                all_events.append(event)
                if "graph" in event:
                    graph = event.pop("graph")
            if graph is None:
                events.append({"stage": "parsing", "status": "error", "error": "No graph produced"})
                return events
            if SCRUB_ENABLED:
                scrub_graph_content(graph)
            save_graph(graph, str(resolved / "graph.json"))
            set_cached_graph(str(resolved), graph)
            events.append({"stage": "parsing", "status": "done",
                           "total_nodes": len(graph.nodes), "total_edges": len(graph.edges)})
        except Exception as e:
            events.append({"stage": "parsing", "status": "error", "error": str(e)})
            return events

    # Stage 2: Query
    events.append({"stage": "query", "status": "started"})
    slurp_budget = max(token_budget * 8, 4000)
    try:
        subgraph_md, node_count, tokens_used = _query_graph(graph, query, slurp_budget)
        events.append({"stage": "query", "status": "done",
                       "selected_nodes": node_count, "subgraph_tokens": tokens_used})
    except Exception as e:
        events.append({"stage": "query", "status": "error", "error": str(e)})
        return events

    # Stage 3: Compress
    events.append({"stage": "compress", "status": "started"})
    try:
        compression = _compress(context=subgraph_md, question=query, target_tokens=token_budget)
        events.append({"stage": "compress", "status": "done",
                       "optimized_context": compression["compressed"],
                       "compressed_tokens": compression["compressed_tokens"]})
    except Exception as e:
        events.append({"stage": "compress", "status": "error", "error": str(e)})

    return events


# ── v0.2.0: Multi-Query Batching ──────────────────────────────────────────────

@mcp.tool()
def optimize_context_batch(
    codebase_path: str,
    queries: list[str],
    token_budget: int = 500,
) -> dict:
    """One parse, many queries. Builds graph once, runs slurp+compress per query."""
    logger.info("optimize_context_batch: %s (%d queries)", codebase_path, len(queries))

    resolved = Path(codebase_path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {codebase_path}"}
    if not queries:
        return {"error": "No queries provided"}

    # Parse once (with cache)
    graph = get_cached_graph(str(resolved))
    if graph is None:
        try:
            graph = parse_codebase(str(resolved))
            if SCRUB_ENABLED:
                scrub_graph_content(graph)
            save_graph(graph, str(resolved / "graph.json"))
            set_cached_graph(str(resolved), graph)
        except Exception as e:
            return {"error": f"Parse failed: {e}"}

    graph_stats = {"total_nodes": graph.metadata.total_nodes,
                   "total_edges": graph.metadata.total_edges}

    # Per-query pipeline
    results: list[dict] = []
    for q in queries:
        slurp_budget = max(token_budget * 8, 4000)
        try:
            subgraph_md, _node_count, _tokens_used = _query_graph(graph, q, slurp_budget)
            compression = _compress(context=subgraph_md, question=q, target_tokens=token_budget)
            results.append({
                "query": q,
                "optimized_context": compression["compressed"],
                "compressed_tokens": compression["compressed_tokens"],
                "selected_nodes": _node_count,
            })
        except Exception as e:
            results.append({"query": q, "error": str(e)})

    return {"graph_stats": graph_stats, "results": results}


# ── v0.2.0: Structured Output ────────────────────────────────────────────────

@mcp.tool()
def optimize_context_structured(
    codebase_path: str,
    query: str,
    token_budget: int = 500,
    output_format: str = "markdown",
) -> dict:
    """Full pipeline with structured JSON output mode.

    Args:
        output_format: "markdown" (default) or "json" for structured output.
    """
    logger.info("optimize_context_structured: %s (format=%s)", codebase_path, output_format)

    resolved = Path(codebase_path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {codebase_path}"}

    # Parse
    graph = get_cached_graph(str(resolved))
    if graph is None:
        try:
            graph = parse_codebase(str(resolved))
            if SCRUB_ENABLED:
                scrub_graph_content(graph)
            save_graph(graph, str(resolved / "graph.json"))
            set_cached_graph(str(resolved), graph)
        except Exception as e:
            return {"error": f"Parse failed: {e}"}

    slurp_budget = max(token_budget * 8, 4000)
    try:
        subgraph_md, _node_count, _tokens_used = _query_graph(graph, query, slurp_budget)
    except Exception as e:
        return {"error": f"Query failed: {e}"}

    try:
        compression = _compress(context=subgraph_md, question=query, target_tokens=token_budget)
    except Exception as e:
        return {"error": f"Compression failed: {e}"}

    if output_format == "json":
        # Build structured supporting nodes
        supporting_nodes = []
        for n in graph.nodes:
            if n.content and any(
                kw in n.content.lower() or kw in n.name.lower()
                for kw in query.lower().split()
            ):
                supporting_nodes.append({
                    "id": n.id,
                    "name": n.name,
                    "type": n.type.value,
                    "file": n.file,
                    "lines": f"{n.line_start}-{n.line_end}",
                    "content_snippet": n.content[:200],
                })

        return {
            "query": query,
            "answer": compression["compressed"],
            "supporting_nodes": supporting_nodes[:20],
            "graph_stats": {
                "total_nodes": graph.metadata.total_nodes,
                "total_edges": graph.metadata.total_edges,
            },
            "compression_stats": {
                "original_tokens": compression["original_tokens"],
                "compressed_tokens": compression["compressed_tokens"],
                "compression_ratio": compression["compression_ratio"],
            },
        }

    # Default: markdown
    return {
        "optimized_context": compression["compressed"],
        "graph_stats": {"total_nodes": graph.metadata.total_nodes,
                        "total_edges": graph.metadata.total_edges},
        "compression_stats": {
            "original_tokens": compression["original_tokens"],
            "compressed_tokens": compression["compressed_tokens"],
            "compression_ratio": compression["compression_ratio"],
        },
    }


# ── v0.2.0: Chunked Parsing ──────────────────────────────────────────────────

@mcp.tool()
async def parse_codebase_stream_tool(
    path: str,
    batch_size: int = 50,
) -> list[dict]:
    """Stream parse progress. Parses in batches, returns progress events."""
    logger.info("parse_codebase_stream: %s (batch=%d)", path, batch_size)
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        return [{"error": f"Not a directory: {path}"}]

    events: list[dict] = []
    graph = None
    try:
        async for event in parse_codebase_stream(str(resolved), batch_size):
            events.append(event)
            if "graph" in event:
                graph = event.pop("graph")
    except Exception as e:
        events.append({"error": str(e)})
        return events

    if graph:
        if SCRUB_ENABLED:
            scrub_graph_content(graph)
        output_path = str(resolved / "graph.json")
        save_graph(graph, output_path)
        set_cached_graph(str(resolved), graph)
        events.append({"status": "saved", "graph_json_path": output_path})

    return events


# ── v0.2.0: Watch Mode ───────────────────────────────────────────────────────

@mcp.tool()
def watch_start_tool(path: str) -> dict:
    """Start watching a codebase for incremental graph updates."""
    logger.info("watch_start: %s", path)
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {path}"}

    graph = get_cached_graph(str(resolved))
    if graph is None:
        graph = parse_codebase(str(resolved))
        if SCRUB_ENABLED:
            scrub_graph_content(graph)
        save_graph(graph, str(resolved / "graph.json"))
        set_cached_graph(str(resolved), graph)

    return start_watch(str(resolved), graph)


@mcp.tool()
def watch_stop_tool(path: str) -> dict:
    """Stop watching a codebase."""
    return stop_watch(path)


@mcp.tool()
def get_graph_stats_tool(graph_path: str) -> dict:
    """Return metadata and stats for a graph."""
    if not os.path.isfile(graph_path):
        return {"error": f"Graph file not found: {graph_path}"}
    try:
        graph = load_graph(graph_path)
    except Exception as e:
        return {"error": f"Failed to load graph: {e}"}

    node_types = {}
    for n in graph.nodes:
        node_types[n.type.value] = node_types.get(n.type.value, 0) + 1

    edge_types = {}
    for e in graph.edges:
        edge_types[e.type.value] = edge_types.get(e.type.value, 0) + 1

    return {
        "root": graph.metadata.root,
        "total_nodes": graph.metadata.total_nodes,
        "total_edges": graph.metadata.total_edges,
        "node_types": node_types,
        "edge_types": edge_types,
        "generated_at": graph.metadata.generated_at,
    }


@mcp.tool()
def reset_graph_tool(path: str) -> dict:
    """Force a full re-parse of the codebase, ignoring cache."""
    logger.info("reset_graph: %s", path)
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {path}"}

    graph = parse_codebase(str(resolved))
    if SCRUB_ENABLED:
        scrub_graph_content(graph)
    output_path = str(resolved / "graph.json")
    save_graph(graph, output_path)
    set_cached_graph(str(resolved), graph)

    return {
        "graph_json_path": output_path,
        "total_nodes": graph.metadata.total_nodes,
        "total_edges": graph.metadata.total_edges,
        "reset": True,
    }


# ── v0.2.0: Graph Diffing ────────────────────────────────────────────────────

@mcp.tool()
def diff_graph_tool(graph_path_a: str, graph_path_b: str) -> dict:
    """Compare two knowledge graphs and return structural differences."""
    logger.info("diff_graph: %s vs %s", graph_path_a, graph_path_b)

    for p in (graph_path_a, graph_path_b):
        if not os.path.isfile(p):
            return {"error": f"Graph file not found: {p}"}

    try:
        graph_a = load_graph(graph_path_a)
        graph_b = load_graph(graph_path_b)
    except Exception as e:
        return {"error": f"Failed to load graph: {e}"}

    return diff_graphs(graph_a, graph_b)


# ── v0.2.0: PII Scrubbing ────────────────────────────────────────────────────

@mcp.tool()
def scrub_text_tool(text: str, custom_patterns: dict | None = None) -> dict:
    """Scrub PII (secrets, emails, API keys, etc.) from text."""
    return _scrub_text(text, custom_patterns)


@mcp.tool()
def scrub_file_tool(path: str, output_path: str | None = None) -> dict:
    """Scrub PII from a file."""
    return _scrub_file(path, output_path)


@mcp.tool()
def list_pii_patterns_tool() -> dict:
    """List all registered PII detection patterns."""
    return _list_pii_patterns()


@mcp.tool()
def add_pii_pattern_tool(name: str, regex: str) -> dict:
    """Add or override a PII detection pattern."""
    return _add_pii_pattern(name, regex)


# ── v0.2.0: Cache Management ─────────────────────────────────────────────────

@mcp.tool()
def clear_cache_tool(scope: str = "all") -> dict:
    """Clear the cache. Scope: 'all' | 'graph' | 'query' | 'compression'."""
    return _clear_cache(scope)


@mcp.tool()
def cache_stats_tool() -> dict:
    """Return cache hit/miss statistics and size."""
    return _cache_stats()


# ── v0.2.0: Web UI ───────────────────────────────────────────────────────────

@mcp.tool()
def launch_ui_tool(port: int = 8765, open_browser: bool = True) -> dict:
    """Launch the web visualization UI for graph exploration."""
    from .ui import launch_ui
    return launch_ui(port=port, open_browser=open_browser)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Entry point for `python -m context_optimizer`."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
