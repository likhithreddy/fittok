"""MCP server for context optimization.

Exposes four tools:
  - parse_codebase: Parse code into a knowledge graph
  - query_graph: Query the graph for relevant subgraph
  - compress_context: Compress text using LLMLingua
  - optimize_context: Full pipeline in one call
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .graphify import load_graph, parse_codebase, save_graph
from .llmlingua_wrapper import compress_context as _compress
from .slurp import query_graph as _query_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "context-optimizer",
    instructions="Filter and compress LLM context using graph-based analysis + LLMLingua",
)


# ── Tool: parse_codebase ──────────────────────────────────────────────────────

@mcp.tool()
def parse_codebase_tool(path: str) -> dict:
    """Recursively parse all code files in the given path and generate a knowledge graph.

    Args:
        path: Root directory of the codebase to parse.

    Returns:
        Dict with graph_json_path, total_nodes, total_edges.
    """
    logger.info("parse_codebase: %s", path)
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {path}"}

    graph = parse_codebase(str(resolved))
    output_path = str(resolved / "graph.json")
    graph_json_path = save_graph(graph, output_path)

    return {
        "graph_json_path": graph_json_path,
        "total_nodes": graph.metadata.total_nodes,
        "total_edges": graph.metadata.total_edges,
    }


# ── Tool: query_graph ─────────────────────────────────────────────────────────

@mcp.tool()
def query_graph_tool(
    graph_path: str,
    query: str,
    token_budget: int = 4000,
) -> dict:
    """Query a knowledge graph to select the most relevant subgraph within a token budget.

    Uses PageRank + TF-IDF to rank nodes by relevance to the query.

    Args:
        graph_path: Path to the graph.json file.
        query: Natural language query describing what you need.
        token_budget: Maximum tokens for the output subgraph (default 4000).

    Returns:
        Dict with subgraph_markdown, selected_node_count, tokens_used.
    """
    logger.info("query_graph: %s (query=%r, budget=%d)", graph_path, query[:80], token_budget)

    if not os.path.isfile(graph_path):
        return {"error": f"Graph file not found: {graph_path}"}

    try:
        graph = load_graph(graph_path)
    except Exception as e:
        return {"error": f"Failed to load graph: {e}"}

    markdown, node_count, tokens_used = _query_graph(graph, query, token_budget)

    return {
        "subgraph_markdown": markdown,
        "selected_node_count": node_count,
        "tokens_used": tokens_used,
    }


# ── Tool: compress_context ────────────────────────────────────────────────────

@mcp.tool()
def compress_context_tool(
    context: str,
    question: str,
    target_tokens: int = 500,
    rate: float | None = None,
) -> dict:
    """Compress text context using LLMLingua with a local model.

    Args:
        context: The text to compress.
        question: Guiding question to steer what to keep.
        target_tokens: Target output token count (default 500).
        rate: Optional compression ratio override (0.0-1.0).

    Returns:
        Dict with compressed text and compression stats.
    """
    logger.info("compress_context: %d chars -> %d tokens target", len(context), target_tokens)

    try:
        return _compress(
            context=context,
            question=question,
            target_tokens=target_tokens,
            rate=rate,
        )
    except Exception as e:
        return {"error": f"Compression failed: {e}"}


# ── Tool: optimize_context (composite pipeline) ───────────────────────────────

@mcp.tool()
def optimize_context_tool(
    codebase_path: str,
    query: str,
    token_budget: int = 500,
) -> dict:
    """Run the full optimization pipeline: parse → query → compress.

    Parses the codebase into a graph, queries for relevant subgraph,
    then compresses the result to the target token budget.

    Args:
        codebase_path: Root directory of the codebase.
        query: User question to guide context selection.
        token_budget: Final target token count (default 500).

    Returns:
        Dict with optimized_context and stats for each stage.
    """
    logger.info("optimize_context: %s (budget=%d)", codebase_path, token_budget)

    resolved = Path(codebase_path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {codebase_path}"}

    # Stage 1: Parse
    try:
        graph = parse_codebase(str(resolved))
        graph_path = save_graph(graph, str(resolved / "graph.json"))
        graph_stats = {
            "total_nodes": graph.metadata.total_nodes,
            "total_edges": graph.metadata.total_edges,
            "graph_path": graph_path,
        }
    except Exception as e:
        return {"error": f"Parse failed: {e}"}

    # Stage 2: Query (use a larger budget for the subgraph, then compress down)
    slurp_budget = max(token_budget * 8, 4000)
    try:
        subgraph_md, node_count, tokens_used = _query_graph(graph, query, slurp_budget)
        slurp_stats = {
            "selected_nodes": node_count,
            "tokens_used": tokens_used,
            "budget": slurp_budget,
        }
    except Exception as e:
        return {"error": f"Query failed: {e}", "graph_stats": graph_stats}

    # Stage 3: Compress
    try:
        compression = _compress(
            context=subgraph_md,
            question=query,
            target_tokens=token_budget,
        )
        compression_stats = {
            "original_tokens": compression["original_tokens"],
            "compressed_tokens": compression["compressed_tokens"],
            "compression_ratio": compression["compression_ratio"],
        }
    except Exception as e:
        return {
            "error": f"Compression failed: {e}",
            "graph_stats": graph_stats,
            "slurp_stats": slurp_stats,
            "optimized_context": subgraph_md,
        }

    return {
        "optimized_context": compression["compressed"],
        "graph_stats": graph_stats,
        "slurp_stats": slurp_stats,
        "compression_stats": compression_stats,
    }


# ── __main__ support ──────────────────────────────────────────────────────────

def main():
    """Entry point for `python -m context_optimizer.server`."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
