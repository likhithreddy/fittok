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
import threading
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from .cache import (
    CACHE_DIR,
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
from .watcher import start_watch, stop_watch, get_watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "fittok",
    instructions=(
        "Retrieves the most relevant REAL source code for a question about a "
        "codebase, within a token budget. Prefer `optimize_context` for "
        "'how does X work / where is Y' questions instead of reading files or "
        "grepping. For MULTI-ASPECT questions make AT MOST 2–3 focused calls "
        "total (one per facet), then synthesize and answer — do not re-query "
        "the same file at a higher budget. Every line in the output is numbered "
        "with its real source line, so answer directly from `optimized_context` "
        "and cite file:line from the numbers shown; do NOT separately Read (or "
        "`nl`/`sed`) the files it came from — their bodies are already complete "
        "and line-anchored here, so re-reading only adds cost."
    ),
)

SCRUB_ENABLED = os.environ.get("FITTOK_SCRUB", "false").lower() in ("true", "1", "yes")
# When on, the tool asks the model to surface a one-line savings footer to the
# user. Default on — set FITTOK_SHOW_SAVINGS=false to disable.
SHOW_SAVINGS = os.environ.get("FITTOK_SHOW_SAVINGS", "true").lower() not in ("false", "0", "no")
# Auto-start the file watcher for any codebase fittok parses/queries, so graph
# updates are incremental (only changed files re-parsed & merged, not a full
# re-parse). Default on — set FITTOK_AUTOWATCH=false to disable (e.g. where
# background file watching is undesirable).
AUTOWATCH_ENABLED = os.environ.get("FITTOK_AUTOWATCH", "true").lower() in ("true", "1", "yes")


def _graph_output_path(resolved: Path) -> str:
    """On-disk graph.json path. Delegates to ``cache.graph_output_path`` (the
    single implementation, shared with the watcher) so the two never diverge."""
    from .cache import graph_output_path
    return graph_output_path(resolved)


def _live_graph(resolved: Path):
    """Return the watcher's live, incrementally-updated graph for this path
    (flushing any pending edits first), or None if no active watcher.

    When a watcher is active, queries use this live graph directly instead of
    the mtime-keyed cache — so an edit does NOT trigger a full re-parse; the
    watcher has already merged only the changed files.
    """
    if not AUTOWATCH_ENABLED:
        return None
    state = get_watcher(str(resolved))
    if state is None or state.observer is None or not state.observer.is_alive():
        return None
    state.flush()
    return state.graph


def _ensure_watching(root_path: str, graph) -> None:
    """Best-effort auto-start the watcher for a codebase so future edits update
    the graph incrementally. Idempotent (no-op if already watching or disabled).
    """
    if not AUTOWATCH_ENABLED:
        return
    try:
        if get_watcher(root_path) is None:
            start_watch(root_path, graph)
    except Exception as exc:
        logger.warning("Auto-watch failed for %s: %s", root_path, exc)


def _resolve_graph(resolved: Path):
    """Acquire the current graph for a codebase via ONE path used by every tool:
    watcher live graph → mtime-keyed cache → full parse. Auto-starts the watcher
    so future edits update incrementally. Returns (graph, graph_stats) and lets
    parse errors propagate so each caller can format them.
    """
    graph = _live_graph(resolved)
    if graph is not None:
        return graph, {"total_nodes": graph.metadata.total_nodes,
                       "total_edges": graph.metadata.total_edges, "watched": True}
    graph = get_cached_graph(str(resolved))
    if graph is None:
        graph = parse_codebase(str(resolved))
        if SCRUB_ENABLED:
            scrub_graph_content(graph)
        save_graph(graph, _graph_output_path(resolved))
        set_cached_graph(str(resolved), graph)
    _ensure_watching(str(resolved), graph)
    return graph, {"total_nodes": graph.metadata.total_nodes,
                   "total_edges": graph.metadata.total_edges, "cached": True}


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
        output_path = _graph_output_path(resolved)
        save_graph(cached, output_path)
        _ensure_watching(str(resolved), cached)
        return {
            "graph_json_path": output_path,
            "total_nodes": cached.metadata.total_nodes,
            "total_edges": cached.metadata.total_edges,
            "cached": True,
        }

    graph = parse_codebase(str(resolved))
    if SCRUB_ENABLED:
        scrub_graph_content(graph)
    output_path = _graph_output_path(resolved)
    graph_json_path = save_graph(graph, output_path)
    set_cached_graph(str(resolved), graph)
    _ensure_watching(str(resolved), graph)

    return {
        "graph_json_path": graph_json_path,
        "total_nodes": graph.metadata.total_nodes,
        "total_edges": graph.metadata.total_edges,
    }


@mcp.tool()
def query_graph_tool(
    graph_path: str = "",
    query: str = "",
    token_budget: int = 4000,
    codebase_path: str = "",
) -> dict:
    """Query a knowledge graph for the most relevant subgraph within a token budget.

    Prefer ``codebase_path`` (a directory) — it uses the live, incrementally
    updated graph. ``graph_path`` (a saved graph.json) is kept as a fallback.
    """
    logger.info("query_graph: codebase=%r graph=%r (query=%r, budget=%d)",
                codebase_path, graph_path, query[:80], token_budget)

    graph = None
    version = ""
    cache_key = ""
    if codebase_path:
        resolved = Path(codebase_path).resolve()
        if not resolved.is_dir():
            return {"error": f"Not a directory: {codebase_path}"}
        try:
            graph, _gs = _resolve_graph(resolved)
        except Exception as e:
            return {"error": f"Parse failed: {e}"}
        version = graph.metadata.generated_at
        cache_key = f"live:{resolved}"
    else:
        if not graph_path or not os.path.isfile(graph_path):
            return {"error": f"Graph file not found: {graph_path}"}
        try:
            graph = load_graph(graph_path)
        except Exception as e:
            return {"error": f"Failed to load graph: {e}"}
        try:
            version = str(os.path.getmtime(graph_path))
        except OSError:
            version = ""
        cache_key = graph_path

    # Versioned by graph revision so stale subgraphs aren't returned after edits.
    cached = get_cached_query(cache_key, query, token_budget, graph_version=version)
    if cached is not None:
        return {**cached, "cached": True}

    markdown, node_count, tokens_used = _query_graph(graph, query, token_budget)
    result = {
        "subgraph_markdown": markdown,
        "selected_node_count": node_count,
        "tokens_used": tokens_used,
    }
    set_cached_query(cache_key, query, token_budget, result, graph_version=version)
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
    cached = get_cached_compression(context, question, target_tokens, rate=rate)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        result = _compress(context=context, question=question, target_tokens=target_tokens, rate=rate)
        set_cached_compression(context, question, target_tokens, result, rate=rate)
        return result
    except Exception as e:
        return {"error": f"Compression failed: {e}"}


@mcp.tool()
def optimize_context_tool(
    codebase_path: str,
    query: str,
    token_budget: int = 0,  # 0 = adaptive (auto-sized from relevance)
) -> dict:
    """Return the most relevant REAL source code for a question, within a token budget.

    Use this FIRST for any "how does X work / where is Y" question about the
    codebase, and prefer it over reading files or grepping.

    For a MULTI-ASPECT question, make AT MOST 2–3 focused calls TOTAL (one per
    facet), then synthesize and answer — do NOT fan out further. Do NOT re-query
    the same file at a higher token_budget: if a function body is missing, make
    ONE more call that names the function explicitly, then stop. Re-fetching what
    you already have discards the savings this tool exists to provide.

    The output contains ACTUAL source code with every line numbered by its real
    source line — answer directly from it and cite file:line from the numbers
    shown. Do NOT use the Read tool (or `nl`/`sed`) for any file that appears in
    the output: the bodies are already complete and line-anchored, so reading
    them again adds nothing but cost.
    """
    logger.info("optimize_context: %s (budget=%d)", codebase_path, token_budget)

    resolved = Path(codebase_path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {codebase_path}"}

    # Stage 1: Parse — one shared resolver (live watcher graph → cache → parse).
    try:
        graph, graph_stats = _resolve_graph(resolved)
    except Exception as e:
        return {"error": f"Parse failed: {e}"}

    # Stage 2: Select readable relevant code within the budget.
    # Selection IS the compression — we return real source (trimmed to budget),
    # not LLMLingua-compressed text, so the model can answer from it directly
    # instead of re-reading files (which previously negated the token savings).
    try:
        readable, ctx_stats, files = _select_readable_context(graph, query, token_budget, codebase_key=str(resolved))
    except Exception as e:
        return {"error": f"Query failed: {e}", "graph_stats": graph_stats}

    savings = _compute_savings(resolved, files, ctx_stats["tokens_sent"])

    # Optionally instruct the model to surface the savings to the user inline.
    # Worded as POTENTIAL — fittok can't see whether the client re-reads the
    # files afterward, so claiming the saving unconditionally would overstate it
    # in exactly the failure case this tool exists to prevent.
    if SHOW_SAVINGS and savings.get("baseline_full_files_tokens"):
        readable += (
            "\n\n---\n[fittok] After answering, append this footer on its own "
            f"line:\n🪙 fittok: up to {savings['reduction_pct']}% smaller "
            f"(~{savings['baseline_full_files_tokens']} → "
            f"{savings['tokens_sent_with_mcp']} tokens) — answer from this, "
            f"don't re-read.\n"
        )

    return {
        "optimized_context": readable,
        "graph_stats": graph_stats,
        "slurp_stats": ctx_stats,
        "savings": savings,
    }


# Per-codebase memory of recently-returned node IDs, so fanned-out / repeated
# calls don't re-send the same code. Capped FIFO; rotates naturally.
_RECENT_RETURNED: dict[str, list] = {}
_RECENT_CAP = 250
_RECENT_LOCK = threading.Lock()


def _record_returned(key: str, ids: list) -> None:
    # FastMCP can dispatch tool calls concurrently (batch/stream); guard the FIFO
    # so concurrent optimize_context calls don't drop entries or corrupt it.
    with _RECENT_LOCK:
        lst = _RECENT_RETURNED.setdefault(key, [])
        seen = set(lst)
        for i in ids:
            if i not in seen:
                lst.append(i)
                seen.add(i)
        if len(lst) > _RECENT_CAP:
            del lst[: len(lst) - _RECENT_CAP]


# Prepended to every optimized_context so the "answer from this, don't re-read"
# instruction lands right above the code — at the moment of action, not buried in
# a tool description the client may not surface. This is the in-output lever that
# replaces the (ineffective, repo-polluting) INSTRUCTIONS.md idea.
_AUTHORITY_NOTE = (
    "> **This is the complete, real source for your question.** Every line below "
    "is numbered with its actual source line, so cite `file:line` directly from "
    "the numbers shown. Answer from the code below — re-opening these files "
    "(Read, grep, or `nl`/`sed`) only discards the token savings: the function "
    "bodies are already whole and line-anchored here, so reading them again "
    "adds nothing but cost.\n\n"
)


def _select_readable_context(graph, query: str, token_budget: int, codebase_key: str | None = None):
    """Select the most relevant *real* source within the token budget.

    Returns (readable_markdown, stats, files). Unlike the old pipeline this
    returns actual code trimmed to the budget — not lossy LLMLingua output — so
    it stays usable and the model doesn't fall back to reading whole files.
    When codebase_key is given, nodes returned by recent calls are excluded so
    repeated/fanned-out queries don't re-send the same code.
    """
    from .tokens import count_tokens
    if codebase_key:
        # Copy under the lock — a concurrent _record_returned can append/trim the
        # list mid-iteration (RuntimeError) or corrupt the fan-out dedup.
        with _RECENT_LOCK:
            exclude = set(_RECENT_RETURNED.get(codebase_key, []))
    else:
        exclude = None

    q = _query_graph(graph, query, token_budget, with_diagnostics=True, exclude_ids=exclude)
    if codebase_key:
        # Dedup BOTH query-selected nodes and surfaced dependencies, so a
        # repeated/fanned-out query never re-sends code already returned.
        _record_returned(codebase_key, q.get("selected_ids", []) + q.get("neighbor_ids", []))
    eff_budget = q.get("budget", token_budget) or 0
    readable = q["markdown"]
    # NOTE: no mid-body truncation here. slurp.format_subgraph already enforces
    # the budget at NODE boundaries (whole functions only) and the referenced-
    # dependency lines are whole — so q["markdown"] is node-boundary-clean and
    # within budget. A truncate_to_tokens call here would slice the last
    # function in half, which is exactly what forced the model to re-open the
    # file (via `nl -ba | sed`) to see the rest of the cut-off body.
    # Prepend the authority note + codebase map + flow trace. MAX_BUDGET (1600)
    # is sized to leave room for these so the TOTAL output stays under Copilot's
    # ~10 KB MCP truncation wall (copilot-cli#1732) — over it, the host chops
    # the tail and the model sees "elided" bodies → escalation + re-reads.
    from .slurp import generate_codebase_map, generate_flow_trace
    codebase_map = generate_codebase_map(graph)
    # Flow trace = the complete call chain among the selected nodes. Gives the
    # model verifiable completeness for flow questions ("how does X work") so it
    # stops re-reading files fittok already returned — the failure mode where it
    # couldn't tell the slice was complete.
    # Preserve relevance order (selected in score order, then neighbors) so the
    # flow trace leads with the query-relevant entry points, not graph order.
    _order = q.get("selected_ids", []) + q.get("neighbor_ids", [])
    _by_id = {n.id: n for n in graph.nodes}
    sel_nodes = [_by_id[i] for i in _order if i in _by_id]
    flow_trace = generate_flow_trace(sel_nodes, graph.edges)
    readable = _AUTHORITY_NOTE + codebase_map + flow_trace + "\n\n---\n\n" + readable
    tokens_sent = count_tokens(readable)
    stats = {
        "selected_nodes": q["selected_nodes"],
        "neighbor_nodes": q.get("neighbor_nodes", 0),
        "tokens_sent": tokens_sent,
        "budget": eff_budget,
        "budget_mode": q.get("budget_mode", "explicit"),
        "method": q["method"],
        "confidence": q["confidence"],
        "confidence_label": q["confidence_label"],
        "mode": "readable",
    }
    return readable, stats, q.get("files", [])


def _compute_savings(root: Path, rel_files: list[str], tokens_sent: int) -> dict:
    """Estimate tokens saved vs. the no-MCP baseline (reading the whole files).

    Baseline = total tokens of the full source files the selected context came
    from — a realistic stand-in for "without the MCP, the LLM reads these files."
    """
    from .tokens import count_tokens
    baseline = 0
    counted = 0
    for rel in rel_files:
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        baseline += count_tokens(text)
        counted += 1
    saved = max(baseline - tokens_sent, 0)
    pct = round(100.0 * saved / baseline, 1) if baseline else 0.0
    return {
        "tokens_sent_with_mcp": tokens_sent,
        "baseline_full_files_tokens": baseline,
        "files_counted": counted,
        "tokens_saved": saved,
        "reduction_pct": pct,
        # POTENTIAL, not achieved: fittok can't observe whether the client
        # re-reads the files afterward, so the saving is only realized if the
        # model answers from this context without re-reading. Reporting it as
        # unconditional overstates reality precisely in the failure case.
        "potential": True,
        "summary": (
            f"Up to ~{pct}% smaller than reading the {counted} relevant file(s) "
            f"in full (~{baseline} → {tokens_sent} tokens). Realized only if the "
            f"model answers without re-reading — fittok can't observe that."
            if baseline else "No baseline files available to compare."
        ),
    }


# ── v0.2.0: Streaming ────────────────────────────────────────────────────────

@mcp.tool()
async def optimize_context_stream(
    codebase_path: str,
    query: str,
    token_budget: int = 0,  # 0 = adaptive (auto-sized from relevance)
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
    graph = _live_graph(resolved) or get_cached_graph(str(resolved))
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
            save_graph(graph, _graph_output_path(resolved))
            set_cached_graph(str(resolved), graph)
            events.append({"stage": "parsing", "status": "done",
                           "total_nodes": len(graph.nodes), "total_edges": len(graph.edges)})
        except Exception as e:
            events.append({"stage": "parsing", "status": "error", "error": str(e)})
            return events

    _ensure_watching(str(resolved), graph)

    # Stage 2: Select readable relevant code within budget (no lossy compression)
    events.append({"stage": "select", "status": "started"})
    try:
        readable, ctx_stats, _files = _select_readable_context(graph, query, token_budget, codebase_key=str(resolved))
        events.append({"stage": "select", "status": "done",
                       "optimized_context": readable,
                       "tokens_sent": ctx_stats["tokens_sent"],
                       "selected_nodes": ctx_stats["selected_nodes"],
                       "confidence": ctx_stats["confidence"]})
    except Exception as e:
        events.append({"stage": "select", "status": "error", "error": str(e)})

    return events


# ── v0.2.0: Multi-Query Batching ──────────────────────────────────────────────

@mcp.tool()
def optimize_context_batch(
    codebase_path: str,
    queries: list[str],
    token_budget: int = 0,  # 0 = adaptive (auto-sized from relevance)
) -> dict:
    """One parse, many queries. Builds graph once, runs slurp+compress per query."""
    logger.info("optimize_context_batch: %s (%d queries)", codebase_path, len(queries))

    resolved = Path(codebase_path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {codebase_path}"}
    if not queries:
        return {"error": "No queries provided"}

    # Parse once — shared resolver (live watcher graph → cache → parse).
    try:
        graph, _gs = _resolve_graph(resolved)
    except Exception as e:
        return {"error": f"Parse failed: {e}"}

    graph_stats = {"total_nodes": graph.metadata.total_nodes,
                   "total_edges": graph.metadata.total_edges}

    # Per-query pipeline — readable selected code within budget (no lossy compression)
    results: list[dict] = []
    for q in queries:
        try:
            readable, ctx_stats, _files = _select_readable_context(graph, q, token_budget, codebase_key=str(resolved))
            results.append({
                "query": q,
                "optimized_context": readable,
                "tokens_sent": ctx_stats["tokens_sent"],
                "selected_nodes": ctx_stats["selected_nodes"],
                "confidence": ctx_stats["confidence"],
            })
        except Exception as e:
            results.append({"query": q, "error": str(e)})

    return {"graph_stats": graph_stats, "results": results}


# ── v0.2.0: Structured Output ────────────────────────────────────────────────

@mcp.tool()
def optimize_context_structured(
    codebase_path: str,
    query: str,
    token_budget: int = 0,  # 0 = adaptive (auto-sized from relevance)
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

    # Parse — shared resolver (live watcher graph → cache → parse).
    try:
        graph, _gs = _resolve_graph(resolved)
    except Exception as e:
        return {"error": f"Parse failed: {e}"}

    try:
        readable, ctx_stats, _files = _select_readable_context(graph, query, token_budget, codebase_key=str(resolved))
    except Exception as e:
        return {"error": f"Query failed: {e}"}

    if output_format == "json":
        # Use slurp's PageRank+TF-IDF scoring for genuinely relevant supporting nodes
        from .models import NodeType
        from .slurp import _compute_combined_scores

        content_nodes = [n for n in graph.nodes if n.type != NodeType.FILE]
        if content_nodes:
            from . import embeddings as _embeddings
            # Pass semantic scores through so supporting_nodes ranking agrees
            # with the semantic-ranked `readable` answer returned alongside.
            scores = _compute_combined_scores(
                content_nodes, graph.edges, query,
                semantic=_embeddings.semantic_scores(content_nodes, query))
            # Pick top-N nodes by score (up to 20)
            ranked_nodes = sorted(content_nodes, key=lambda n: scores.get(n.id, 0), reverse=True)
        else:
            ranked_nodes = []

        supporting_nodes = []
        for n in ranked_nodes[:20]:
            score = scores.get(n.id, 0)
            if score <= 0:
                continue
            supporting_nodes.append({
                "id": n.id,
                "name": n.name,
                "type": n.type.value,
                "file": n.file,
                "lines": f"{n.line_start}-{n.line_end}",
                "relevance_score": round(score, 4),
                "content_snippet": n.content[:200],
            })

        return {
            "query": query,
            "answer": readable,
            "supporting_nodes": supporting_nodes[:20],
            "graph_stats": {
                "total_nodes": graph.metadata.total_nodes,
                "total_edges": graph.metadata.total_edges,
            },
            "slurp_stats": ctx_stats,
        }

    # Default: markdown
    return {
        "optimized_context": readable,
        "graph_stats": {"total_nodes": graph.metadata.total_nodes,
                        "total_edges": graph.metadata.total_edges},
        "slurp_stats": ctx_stats,
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
        output_path = _graph_output_path(resolved)
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
        save_graph(graph, _graph_output_path(resolved))
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
    output_path = _graph_output_path(resolved)
    save_graph(graph, output_path)
    set_cached_graph(str(resolved), graph)

    # If a watcher is running, swap its live graph too — otherwise the tools that
    # prefer the live graph (optimize_context / show_graph) keep serving the
    # pre-reset graph, silently breaking "ignore cache" under autowatch.
    state = get_watcher(str(resolved))
    if state is not None:
        state.replace_graph(graph)

    return {
        "graph_json_path": output_path,
        "total_nodes": graph.metadata.total_nodes,
        "total_edges": graph.metadata.total_edges,
        "reset": True,
    }


@mcp.tool()
def show_graph_tool(path: str, query: str = "") -> dict:
    """Open an INTERACTIVE BROWSER visualization of the codebase knowledge graph.

    ⚠️ Call this ONLY when the user EXPLICITLY asks to SEE / VISUALIZE / SHOW /
    DRAW the graph — e.g. "show me the knowledge graph", "visualize this
    codebase", "draw the call graph". Do NOT call it for ordinary codebase
    questions; use `optimize_context` for those. This opens a browser tab on the
    user's machine (the server runs locally) and returns a short confirmation —
    it does NOT return source code to answer from. Optional `query` highlights
    the most relevant nodes.
    """
    import webbrowser
    from .graph_viz import build_graph_html

    logger.info("show_graph: %s (query=%r)", path, query[:80] if query else "")

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        return {"error": f"Not a directory: {path}"}

    # Prefer the watcher's live graph; else cache; else full parse (+ auto-watch).
    graph = _live_graph(resolved)
    src = "watched"
    if graph is None:
        graph = get_cached_graph(str(resolved))
        if graph is not None:
            src = "cached"
        else:
            try:
                graph = parse_codebase(str(resolved))
                if SCRUB_ENABLED:
                    scrub_graph_content(graph)
                save_graph(graph, _graph_output_path(resolved))
                set_cached_graph(str(resolved), graph)
                src = "parsed"
            except Exception as e:
                return {"error": f"Parse failed: {e}"}
        _ensure_watching(str(resolved), graph)

    highlight_ids: set[str] = set()
    if query:
        from .slurp import score_nodes
        scores = score_nodes(graph, query)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        cutoff = max(10, len(sorted_scores) // 20)
        highlight_ids = {nid for nid, _ in sorted_scores[:cutoff]}

    try:
        out_path = build_graph_html(graph, str(resolved), highlight_ids, query or None)
    except ImportError:
        return {"error": "pyvis not installed. Install with: uv pip install 'fittok[ui]'"}
    except Exception as e:
        return {"error": f"Graph render failed: {e}"}

    # webbrowser.open returns False (without raising) on headless / remote /
    # sandboxed hosts with no browser — honor the bool so we don't mis-report
    # success. The path is always returned so the user can open it manually.
    try:
        opened = bool(webbrowser.open(f"file://{out_path}"))
    except Exception:
        opened = False

    note = ("Interactive graph opened in your browser."
            if opened else
            "Could not open a browser (headless/remote?). Open the graph HTML path below manually.")
    if query:
        note += f" {len(highlight_ids)} nodes highlighted for the query."
    return {
        "opened_in_browser": opened,
        "graph_html_path": out_path,
        "total_nodes": graph.metadata.total_nodes,
        "total_edges": graph.metadata.total_edges,
        "highlighted_nodes": len(highlight_ids),
        "graph_source": src,
        "note": note,
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
    """Entry point for `python -m fittok`."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
