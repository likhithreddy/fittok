"""slurp — query a knowledge graph within a token budget.

Uses PageRank + TF-IDF to select the most relevant nodes for a given query,
then formats them as markdown within the specified token limit.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import KnowledgeGraph, NodeType
from .tokens import count_tokens

logger = logging.getLogger(__name__)


# ── Token counting ────────────────────────────────────────────────────────────


# ── PageRank ──────────────────────────────────────────────────────────────────

def _build_adjacency(nodes: list, edges: list) -> dict[str, set[str]]:
    """Build an undirected adjacency map from graph edges."""
    adj: dict[str, set[str]] = defaultdict(set)
    node_ids = {n.id for n in nodes}
    for edge in edges:
        if edge.source in node_ids and edge.target in node_ids:
            adj[edge.source].add(edge.target)
            adj[edge.target].add(edge.source)
    return adj


def pagerank(
    nodes: list,
    edges: list,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute PageRank scores for all nodes in the graph."""
    node_ids = [n.id for n in nodes]
    n = len(node_ids)
    if n == 0:
        return {}

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    adj = _build_adjacency(nodes, edges)

    # Initialize scores uniformly
    scores = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        prev = scores.copy()
        for i, nid in enumerate(node_ids):
            neighbors = adj.get(nid, set())
            rank_sum = 0.0
            for neighbor in neighbors:
                j = id_to_idx[neighbor]
                degree = len(adj.get(neighbor, set()))
                rank_sum += prev[j] / max(degree, 1)
            scores[i] = (1 - damping) / n + damping * rank_sum

        # Normalize
        total = scores.sum()
        if total > 0:
            scores /= total

        # Check convergence
        if np.linalg.norm(scores - prev, 1) < tol:
            break

    return {nid: float(scores[i]) for i, nid in enumerate(node_ids)}


# ── TF-IDF scoring ───────────────────────────────────────────────────────────

def tfidf_scores(
    nodes: list,
    query: str,
) -> dict[str, float]:
    """Compute TF-IDF similarity of each node's content to the query."""
    if not nodes:
        return {}

    documents = [query] + [n.content for n in nodes]
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Empty vocabulary — all documents may be empty
        return {n.id: 0.0 for n in nodes}

    query_vec = tfidf_matrix[0]
    node_vecs = tfidf_matrix[1:]

    # Cosine similarity
    similarities = (node_vecs @ query_vec.transpose()).toarray().flatten()

    return {n.id: float(similarities[i]) for i, n in enumerate(nodes)}


# ── Combined scoring and selection ────────────────────────────────────────────

# Lexical-only weights (no embeddings available)
PAGERANK_WEIGHT = 0.4
TFIDF_WEIGHT = 0.6
# Hybrid weights (semantic embeddings available) — semantic dominates because
# it is what makes natural-language queries match differently-worded code.
SEMANTIC_WEIGHT = 0.55
HYBRID_TFIDF_WEIGHT = 0.15
HYBRID_PAGERANK_WEIGHT = 0.30
NEIGHBOR_DECAY = 0.5


_TEST_PENALTY = 0.5  # soft down-weight so implementation outranks tests


def _is_test_file(path: str) -> bool:
    """Heuristic: is this file test code rather than implementation?"""
    p = path.lower()
    return (
        "/tests/" in p or "/test/" in p or "__tests__" in p
        or p.startswith("test/") or p.startswith("tests/")
        or ".test." in p or ".spec." in p or p.endswith("_test.py")
    )


def _minmax(d: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a score dict to [0, 1]."""
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi > lo else 1.0
    return {k: (v - lo) / rng for k, v in d.items()}


def _compute_combined_scores(
    nodes: list,
    edges: list,
    query: str,
    pagerank_weight: float = PAGERANK_WEIGHT,
    tfidf_weight: float = TFIDF_WEIGHT,
    semantic: dict[str, float] | None = None,
) -> dict[str, float]:
    """Combine PageRank + TF-IDF (+ semantic, if provided) into one score per node.

    When *semantic* embeddings scores are supplied, they dominate the blend;
    otherwise this is the original PageRank/TF-IDF combination.
    """
    pr = pagerank(nodes, edges)
    tf = tfidf_scores(nodes, query)

    if semantic:
        pr_n, tf_n, sem_n = _minmax(pr), _minmax(tf), _minmax(semantic)
        combined: dict[str, float] = {}
        for n in nodes:
            combined[n.id] = (
                SEMANTIC_WEIGHT * sem_n.get(n.id, 0.0)
                + HYBRID_TFIDF_WEIGHT * tf_n.get(n.id, 0.0)
                + HYBRID_PAGERANK_WEIGHT * pr_n.get(n.id, 0.0)
            )
        return combined

    if not pr:
        return tf
    if not tf:
        return pr

    pr_n, tf_n = _minmax(pr), _minmax(tf)
    return {
        n.id: pagerank_weight * pr_n.get(n.id, 0.0) + tfidf_weight * tf_n.get(n.id, 0.0)
        for n in nodes
    }


def _select_nodes(
    nodes: list,
    edges: list,
    scores: dict[str, float],
    token_budget: int,
) -> list:
    """3-phase greedy selection: pick → accumulate boosts → fill.

    Phase 1: Greedy selection by score, accumulating adjacency boosts
             without re-sorting. O(n log n) for the initial sort.
    Phase 2: Apply all accumulated boosts to remaining nodes, sort once.
    Phase 3: Fill remaining budget from boosted rankings.

    Overall: O(n log n + n×d) where d = avg neighbor degree.
    """
    node_map = {n.id: n for n in nodes}
    adj = _build_adjacency(nodes, edges)

    # Phase 1: Initial greedy pass — select high-scoring nodes, collect boosts
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    selected_ids: set[str] = set()
    selected_nodes: list = []
    tokens_used = 0
    boost_accumulator: dict[str, float] = defaultdict(float)

    for nid, score in ranked:
        if tokens_used >= token_budget:
            break
        if nid in selected_ids:
            continue
        node = node_map.get(nid)
        if node is None:
            continue

        node_tokens = count_tokens(node.content) if node.content else count_tokens(node.name) + 10
        if tokens_used + node_tokens > token_budget:
            continue

        selected_ids.add(nid)
        selected_nodes.append(node)
        tokens_used += node_tokens

        # Accumulate boosts for neighbors (don't apply yet)
        for neighbor_id in adj.get(nid, set()):
            if neighbor_id not in selected_ids:
                boost_accumulator[neighbor_id] += NEIGHBOR_DECAY * score

    # Phase 2: Apply accumulated boosts, re-sort once
    boosted_scores = dict(scores)
    for nid, boost in boost_accumulator.items():
        if nid not in selected_ids:
            boosted_scores[nid] = boosted_scores.get(nid, 0) + boost

    ranked_boosted = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)

    # Phase 3: Fill remaining budget with boosted nodes
    for nid, _score in ranked_boosted:
        if tokens_used >= token_budget:
            break
        if nid in selected_ids:
            continue
        node = node_map.get(nid)
        if node is None:
            continue

        node_tokens = count_tokens(node.content) if node.content else count_tokens(node.name) + 10
        if tokens_used + node_tokens > token_budget:
            continue

        selected_ids.add(nid)
        selected_nodes.append(node)
        tokens_used += node_tokens

    return selected_nodes


# ── Markdown formatting ──────────────────────────────────────────────────────

_LANG_FROM_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}


def _format_node(node) -> str:
    """Format a single graph node as markdown."""
    import os
    _, ext = os.path.splitext(node.file)
    lang = _LANG_FROM_EXT.get(ext, "")

    header = f"### {node.name} ({node.file}:{node.line_start}-{node.line_end})"
    if node.type.value in ("function", "method"):
        header = f"### {node.name}() ({node.file}:{node.line_start}-{node.line_end})"
    elif node.type.value == "class":
        header = f"### class {node.name} ({node.file}:{node.line_start}-{node.line_end})"

    if node.content:
        code_block = f"```{lang}\n{node.content}\n```"
    else:
        code_block = f"*({node.type.value} — no content stored)*"

    return f"{header}\n\n{code_block}"


def format_subgraph(nodes: list, token_budget: int) -> str:
    """Format selected nodes into a markdown document."""
    tokens_used = sum(
        count_tokens(n.content) if n.content else count_tokens(n.name) + 10
        for n in nodes
    )

    parts = [f"## Selected Nodes ({tokens_used} / {token_budget} tokens)\n"]

    # Group by file
    by_file: dict[str, list] = defaultdict(list)
    for n in nodes:
        by_file[n.file].append(n)

    for filepath in sorted(by_file.keys()):
        parts.append(f"\n#### {filepath}\n")
        for node in by_file[filepath]:
            parts.append(_format_node(node))
            parts.append("")

    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

# Below this raw cosine similarity, semantic matches are considered weak.
# Calibrated against real runs (good matches land ~0.35-0.70; genuine misses
# fall well below): 0.15 flags only true no-matches, avoiding false "low" labels
# on correct results (which previously nudged the model to re-read files).
SEMANTIC_CONFIDENCE_THRESHOLD = 0.15

# Adaptive budgeting bounds. When token_budget <= 0 the budget is sized from how
# much is actually relevant (sum of clearly-relevant nodes' tokens), clamped to
# this range. A hard MAX also caps explicit budgets so a stray huge value (e.g.
# the model passing 64000) can't blow up context.
ADAPTIVE_MIN = 1200
ADAPTIVE_MAX = 3500   # keep each response small so even several calls stay bounded
MAX_BUDGET = 4000     # (Claude Code spills >~10k tool results to disk → re-reads)
ADAPTIVE_ABS_THRESHOLD = 0.28   # raw cosine: nodes this similar count as "clearly relevant"
ADAPTIVE_REL_FRACTION = 0.6     # lexical fallback: fraction of top score


def _resolve_budget(token_budget: int, nodes: list, scores: dict[str, float],
                    relevance: dict[str, float] | None = None) -> int:
    """Compute the effective token budget.

    token_budget > 0 → honor it, capped at MAX_BUDGET.
    token_budget <= 0 → adaptive: sum the tokens of *clearly-relevant* nodes,
    clamped to [MIN, MAX]. When raw semantic similarities are available, relevance
    is judged on an absolute cosine threshold (so a narrow question with few
    strong matches yields a small budget); otherwise a relative cliff on the
    normalized lexical scores.
    """
    if token_budget > 0:
        return min(token_budget, MAX_BUDGET)

    use_abs = relevance is not None and len(relevance) > 0
    src = relevance if use_abs else scores
    if not src:
        return ADAPTIVE_MIN
    if use_abs:
        threshold = ADAPTIVE_ABS_THRESHOLD
    else:
        top = max(src.values())
        if top <= 0:
            return ADAPTIVE_MIN
        threshold = ADAPTIVE_REL_FRACTION * top

    node_by_id = {n.id: n for n in nodes}
    total = 0
    for nid, s in src.items():
        if s >= threshold:
            n = node_by_id.get(nid)
            if n is not None:
                total += count_tokens(n.content) if n.content else count_tokens(n.name) + 10
    return max(ADAPTIVE_MIN, min(total, ADAPTIVE_MAX))


def query_graph(
    graph: KnowledgeGraph,
    query: str,
    token_budget: int = 0,
    pagerank_weight: float = PAGERANK_WEIGHT,
    tfidf_weight: float = TFIDF_WEIGHT,
    with_diagnostics: bool = False,
    exclude_ids: set | None = None,
):
    """Query a knowledge graph and return relevant subgraph markdown.

    Returns the 3-tuple ``(markdown, selected_node_count, tokens_used)`` by
    default. With ``with_diagnostics=True`` returns a dict that additionally
    reports the scoring method, a confidence value/label, and per-node scores —
    so callers can see *why* a result was produced and whether to trust it.
    """
    empty_diag = {"markdown": "", "selected_nodes": 0, "tokens_used": 0,
                  "method": "none", "confidence": 0.0, "confidence_label": "none",
                  "top_nodes": []}

    if not graph.nodes:
        md = "## No nodes found in graph\nThe codebase graph is empty."
        if with_diagnostics:
            return {**empty_diag, "markdown": md}
        return md, 0, 0

    # Score content-bearing nodes. Skip *empty* file nodes, but keep file nodes
    # that were fallback-indexed with body content (so no file is invisible).
    content_nodes = [
        n for n in graph.nodes
        if n.type != NodeType.FILE or (n.content and n.content.strip())
    ]
    if not content_nodes:
        content_nodes = graph.nodes

    # Cross-call dedup: drop nodes already returned by a recent call so repeated
    # / fanned-out queries don't re-send the same code. Ignored if it would
    # leave nothing to select.
    if exclude_ids:
        filtered = [n for n in content_nodes if n.id not in exclude_ids]
        if filtered:
            content_nodes = filtered

    # Semantic scoring (None if embeddings unavailable → lexical fallback).
    from . import embeddings
    semantic = embeddings.semantic_scores(content_nodes, query)
    method = "semantic+lexical" if semantic else "lexical"

    scores = _compute_combined_scores(
        content_nodes, graph.edges, query, pagerank_weight, tfidf_weight, semantic=semantic
    )

    # Soft down-rank test files so implementation surfaces first — unless the
    # query is itself about tests.
    if "test" not in query.lower() and "spec" not in query.lower():
        file_by_id = {n.id: n.file for n in content_nodes}
        for nid in list(scores):
            if _is_test_file(file_by_id.get(nid, "")):
                scores[nid] *= _TEST_PENALTY

    # Confidence = raw top relevance (cosine if semantic, else top TF-IDF cosine).
    if semantic:
        confidence = max(semantic.values()) if semantic else 0.0
        low_conf = confidence < SEMANTIC_CONFIDENCE_THRESHOLD
    else:
        tf = tfidf_scores(content_nodes, query)
        confidence = max(tf.values()) if tf else 0.0
        low_conf = confidence < 0.05
    confidence_label = "low" if low_conf else ("high" if confidence >= 0.5 else "medium")

    # Resolve the effective budget (adaptive when token_budget <= 0). Use raw
    # semantic similarities for an absolute relevance judgment when available.
    effective_budget = _resolve_budget(token_budget, content_nodes, scores, relevance=semantic)

    selected = _select_nodes(content_nodes, graph.edges, scores, effective_budget)

    if not selected:
        md = "## No relevant nodes found\nTry broadening your query."
        if with_diagnostics:
            return {**empty_diag, "markdown": md, "method": method,
                    "confidence": confidence, "confidence_label": "low"}
        return md, 0, 0

    markdown = format_subgraph(selected, effective_budget)

    # Low-confidence note: keep it informational and do NOT tell the model to go
    # read files — that would defeat the token savings. The most relevant code
    # found is still included below for it to answer from.
    if low_conf:
        banner = (
            f"> Note: best match was weak ({confidence:.2f} via {method}); the "
            f"query may not map cleanly to this codebase. The closest relevant "
            f"code is included below — answer from it.\n\n"
        )
        markdown = banner + markdown

    tokens_used = count_tokens(markdown)

    if with_diagnostics:
        top_nodes = [
            {"id": nid, "score": round(s, 4)}
            for nid, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        return {
            "markdown": markdown,
            "selected_nodes": len(selected),
            "tokens_used": tokens_used,
            "budget": effective_budget,
            "budget_mode": "adaptive" if token_budget <= 0 else "explicit",
            "method": method,
            "confidence": round(confidence, 4),
            "confidence_label": confidence_label,
            "top_nodes": top_nodes,
            "files": sorted({n.file for n in selected}),
            "selected_ids": [n.id for n in selected],
        }

    return markdown, len(selected), tokens_used
