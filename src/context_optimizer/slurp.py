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

PAGERANK_WEIGHT = 0.4
TFIDF_WEIGHT = 0.6
NEIGHBOR_DECAY = 0.5


def _compute_combined_scores(
    nodes: list,
    edges: list,
    query: str,
    pagerank_weight: float = PAGERANK_WEIGHT,
    tfidf_weight: float = TFIDF_WEIGHT,
) -> dict[str, float]:
    """Combine PageRank and TF-IDF into a single relevance score per node."""
    pr = pagerank(nodes, edges)
    tf = tfidf_scores(nodes, query)

    if not pr:
        return tf
    if not tf:
        return pr

    # Normalize each to [0, 1]
    pr_vals = list(pr.values())
    tf_vals = list(tf.values())

    pr_min, pr_max = min(pr_vals), max(pr_vals)
    tf_min, tf_max = min(tf_vals), max(tf_vals)

    pr_range = pr_max - pr_min if pr_max > pr_min else 1.0
    tf_range = tf_max - tf_min if tf_max > tf_min else 1.0

    combined: dict[str, float] = {}
    for n in nodes:
        pr_norm = (pr.get(n.id, 0) - pr_min) / pr_range
        tf_norm = (tf.get(n.id, 0) - tf_min) / tf_range
        combined[n.id] = pagerank_weight * pr_norm + tfidf_weight * tf_norm

    return combined


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
    ".tsx": "typescript",
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

def query_graph(
    graph: KnowledgeGraph,
    query: str,
    token_budget: int = 4000,
    pagerank_weight: float = PAGERANK_WEIGHT,
    tfidf_weight: float = TFIDF_WEIGHT,
) -> tuple[str, int, int]:
    """Query a knowledge graph and return relevant subgraph markdown.

    Returns:
        (markdown, selected_node_count, tokens_used)
    """
    if not graph.nodes:
        return "## No nodes found in graph\nThe codebase graph is empty.", 0, 0

    # Filter to content-bearing nodes (skip file nodes for scoring)
    content_nodes = [n for n in graph.nodes if n.type != NodeType.FILE]

    if not content_nodes:
        content_nodes = graph.nodes  # fallback to all nodes

    scores = _compute_combined_scores(
        content_nodes, graph.edges, query, pagerank_weight, tfidf_weight
    )

    selected = _select_nodes(content_nodes, graph.edges, scores, token_budget)

    if not selected:
        return "## No relevant nodes found\nTry broadening your query.", 0, 0

    markdown = format_subgraph(selected, token_budget)
    tokens_used = count_tokens(markdown)

    return markdown, len(selected), tokens_used
