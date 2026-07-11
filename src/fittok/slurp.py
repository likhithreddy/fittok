"""slurp — query a knowledge graph within a token budget.

Uses PageRank + TF-IDF to select the most relevant nodes for a given query,
then formats them as markdown within the specified token limit.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
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
    # Precompute per-node degree once so the hot inner loop doesn't call
    # len(adj.get(...)) for every neighbor on every iteration.
    degree = {nid: len(nb) for nid, nb in adj.items()}

    # Initialize scores uniformly
    scores = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        prev = scores.copy()
        for i, nid in enumerate(node_ids):
            neighbors = adj.get(nid)
            if not neighbors:
                scores[i] = (1 - damping) / n
                continue
            rank_sum = 0.0
            for neighbor in neighbors:
                rank_sum += prev[id_to_idx[neighbor]] / max(degree.get(neighbor, 1), 1)
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


# ── BM25 scoring ──────────────────────────────────────────────────────────────


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize for BM25 — split identifiers (camelCase, snake_case, paths) into
    individual terms so 'runSandboxQuery' → ['run', 'sandbox', 'query'] and
    'execute.ts' → ['execute', 'ts']. This bridges the vocabulary gap between
    natural-language queries and code identifiers."""
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    tokens: list[str] = []
    for w in words:
        # Split camelCase: "runSandboxQuery" → "run" "Sandbox" "Query"
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w).split()
        for p in parts:
            tokens.extend(p.lower().split("_"))
    return tokens


def bm25_scores(nodes: list, query: str) -> dict[str, float]:
    """BM25 similarity of each node (name + file + content) to the query.

    Stronger than TF-IDF for code: term-frequency saturation rewards repeated
    terms without over-boosting; document-length normalization handles the
    large-vs-small function disparity. camelCase/snake_case splitting lets
    'execute' match 'execute.ts' and 'query' match 'runSandboxQuery'.
    """
    if not nodes:
        return {}
    docs = [_tokenize_for_bm25(f"{n.name} {n.file} {n.content or ''}") for n in nodes]
    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        return {n.id: 0.0 for n in nodes}
    bm25 = BM25Okapi(docs)
    raw_scores = bm25.get_scores(query_tokens)
    return {nodes[i].id: float(raw_scores[i]) for i in range(len(nodes))}


def _build_edge_indexes(edges) -> tuple[dict, dict]:
    """Build source→targets and target→sources indexes for O(1) caller/callee lookup."""
    by_source: dict[str, list] = defaultdict(list)
    by_target: dict[str, list] = defaultdict(list)
    for e in edges:
        by_source[e.source].append(e.target)
        by_target[e.target].append(e.source)
    return dict(by_source), dict(by_target)


def _short_name(node_id: str) -> str:
    """Extract the human-readable name from a node ID like 'function:lib/foo.ts:bar:12'."""
    parts = node_id.split(":")
    return parts[-2] if len(parts) >= 2 else node_id


def _node_summary(node, by_source: dict, by_target: dict) -> str:
    """One-line summary of a node for summary-BM25 — includes structural context
    (callers + callees) that bridges query↔code vocabulary better than raw content.

    Example: 'runSandboxQuery lib/sandbox/execute.ts function callers:run,submit calls:validateQuery,connect,query'
    """
    callers = by_target.get(node.id, [])
    callees = by_source.get(node.id, [])
    parts = [node.name, node.file, node.type.value]
    if callers:
        parts.append("callers:" + ",".join(_short_name(c) for c in callers[:5]))
    if callees:
        parts.append("calls:" + ",".join(_short_name(c) for c in callees[:5]))
    return " ".join(parts)


def summary_bm25_scores(nodes, query, by_source, by_target) -> dict[str, float]:
    """BM25 on one-line node summaries (name + file + callers + callees).

    Richer than content-BM25: the summary 'runSandboxQuery execute.ts callers:run,submit
    calls:validateQuery,connect' matches 'server executes SQL query' via structural
    context (callers connect UI→server, callees reveal execution helpers).
    """
    if not nodes:
        return {}
    summaries = [_tokenize_for_bm25(_node_summary(n, by_source, by_target)) for n in nodes]
    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        return {n.id: 0.0 for n in nodes}
    bm25 = BM25Okapi(summaries)
    raw_scores = bm25.get_scores(query_tokens)
    return {nodes[i].id: float(raw_scores[i]) for i in range(len(nodes))}


def _extract_purpose(content: str) -> str:
    """Extract a one-line purpose from a node's docstring/JSDoc/comment."""
    if not content:
        return ""
    # JSDoc: /** ... */
    m = re.search(r'/\*\*\s*(.+?)\*/', content, re.DOTALL)
    if m:
        text = re.sub(r'^\s*\*\s?', '', m.group(1), flags=re.MULTILINE).strip()
        first = text.split('.')[0].strip()
        if first and len(first) > 5:
            return first[:150]
    # Python docstring
    m = re.search(r'"""(.+?)"""', content, re.DOTALL)
    if m:
        first = m.group(1).strip().split('\n')[0].strip()
        if first and len(first) > 5:
            return first[:150]
    # Leading // comment
    m = re.match(r'\s*//\s*(.+)', content)
    if m:
        return m.group(1).strip()[:150]
    return ""


def generate_codebase_map(graph, max_tokens: int = 500) -> str:
    """Generate a compact codebase table-of-contents with PURPOSE descriptions.

    Lists functions/classes by file with their docstrings (human-written
    descriptions). This is the 'map' the model reads to route its queries —
    inspired by Karpathy's LLM Wiki and Google's OKF. If the code slice misses
    a file, the model sees it in the map (with its purpose) and can make a
    precise follow-up call.
    """
    by_file: dict[str, list] = defaultdict(list)
    for n in graph.nodes:
        if n.type.value not in ("file", "module") and n.name:
            by_file[n.file].append(n)
    lines = ["## Codebase map (functions + purpose)\n"]
    tokens = count_tokens(lines[0])
    for file_path in sorted(by_file):
        nodes = sorted(by_file[file_path], key=lambda n: n.line_start)
        for n in nodes[:3]:  # top 3 per file (purpose lines are token-expensive)
            purpose = _extract_purpose(n.content or "")
            entry = f"- `{n.name}` ({file_path}:L{n.line_start})"
            if purpose:
                entry += f" — {purpose}"
            t = count_tokens(entry)
            if tokens + t > max_tokens:
                return "\n".join(lines)
            lines.append(entry)
            tokens += t
    return "\n".join(lines)


def generate_flow_trace(selected, edges, max_tokens: int = 300) -> str:
    """Deterministic call-path narrative among the selected nodes.

    This is the lever for FLOW questions ("how does X work?"). The evidence:
    on a flow query the model re-read files fittok had ALREADY returned in full,
    because it couldn't verify the slice was complete — flows have fuzzy edges,
    so "is there more?" stays open and the model re-reads to be sure. On a
    single-function query it trusts fittok, because completeness is obvious.

    This trace gives the model that same verifiable completeness for flows: it
    lists the WHOLE call chain among the selected nodes (entry → callees) with
    file:line anchors, so the model sees nothing is omitted and stops
    re-reading. No LLM — derived from the CALLS edges of the knowledge graph.
    Returns "" for a single node (no flow to show).
    """
    from .models import EdgeType

    if len(selected) < 2:
        return ""

    sel_ids = {n.id for n in selected}
    id_to_node = {n.id: n for n in selected}

    # Calls WITHIN the surfaced set = the internal flow we render.
    calls: dict[str, list[str]] = defaultdict(list)
    called: set[str] = set()
    for e in edges:
        if e.type == EdgeType.CALLS and e.source in sel_ids and e.target in sel_ids:
            calls[e.source].append(e.target)
            called.add(e.target)

    # Only render nodes that participate in a CALLS edge — isolated nodes (no
    # call relationship to anything else surfaced) aren't part of a 'flow' and
    # would just be retrieval noise (e.g. an unrelated health-check route).
    in_flow = set(calls.keys()) | called
    if len(in_flow) < 2:
        return ""

    # Entry points: in-flow nodes not called by another in-flow node.
    entries = [n for n in selected if n.id in in_flow and n.id not in called]
    if not entries:
        entries = [n for n in selected if n.id in in_flow]

    def _loc(node) -> str:
        return f"{node.file}:L{node.line_start}"

    lines = ["## Flow — the complete call chain for this query\n"]
    seen: set[str] = set()
    budget = count_tokens(lines[0])

    def render(nid: str, depth: int) -> bool:
        nonlocal budget
        node = id_to_node.get(nid)
        if node is None or nid in seen or depth > 3:
            return True  # ok
        line = f"{'  ' * depth}- `{node.name}()` ({_loc(node)})"
        lt = count_tokens(line + "\n")
        if budget + lt > max_tokens:
            return False  # over budget — stop
        lines.append(line)
        seen.add(nid)
        budget += lt
        for t in calls.get(nid, [])[:4]:
            if not render(t, depth + 1):
                return False
        return True

    for n in entries[:3]:
        if not render(n.id, 0):
            break

    if len(lines) <= 1:
        return ""  # header only — no edges rendered → nothing useful
    return "\n".join(lines)


def _reciprocal_rank_fusion(score_dicts: list, k: int = 60) -> dict[str, float]:
    """Fuse multiple ranked score dicts via Reciprocal Rank Fusion.

    Each node gets 1/(k+rank) per signal, summed across signals. A node
    consistently top-ranked across semantic, BM25, and PageRank scores highest.
    Rank-based fusion avoids the calibration problem of mixing cosine [0,1]
    with BM25 [0,∞] — their RANKS are directly comparable even when their raw
    scores are not.
    """
    fused: dict[str, float] = defaultdict(float)
    for scores in score_dicts:
        if not scores:
            continue
        # Skip all-zero signals — they add rank noise without information.
        if max(scores.values()) <= 0:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (nid, _) in enumerate(ranked, 1):
            fused[nid] += 1.0 / (k + rank)
    return dict(fused)


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
    tf: dict[str, float] | None = None,
) -> dict[str, float]:
    """Combine PageRank + TF-IDF (+ semantic, if provided) into one score per node.

    When *semantic* embeddings scores are supplied, they dominate the blend;
    otherwise this is the original PageRank/TF-IDF combination. Pass ``tf`` to
    reuse a TF-IDF dict the caller already computed (avoids recomputing it).
    """
    pr = pagerank(nodes, edges)
    if tf is None:
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


def _dir_of(file_path: str) -> str:
    """Extract the directory for MMR diversity (e.g. 'lib/auth/server.ts' → 'lib/auth')."""
    parts = file_path.rsplit("/", 1)
    return parts[0] if len(parts) > 1 else file_path


# MMR diversity: penalize selecting multiple nodes from the same directory so
# the slice spans multiple code areas (execution + rendering + isolation) instead
# of clustering in one dominant area (e.g. auth). λ=0.7 means 70% relevance,
# 30% diversity — enough to surface a facet-2 node before a facet-1 node #4.
MMR_LAMBDA = 0.7
MMR_DIR_PENALTY = 0.35  # per same-dir node already selected
MAX_PER_DIR = 3  # hard cap: forces facet coverage on multi-aspect queries
# A selected node is always returned WHOLE (budget controls HOW MANY nodes, not
# how much of each). The cap only guards against one pathological giant eating
# the entire budget — set at 60% so a normal 90-line route handler (~600-800
# tokens) fits at a 1500-2500 token budget and is no longer SKIPPED (the old
# 0.25 cap excluded exactly these handlers, which is what forced the model to
# re-query at ever-higher budgets: 1800 → 2200 → 2600).
MAX_NODE_BUDGET_RATIO = 0.6


def _select_nodes(
    nodes: list,
    edges: list,
    scores: dict[str, float],
    token_budget: int,
    eligible_ids: set | None = None,
) -> list:
    """Greedy selection with MMR (Maximal Marginal Relevance) diversity.

    Picks the highest-scoring eligible node, then for each subsequent pick,
    penalizes nodes from directories already represented. This forces the slice
    to span multiple code areas (execution, rendering, isolation) instead of
    clustering in one dominant area (e.g. auth — which structurally out-ranks
    everything for any query containing "user").
    """
    node_map = {n.id: n for n in nodes}
    adj = _build_adjacency(nodes, edges)

    if eligible_ids is None:
        eligible = dict(scores)
    else:
        eligible = {nid: s for nid, s in scores.items() if nid in eligible_ids}

    selected_ids: set[str] = set()
    selected_nodes: list = []
    tokens_used = 0
    boost_accumulator: dict[str, float] = defaultdict(float)
    dir_counts: dict[str, int] = defaultdict(int)  # MMR: per-directory selection count

    # Round-robin directory interleaving: pick the best node from EACH directory
    # before any directory gets a second. This GUARANTEES facet coverage on
    # multi-aspect queries — runSandboxQuery (top of lib/sandbox/) is picked in
    # round 1 alongside getCurrentUser (top of lib/auth/), instead of being
    # crowded out by auth's higher raw scores. MAX_PER_DIR caps depth.
    dir_queues: dict[str, list] = defaultdict(list)
    for nid, score in sorted(eligible.items(), key=lambda x: x[1], reverse=True):
        node = node_map.get(nid)
        if node:
            dir_queues[_dir_of(node.file)].append((nid, score, node))

    while tokens_used < token_budget:
        picked_any = False
        max_node_tokens = int(token_budget * MAX_NODE_BUDGET_RATIO)
        # Each round: pick one node from each directory (highest-scored first)
        for d in sorted(dir_queues, key=lambda dd: dir_queues[dd][0][1] if dir_queues[dd] else 0, reverse=True):
            if not dir_queues[d] or dir_counts[d] >= MAX_PER_DIR:
                continue
            nid, score, node = dir_queues[d].pop(0)
            if nid in selected_ids:
                continue
            node_tokens = node.token_count or (count_tokens(node.content) if node.content else count_tokens(node.name) + 10)
            # Per-node cap: only guards against one pathological giant eating the
            # whole budget. At 0.6, a normal handler fits and is included whole —
            # we never skip a relevant handler just for being long (that was the
            # cause of the budget-escalation re-query loop).
            if node_tokens > max_node_tokens:
                continue
            if tokens_used + node_tokens > token_budget:
                continue
            selected_ids.add(nid)
            selected_nodes.append(node)
            tokens_used += node_tokens
            dir_counts[d] += 1
            for neighbor_id in adj.get(nid, set()):
                if neighbor_id not in selected_ids:
                    boost_accumulator[neighbor_id] += NEIGHBOR_DECAY * score
            picked_any = True
        if not picked_any:
            break

    return selected_nodes


# ── Referenced-dependency expansion ──────────────────────────────────────────
#
# The re-read problem: a returned function often references a symbol whose own
# definition was NOT query-relevant (below the relevance cliff) — a module
# constant like MODEL_NAME, or a small helper. The model, staring at an undefined
# name, used to re-open the file to see it, which discards the token savings.
#
# Fix: after selecting the query-relevant nodes, scan their content for
# identifiers that resolve to *definition* nodes in the graph, and surface those
# definitions as compact one-liners. Exempt from the relevance cliff (they're
# structurally required, not semantically hit), but capped by a token sub-budget
# so they can't re-bloat the output. Resolving by name (not by CALLS/REFERENCES
# edges) also catches referenced constants, which carry no call edge.

# Identifiers of length >= 2 (skips single-letter throwaway names). Word-boundary
# anchored so it won't match inside longer tokens.
_NEIGHBOR_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b")

NEIGHBOR_MAX_TOKENS = 300      # absolute cap on the dependency section
NEIGHBOR_BUDGET_FRACTION = 0.3  # …and never more than 30% of the response budget


def _neighbor_line(node) -> str:
    """One compact line for a referenced dependency: its declaration + location."""
    sig = _node_signature(node.content)
    return f"- `{sig}` ({node.file}:{node.line_start})"


def _select_neighbors(
    selected: list,
    all_nodes: list,
    token_cap: int,
    selected_ids: set,
    exclude_ids: set | None = None,
):
    """Return ``(neighbor_nodes, markdown)`` for definitions referenced by
    *selected* but not already included.

    Names are resolved against the graph's definition nodes (FUNCTION, METHOD,
    CLASS, CONSTANT) by name. Order is by reference frequency (most-used first),
    so the most important dependencies win when the cap is tight.
    """
    if token_cap <= 0 or not selected:
        return [], ""

    name_index: dict[str, object] = {}
    for n in all_nodes:
        if n.type in (NodeType.FILE, NodeType.IMPORT):
            continue
        name_index.setdefault(n.name, n)  # first definition wins

    excluded = set(exclude_ids or ()) | set(selected_ids)
    refcount: dict[str, int] = defaultdict(int)
    for n in selected:
        for ident in _NEIGHBOR_IDENT_RE.findall(n.content or ""):
            if ident in name_index:
                refcount[ident] += 1

    header = "\n\n## Referenced dependencies\n"
    chosen: list = []
    lines: list[str] = []
    total = count_tokens(header)
    for name, _cnt in sorted(refcount.items(), key=lambda kv: (-kv[1], kv[0])):
        node = name_index[name]
        if node.id in excluded:
            continue
        line = _neighbor_line(node)
        line_tokens = count_tokens(line) + 1  # +1 for the joining newline
        if total + line_tokens > token_cap:
            continue  # doesn't fit; a later, smaller one still might
        chosen.append(node)
        lines.append(line)
        total += line_tokens
        excluded.add(node.id)  # don't add the same node twice under aliased names

    if not chosen:
        return [], ""
    return chosen, header + "\n".join(lines)


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


# The top-ranked nodes are returned in full; the rest as signature-only context.
FULL_DETAIL_NODES = 6


def _node_signature(content: str) -> str:
    """First meaningful line of a node — its declaration — for supporting context."""
    for line in (content or "").splitlines():
        if line.strip():
            return line.strip()[:160]
    return ""


def _number_lines(content: str, start_line: int) -> str:
    """Prefix each line with its REAL source line number, ``nl``-style, so the
    model can cite ``file:line`` directly from the output.

    This is the fix for the re-read loop we saw in Copilot: the model ran
    ``nl -ba <file> | sed`` to recover line anchors for its citations because
    the returned code had none. Numbering starts at the node's ``line_start``
    so the numbers match what the model sees in its editor / Read tool — the
    anchor is already in the output, so re-opening the file adds nothing.
    """
    if not content:
        return content
    return "\n".join(
        f"{start_line + i}: {line}" for i, line in enumerate(content.splitlines())
    )


def _format_node(node, full: bool) -> str:
    """Format a node as markdown. *full* → whole body (line-numbered); else
    signature-only (also line-numbered, for citation)."""
    import os
    _, ext = os.path.splitext(node.file)
    lang = _LANG_FROM_EXT.get(ext, "")

    if node.type.value in ("function", "method"):
        header = f"### {node.name}() ({node.file}:{node.line_start}-{node.line_end})"
    elif node.type.value == "class":
        header = f"### class {node.name} ({node.file}:{node.line_start}-{node.line_end})"
    else:
        header = f"### {node.name} ({node.file}:{node.line_start}-{node.line_end})"

    if not node.content:
        body = f"*({node.type.value} — no content stored)*"
    elif full:
        body = f"```{lang}\n{_number_lines(node.content, node.line_start)}\n```"
    else:
        # Supporting node: declaration only, with its line number for citation.
        body = f"```{lang}\n{node.line_start}: {_node_signature(node.content)}  // …\n```"
    return f"{header}\n\n{body}"


def format_subgraph(nodes: list, token_budget: int) -> str:
    """Format selected nodes into markdown within *token_budget*, NEVER cutting a
    node mid-body.

    Full detail for the top nodes; signature-only for the supporting tail. The
    budget controls HOW MANY nodes appear, not how much of each — a selected
    node is always included whole. This is the fix for the mid-body truncation
    that left handlers cut off and forced the model to re-open the file to see
    the rest. The first node is always shown (even if slightly over) so the
    slice is never empty; after that, when the next whole node won't fit, we
    stop rather than slicing it.
    """
    full_ids = {n.id for n in nodes[:FULL_DETAIL_NODES]}

    by_file: dict[str, list] = defaultdict(list)
    for n in nodes:
        by_file[n.file].append(n)

    header = "## Relevant code\n"
    parts: list[str] = [header]
    # No budget → include everything (e.g. the CLI `--json` / graph paths).
    budget_left = (token_budget - count_tokens(header)) if token_budget > 0 else 1 << 30
    shown_any = False

    for filepath in by_file:  # insertion order = relevance order, not alphabetical
        file_header = f"\n#### {filepath}\n"
        ht = count_tokens(file_header)
        if ht > budget_left and shown_any:
            break
        parts.append(file_header)
        budget_left -= ht
        for node in by_file[filepath]:
            node_md = _format_node(node, full=node.id in full_ids) + "\n"
            nt = count_tokens(node_md)
            # Stop at a node boundary — never cut a function mid-body. The first
            # node is force-included so the slice is never empty.
            if shown_any and nt > budget_left:
                return "\n".join(parts)
            parts.append(node_md)
            budget_left -= nt
            shown_any = True

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
ADAPTIVE_MIN = 600    # a 1-2 function answer should be able to come back this small
ADAPTIVE_MAX = 1200   # Ceiling on the CODE slice. The REAL Copilot limit (verified
                      # from a captured content.json): MCP outputs above ~7-8 KB get
                      # cached to content.json, where the ENTIRE markdown collapses
                      # to ONE physical JSON line (newlines escaped) — and Copilot's
                      # Read tool truncates any line at ~2000 chars. So a cached
                      # output is effectively chopped to ~2000 chars regardless of
                      # total size; the model can't see code past that mark. Staying
                      # UNDER ~7 KB keeps the output INLINE (real newlines, fully
                      # visible). 1200 code + map + flow + note ≈ 6.4 KB total.
MAX_BUDGET = 1200     # Hard cap on the CODE slice, for the same inline-delivery
                      # reason. Returning more just pushes the total over the cache
                      # threshold and gets it collapsed+truncated — so a smaller,
                      # fully-visible slice beats a larger, invisible one.
# Relevance cliff on the RAW SEMANTIC cosine (the signal that actually
# discriminates relevant from noise — unlike the combined score, which has a
# floor from the near-uniform PageRank on a flat graph). A node is eligible if
# its cosine is within REL_FRACTION of the top cosine AND above ABS_FLOOR.
REL_FRACTION = 0.6
# Absolute floor for the relevance cliff. Kept LOW so the relative fraction
# (REL_FRACTION * top) governs for normal/medium-confidence queries — a higher
# floor overrides the relative cliff on hard/multifaceted queries (where the top
# raw-cosine is modest) and excludes nodes that are genuinely relevant at, say,
# 60-70% of the top score. The floor only kicks in for near-zero-top (garbage)
# queries to avoid returning everything.
ABS_FLOOR = 0.13


def _eligible_ids(relevance: dict[str, float]) -> set:
    """Set of node ids above the relevance cliff (raw semantic cosine)."""
    if not relevance:
        return set()
    top = max(relevance.values())
    cliff = max(REL_FRACTION * top, ABS_FLOOR)
    return {nid for nid, s in relevance.items() if s >= cliff}


def _resolve_budget(token_budget: int, nodes: list, eligible_ids: set) -> int:
    """Compute the effective token budget.

    token_budget > 0 → honor it, capped at MAX_BUDGET.
    token_budget <= 0 → adaptive: exactly the tokens of the eligible (above-cliff)
    nodes, clamped to [MIN, MAX]. The cliff already excludes the weakly-related
    tail, so the budget self-sizes to the relevant cluster without padding.
    """
    if token_budget > 0:
        return min(token_budget, MAX_BUDGET)
    total = 0
    for n in nodes:
        if n.id in eligible_ids:
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
    method = "hybrid+map (RRF)" if semantic else "bm25+map+pagerank (RRF)"

    # 4-signal RRF: semantic + content-BM25 + summary-BM25 + PageRank.
    # Summary-BM25 scores one-line node summaries (name + file + callers + callees)
    # which bridge the vocabulary gap: 'runSandboxQuery execute.ts callers:run,submit
    # calls:validateQuery' matches 'server executes SQL query' via structural context.
    by_source, by_target = _build_edge_indexes(graph.edges)
    bm25 = bm25_scores(content_nodes, query)
    smb25 = summary_bm25_scores(content_nodes, query, by_source, by_target)
    pr = pagerank(content_nodes, graph.edges)
    scores = _reciprocal_rank_fusion([semantic or {}, bm25, smb25, pr])
    # TF-IDF kept only for the lexical-fallback confidence metric.
    tf = tfidf_scores(content_nodes, query)

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
        confidence = max(tf.values()) if tf else 0.0
        low_conf = confidence < 0.05
    confidence_label = "low" if low_conf else ("high" if confidence >= 0.5 else "medium")

    # Eligibility cliff on raw semantic relevance (falls back to combined scores
    # when embeddings are unavailable). Drives BOTH the adaptive budget and what
    # can be selected, so only the genuinely-relevant cluster is returned.
    # Eligibility: relevant by ANY signal — semantic OR content-BM25 OR summary-BM25.
    # The summary-BM25 channel catches nodes whose structural context (callers,
    # callees, file path) matches the query even when the raw content doesn't.
    if semantic:
        eligible_ids = _eligible_ids(semantic) | _eligible_ids(bm25) | _eligible_ids(smb25)
    else:
        eligible_ids = _eligible_ids(bm25) | _eligible_ids(smb25)
    # Exclude test files from eligibility (raw cosine doesn't see the test
    # penalty) unless the query is itself about tests — keep them only if that
    # would otherwise leave nothing.
    if "test" not in query.lower() and "spec" not in query.lower():
        file_by_id = {n.id: n.file for n in content_nodes}
        non_test = {nid for nid in eligible_ids if not _is_test_file(file_by_id.get(nid, ""))}
        if non_test:
            eligible_ids = non_test
    effective_budget = _resolve_budget(token_budget, content_nodes, eligible_ids)

    selected = _select_nodes(content_nodes, graph.edges, scores, effective_budget,
                             eligible_ids=eligible_ids)

    if not selected:
        md = "## No relevant nodes found\nTry broadening your query."
        if with_diagnostics:
            return {**empty_diag, "markdown": md, "method": method,
                    "confidence": confidence, "confidence_label": "low"}
        return md, 0, 0

    main_md = format_subgraph(selected, effective_budget)
    main_tokens = count_tokens(main_md)

    # Referenced-dependency expansion: surface definitions (constants, helpers)
    # that the selected code USES but that weren't themselves query-relevant
    # (below the cliff). This is the fix for the re-read problem — without it,
    # the model used to re-open the file to resolve an undefined MODEL_NAME.
    # Capped to a fraction of the budget, with headroom for the low-conf banner.
    neighbor_cap = max(150, min(
        NEIGHBOR_MAX_TOKENS,
        effective_budget - main_tokens - 64,
        int(effective_budget * NEIGHBOR_BUDGET_FRACTION),
    ))
    neighbor_nodes, deps_md = _select_neighbors(
        selected, graph.nodes, neighbor_cap,
        selected_ids={n.id for n in selected}, exclude_ids=exclude_ids,
    )
    markdown = main_md + deps_md

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
            "neighbor_nodes": len(neighbor_nodes),
            "tokens_used": tokens_used,
            "budget": effective_budget,
            "budget_mode": "adaptive" if token_budget <= 0 else "explicit",
            "method": method,
            "confidence": round(confidence, 4),
            "confidence_label": confidence_label,
            "top_nodes": top_nodes,
            "files": sorted({n.file for n in selected}),
            "selected_ids": [n.id for n in selected],
            "neighbor_ids": [n.id for n in neighbor_nodes],
        }

    return markdown, len(selected), tokens_used


def score_nodes(graph, query: str) -> dict[str, float]:
    """Return a node_id → combined relevance score dict for the given query.

    Used by `fittok graph --query` to highlight relevant nodes without
    running the full selection pipeline.
    """
    from . import embeddings

    content_nodes = [n for n in graph.nodes if n.type.value not in ("file", "import")]
    if not content_nodes:
        return {}

    semantic = embeddings.semantic_scores(content_nodes, query)
    scores = _compute_combined_scores(
        content_nodes, graph.edges, query,
        pagerank_weight=0.15, tfidf_weight=0.25,
        semantic=semantic,
    )
    return scores
