"""Graph diffing — compare two knowledge graphs and report structural differences."""

from __future__ import annotations

import hashlib
from typing import Any

from .models import KnowledgeGraph


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def diff_graphs(graph_a: KnowledgeGraph, graph_b: KnowledgeGraph) -> dict:
    """Compare two knowledge graphs and return structural differences.

    Args:
        graph_a: First (earlier) graph.
        graph_b: Second (later) graph.

    Returns:
        Dict with nodes_added, nodes_removed, nodes_modified, edges_added,
        edges_removed, and a human-readable summary.
    """
    # Index nodes by ID
    nodes_a = {n.id: n for n in graph_a.nodes}
    nodes_b = {n.id: n for n in graph_b.nodes}

    ids_a = set(nodes_a.keys())
    ids_b = set(nodes_b.keys())

    nodes_added = [
        {"id": n.id, "name": n.name, "type": n.type.value, "file": n.file}
        for nid in sorted(ids_b - ids_a)
        if (n := nodes_b[nid])
    ]

    nodes_removed = [
        {"id": n.id, "name": n.name, "type": n.type.value, "file": n.file}
        for nid in sorted(ids_a - ids_b)
        if (n := nodes_a[nid])
    ]

    # Modified: same ID but different content hash
    nodes_modified: list[dict] = []
    for nid in sorted(ids_a & ids_b):
        a, b = nodes_a[nid], nodes_b[nid]
        if _content_hash(a.content) != _content_hash(b.content):
            nodes_modified.append({
                "id": nid,
                "name": b.name,
                "type": b.type.value,
                "file": b.file,
                "line_start": b.line_start,
                "line_end": b.line_end,
            })

    # Index edges by (source, target, type)
    def _edge_key(e: Any) -> str:
        return f"{e.source}|{e.target}|{e.type.value}"

    edges_a = {_edge_key(e): e for e in graph_a.edges}
    edges_b = {_edge_key(e): e for e in graph_b.edges}

    edge_keys_a = set(edges_a.keys())
    edge_keys_b = set(edges_b.keys())

    edges_added = [
        {"source": e.source, "target": e.target, "type": e.type.value}
        for k in sorted(edge_keys_b - edge_keys_a)
        if (e := edges_b[k])
    ]

    edges_removed = [
        {"source": e.source, "target": e.target, "type": e.type.value}
        for k in sorted(edge_keys_a - edge_keys_b)
        if (e := edges_a[k])
    ]

    # Build summary
    parts: list[str] = []
    if nodes_added:
        files_added = {n["file"] for n in nodes_added}
        parts.append(f"{len(nodes_added)} nodes added across {len(files_added)} file(s)")
    if nodes_removed:
        files_removed = {n["file"] for n in nodes_removed}
        parts.append(f"{len(nodes_removed)} nodes removed across {len(files_removed)} file(s)")
    if nodes_modified:
        files_modified = {n["file"] for n in nodes_modified}
        parts.append(f"{len(nodes_modified)} nodes modified across {len(files_modified)} file(s)")
    if edges_added:
        parts.append(f"{len(edges_added)} edges added")
    if edges_removed:
        parts.append(f"{len(edges_removed)} edges removed")

    summary = "; ".join(parts) if parts else "No changes detected"

    return {
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "nodes_modified": nodes_modified,
        "edges_added": edges_added,
        "edges_removed": edges_removed,
        "stats": {
            "graph_a": {"nodes": len(graph_a.nodes), "edges": len(graph_a.edges)},
            "graph_b": {"nodes": len(graph_b.nodes), "edges": len(graph_b.edges)},
        },
        "summary": summary,
    }
