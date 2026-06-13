"""Tests for graph diffing."""

from fittok.models import (
    KnowledgeGraph, GraphNode, GraphEdge, GraphMetadata,
    NodeType, EdgeType,
)
from fittok.diff import diff_graphs


def _make_graph(nodes, edges, root="/test"):
    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        metadata=GraphMetadata(root=root, total_nodes=len(nodes), total_edges=len(edges)),
    )


class TestDiffGraphs:
    def test_identical_graphs(self):
        nodes = [
            GraphNode(id="f:hello", type=NodeType.FUNCTION, name="hello",
                      file="a.py", content="def hello(): pass"),
        ]
        g = _make_graph(nodes, [])
        result = diff_graphs(g, g)
        assert result["summary"] == "No changes detected"
        assert len(result["nodes_added"]) == 0
        assert len(result["nodes_removed"]) == 0

    def test_nodes_added(self):
        g_a = _make_graph([], [])
        g_b = _make_graph([
            GraphNode(id="f:new", type=NodeType.FUNCTION, name="new",
                      file="b.py", content="def new(): pass"),
        ], [])
        result = diff_graphs(g_a, g_b)
        assert len(result["nodes_added"]) == 1
        assert result["nodes_added"][0]["name"] == "new"

    def test_nodes_removed(self):
        g_a = _make_graph([
            GraphNode(id="f:old", type=NodeType.FUNCTION, name="old",
                      file="a.py", content="def old(): pass"),
        ], [])
        g_b = _make_graph([], [])
        result = diff_graphs(g_a, g_b)
        assert len(result["nodes_removed"]) == 1
        assert result["nodes_removed"][0]["name"] == "old"

    def test_nodes_modified(self):
        g_a = _make_graph([
            GraphNode(id="f:hello", type=NodeType.FUNCTION, name="hello",
                      file="a.py", content="def hello(): pass"),
        ], [])
        g_b = _make_graph([
            GraphNode(id="f:hello", type=NodeType.FUNCTION, name="hello",
                      file="a.py", content="def hello(): return 42"),
        ], [])
        result = diff_graphs(g_a, g_b)
        assert len(result["nodes_modified"]) == 1

    def test_edges_added(self):
        g_a = _make_graph([
            GraphNode(id="f:a", type=NodeType.FUNCTION, name="a",
                      file="a.py", content="def a(): pass"),
            GraphNode(id="f:b", type=NodeType.FUNCTION, name="b",
                      file="b.py", content="def b(): pass"),
        ], [])
        g_b = _make_graph([
            GraphNode(id="f:a", type=NodeType.FUNCTION, name="a",
                      file="a.py", content="def a(): pass"),
            GraphNode(id="f:b", type=NodeType.FUNCTION, name="b",
                      file="b.py", content="def b(): pass"),
        ], [
            GraphEdge(source="f:a", target="f:b", type=EdgeType.CALLS),
        ])
        result = diff_graphs(g_a, g_b)
        assert len(result["edges_added"]) == 1

    def test_summary_content(self):
        g_a = _make_graph([
            GraphNode(id="f:old", type=NodeType.FUNCTION, name="old",
                      file="a.py", content="old"),
        ], [])
        g_b = _make_graph([
            GraphNode(id="f:new", type=NodeType.FUNCTION, name="new",
                      file="b.py", content="new"),
        ], [])
        result = diff_graphs(g_a, g_b)
        assert "nodes added" in result["summary"]
        assert "nodes removed" in result["summary"]
