"""Tests for the graphify module."""

from pathlib import Path

import pytest

from context_optimizer.graphify import parse_codebase, save_graph, load_graph
from context_optimizer.models import KnowledgeGraph, NodeType, EdgeType


@pytest.fixture
def sample_python_project(tmp_path):
    """Create a small sample Python project for testing."""
    # Main module
    (tmp_path / "main.py").write_text(
        '"""Main module."""\n'
        "from utils import helper\n"
        "\n"
        "def run():\n"
        '    """Run the app."""\n'
        "    helper()\n"
        "    return True\n"
        "\n"
        "class App:\n"
        "    def start(self):\n"
        "        run()\n"
    )

    # Utils module
    (tmp_path / "utils.py").write_text(
        "def helper():\n"
        '    """Helper function."""\n'
        "    return 42\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )

    # Non-code file (should be ignored)
    (tmp_path / "README.md").write_text("# Test Project")

    return tmp_path


@pytest.fixture
def sample_js_project(tmp_path):
    """Create a small JS project for testing."""
    (tmp_path / "index.js").write_text(
        'const express = require("express");\n'
        "\n"
        "function handler(req, res) {\n"
        "  res.send('hello');\n"
        "}\n"
        "\n"
        "class Server {\n"
        "  start() {\n"
        "    handler();\n"
        "  }\n"
        "}\n"
    )
    return tmp_path


class TestParseCodebase:
    def test_parse_python_project(self, sample_python_project):
        graph = parse_codebase(str(sample_python_project))

        assert isinstance(graph, KnowledgeGraph)
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

        # Should have file nodes
        file_nodes = [n for n in graph.nodes if n.type == NodeType.FILE]
        assert len(file_nodes) == 2  # main.py, utils.py

        # Should have function nodes
        func_nodes = [n for n in graph.nodes if n.type == NodeType.FUNCTION]
        func_names = {n.name for n in func_nodes}
        assert "run" in func_names
        assert "helper" in func_names
        assert "add" in func_names

        # Should have class nodes
        class_nodes = [n for n in graph.nodes if n.type == NodeType.CLASS]
        assert len(class_nodes) >= 1
        class_names = {n.name for n in class_nodes}
        assert "App" in class_names

    def test_parse_js_project(self, sample_js_project):
        graph = parse_codebase(str(sample_js_project))

        assert len(graph.nodes) > 0
        func_nodes = [n for n in graph.nodes if n.type == NodeType.FUNCTION]
        func_names = {n.name for n in func_nodes}
        assert "handler" in func_names

        class_nodes = [n for n in graph.nodes if n.type == NodeType.CLASS]
        class_names = {n.name for n in class_nodes}
        assert "Server" in class_names

    def test_parse_empty_directory(self, tmp_path):
        graph = parse_codebase(str(tmp_path))
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_parse_nonexistent_directory(self):
        with pytest.raises(ValueError, match="Not a directory"):
            parse_codebase("/nonexistent/path")

    def test_metadata(self, sample_python_project):
        graph = parse_codebase(str(sample_python_project))

        assert graph.metadata.root == str(sample_python_project.resolve())
        assert graph.metadata.total_nodes == len(graph.nodes)
        assert graph.metadata.total_edges == len(graph.edges)
        assert graph.metadata.generated_at is not None

    def test_contains_edges(self, sample_python_project):
        graph = parse_codebase(str(sample_python_project))

        contains_edges = [e for e in graph.edges if e.type == EdgeType.CONTAINS]
        # Each function/class should have a CONTAINS edge from its file
        assert len(contains_edges) >= 4  # run, helper, add, App

    def test_skips_node_modules(self, tmp_path):
        """Verify node_modules and other skip dirs are excluded."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("def hello(): pass\n")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.js").write_text(
            "function vendor() { return 1; }\n"
        )

        graph = parse_codebase(str(tmp_path))
        files = [n.file for n in graph.nodes]
        assert not any("node_modules" in f for f in files)


class TestSaveLoadGraph:
    def test_save_and_load(self, sample_python_project, tmp_path):
        graph = parse_codebase(str(sample_python_project))
        output = str(tmp_path / "output_graph.json")
        path = save_graph(graph, output)

        assert Path(path).exists()

        loaded = load_graph(path)
        assert len(loaded.nodes) == len(graph.nodes)
        assert len(loaded.edges) == len(graph.edges)
        assert loaded.metadata.root == graph.metadata.root

    def test_save_default_path(self, sample_python_project):
        graph = parse_codebase(str(sample_python_project))
        path = save_graph(graph)

        assert Path(path).exists()
        assert path.endswith("graph.json")

        # Cleanup
        Path(path).unlink(missing_ok=True)

    def test_round_trip_preserves_data(self, sample_python_project, tmp_path):
        graph = parse_codebase(str(sample_python_project))
        output = str(tmp_path / "round_trip.json")
        save_graph(graph, output)
        loaded = load_graph(output)

        original_ids = sorted(n.id for n in graph.nodes)
        loaded_ids = sorted(n.id for n in loaded.nodes)
        assert original_ids == loaded_ids
