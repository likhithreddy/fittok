"""Tests for v0.2.0 server tools: batch, structured, diff, PII, cache, streaming."""

from unittest.mock import patch, MagicMock

from fittok.server import (
    optimize_context_batch,
    optimize_context_structured,
    diff_graph_tool,
    scrub_text_tool,
    scrub_file_tool,
    list_pii_patterns_tool,
    add_pii_pattern_tool,
    clear_cache_tool,
    cache_stats_tool,
    reset_graph_tool,
    get_graph_stats_tool,
)


def _make_project(tmp_path):
    """Create a sample project for testing."""
    (tmp_path / "auth.py").write_text(
        "def login(user, pwd):\n    return authenticate(user, pwd)\n\n"
        "def authenticate(user, pwd):\n    return True\n"
    )
    return str(tmp_path)


class TestMultiQueryBatch:
    @patch("fittok.llmlingua_wrapper._get_compressor")
    def test_batch_multiple_queries(self, mock_get, tmp_path):
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.return_value = {"compressed_prompt": "short"}
        mock_get.return_value = mock_compressor

        project = _make_project(tmp_path)
        result = optimize_context_batch(
            codebase_path=project,
            queries=["How does login work?", "What functions exist?"],
            token_budget=50,
        )
        assert "graph_stats" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["query"] == "How does login work?"

    def test_batch_invalid_path(self):
        result = optimize_context_batch("/nonexistent", ["q"])
        assert "error" in result

    def test_batch_empty_queries(self, tmp_path):
        project = _make_project(tmp_path)
        result = optimize_context_batch(project, [])
        assert "error" in result


class TestStructuredOutput:
    @patch("fittok.llmlingua_wrapper._get_compressor")
    def test_json_format(self, mock_get, tmp_path):
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.return_value = {"compressed_prompt": "auth info"}
        mock_get.return_value = mock_compressor

        project = _make_project(tmp_path)
        result = optimize_context_structured(
            codebase_path=project,
            query="login",
            token_budget=50,
            output_format="json",
        )
        assert "answer" in result
        assert "supporting_nodes" in result
        assert "graph_stats" in result

    @patch("fittok.llmlingua_wrapper._get_compressor")
    def test_markdown_format(self, mock_get, tmp_path):
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.return_value = {"compressed_prompt": "md output"}
        mock_get.return_value = mock_compressor

        project = _make_project(tmp_path)
        result = optimize_context_structured(
            codebase_path=project,
            query="login",
            token_budget=50,
            output_format="markdown",
        )
        assert "optimized_context" in result


class TestDiffGraphTool:
    def test_diff_same_graph(self, tmp_path):
        from fittok.graphify import parse_codebase, save_graph
        project = _make_project(tmp_path)
        graph = parse_codebase(project)
        p = str(tmp_path / "graph.json")
        save_graph(graph, p)
        result = diff_graph_tool(p, p)
        assert "No changes detected" in result["summary"]

    def test_diff_different_graphs(self, tmp_path):
        from fittok.graphify import parse_codebase, save_graph
        from fittok.models import KnowledgeGraph, GraphMetadata

        project = _make_project(tmp_path)
        graph_a = parse_codebase(project)
        p_a = str(tmp_path / "graph_a.json")
        save_graph(graph_a, p_a)

        empty = KnowledgeGraph(nodes=[], edges=[], metadata=GraphMetadata(root=project))
        p_b = str(tmp_path / "graph_b.json")
        save_graph(empty, p_b)

        result = diff_graph_tool(p_a, p_b)
        assert len(result["nodes_removed"]) > 0

    def test_diff_missing_file(self):
        result = diff_graph_tool("/missing/a.json", "/missing/b.json")
        assert "error" in result


class TestPIIScrubTools:
    def test_scrub_text(self):
        result = scrub_text_tool("Email: user@test.com")
        assert result["count"] >= 1

    def test_scrub_file(self, tmp_path):
        (tmp_path / "s.txt").write_text("key=AKIAIOSFODNN7EXAMPLE")
        result = scrub_file_tool(str(tmp_path / "s.txt"))
        assert result["count"] >= 1

    def test_list_patterns(self):
        result = list_pii_patterns_tool()
        assert result["count"] >= 8

    def test_add_pattern(self):
        result = add_pii_pattern_tool("custom", r"CUSTOM-\d+")
        assert result["added"] is True


class TestCacheTools:
    @patch("fittok.cache._get_cache")
    def test_cache_stats(self, mock_get):
        mock_get.return_value = None
        result = cache_stats_tool()
        assert "available" in result

    @patch("fittok.cache._get_cache")
    def test_clear_cache(self, mock_get):
        mock_get.return_value = None
        result = clear_cache_tool()
        assert "error" in result  # cache not available


class TestResetAndStats:
    def test_reset_graph(self, tmp_path):
        project = _make_project(tmp_path)
        result = reset_graph_tool(project)
        assert result["reset"] is True
        assert result["total_nodes"] > 0

    def test_get_graph_stats(self, tmp_path):
        from fittok.graphify import parse_codebase, save_graph
        project = _make_project(tmp_path)
        graph = parse_codebase(project)
        p = str(tmp_path / "graph.json")
        save_graph(graph, p)
        result = get_graph_stats_tool(p)
        assert result["total_nodes"] > 0
        assert "node_types" in result
        assert "edge_types" in result

    def test_get_stats_missing(self):
        result = get_graph_stats_tool("/missing/graph.json")
        assert "error" in result
