"""Tests for the MCP server tools."""

from unittest.mock import patch, MagicMock

from context_optimizer.server import (
    parse_codebase_tool,
    query_graph_tool,
    compress_context_tool,
    optimize_context_tool,
)


def _make_sample_graph(tmp_path):
    """Create a sample project and return its graph path."""
    (tmp_path / "app.py").write_text(
        "def hello():\n    return 'hello'\n\ndef world():\n    return hello() + ' world'\n"
    )
    from context_optimizer.graphify import parse_codebase, save_graph

    graph = parse_codebase(str(tmp_path))
    path = save_graph(graph, str(tmp_path / "graph.json"))
    return path


class TestParseCodebaseTool:
    def test_parse_valid_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("def foo(): pass\n")

        result = parse_codebase_tool(str(tmp_path))
        assert "graph_json_path" in result
        assert result["total_nodes"] > 0

    def test_parse_invalid_directory(self):
        result = parse_codebase_tool("/nonexistent")
        assert "error" in result


class TestQueryGraphTool:
    def test_query_valid_graph(self, tmp_path):
        graph_path = _make_sample_graph(tmp_path)

        result = query_graph_tool(graph_path, "hello function", 4000)
        assert "subgraph_markdown" in result
        assert result["selected_node_count"] > 0

    def test_query_missing_graph(self):
        result = query_graph_tool("/missing/graph.json", "query")
        assert "error" in result


class TestCompressContextTool:
    @patch("context_optimizer.llmlingua_wrapper._get_compressor")
    def test_compress_success(self, mock_get):
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.return_value = {
            "compressed_prompt": "short"
        }
        mock_get.return_value = mock_compressor

        result = compress_context_tool("long text " * 50, "summary", 10)
        assert result["compressed"] == "short"

    def test_compress_empty(self):
        result = compress_context_tool("", "question", 10)
        assert result["compressed"] == ""


class TestOptimizeContextTool:
    def test_full_pipeline(self, tmp_path):
        (tmp_path / "app.py").write_text(
            "def greet(name):\n    return f'Hello {name}'\n"
        )

        with patch("context_optimizer.llmlingua_wrapper._get_compressor") as mock_get:
            mock_compressor = MagicMock()
            mock_compressor.compress_prompt.return_value = {
                "compressed_prompt": "greet function"
            }
            mock_get.return_value = mock_compressor

            result = optimize_context_tool(str(tmp_path), "how does greet work", 50)
            assert "optimized_context" in result
            assert "graph_stats" in result
            assert result["graph_stats"]["total_nodes"] > 0

    def test_invalid_path(self):
        result = optimize_context_tool("/nonexistent", "query")
        assert "error" in result
