"""Regression tests for the robustness overhaul (parser, fallback, diagnostics)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fittok import embeddings
from fittok.graphify import parse_codebase
from fittok.models import NodeType
from fittok.slurp import query_graph


def _write(d: Path, name: str, content: str) -> None:
    (d / name).write_text(content)


def test_arrow_function_component_gets_real_name():
    """`const Foo = () => {}` must be named 'Foo', not its parameters."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "comp.jsx", "const LoginButton = (props) => { return null; };\n")
        graph = parse_codebase(str(d))
        names = {n.name for n in graph.nodes if n.type != NodeType.FILE}
        assert "LoginButton" in names, f"got names: {names}"


def test_file_without_definitions_is_indexed():
    """A file that yields no functions/classes/constants must still be
    represented via its body fallback. (Assignment-only files no longer hit
    this path — they now produce real CONSTANT nodes, which is strictly better
    and covered by test_module_level_constants_become_nodes.)"""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # A comment + bare call: nothing extractable into a node.
        _write(d, "notes.py", "# just a comment\nprint('bootstrapping')\n")
        graph = parse_codebase(str(d))
        file_nodes = [n for n in graph.nodes if n.type == NodeType.FILE]
        assert file_nodes, "no file node created"
        assert any(n.content and n.content.strip() for n in file_nodes), \
            "definition-less file was not indexed by body"


def test_query_diagnostics_shape():
    """with_diagnostics returns method + confidence + node scores."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "auth.py", "def login(user, pw):\n    return user and pw\n")
        graph = parse_codebase(str(d))
        res = query_graph(graph, "how does login work", token_budget=2000,
                          with_diagnostics=True)
        assert set(res) >= {"markdown", "selected_nodes", "tokens_used",
                            "method", "confidence", "confidence_label", "top_nodes"}
        assert res["method"] in ("semantic+lexical", "lexical", "hybrid (RRF)", "bm25+pagerank (RRF)", "hybrid+map (RRF)", "bm25+map+pagerank (RRF)")
        assert 0.0 <= res["confidence"] <= 1.0


def test_query_backward_compatible_tuple():
    """Default call still returns the (markdown, count, tokens) 3-tuple."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "auth.py", "def login(user, pw):\n    return user and pw\n")
        graph = parse_codebase(str(d))
        result = query_graph(graph, "login", token_budget=2000)
        assert isinstance(result, tuple) and len(result) == 3


@pytest.mark.skipif(not embeddings.is_available(),
                    reason="embedding model unavailable (offline)")
def test_semantic_beats_keyword_on_paraphrase():
    """A query whose words are absent from the code still ranks the right node."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "ws.py",
               "class StreamingSession:\n"
               "    '''Live websocket transport for model token streaming.'''\n"
               "    async def stream_tokens(self, prompt):\n        return prompt\n")
        _write(d, "color.py",
               "def hex_to_rgb(c):\n    return (0, 0, 0)\n")
        graph = parse_codebase(str(d))
        res = query_graph(graph, "real-time conversation with the AI model",
                          token_budget=4000, with_diagnostics=True)
        top_id = res["top_nodes"][0]["id"]
        assert "ws.py" in top_id, f"expected ws.py node on top, got {top_id}"
