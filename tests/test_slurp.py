"""Tests for the slurp module."""

import pytest

from fittok.models import (
    KnowledgeGraph,
    GraphNode,
    GraphEdge,
    GraphMetadata,
    NodeType,
    EdgeType,
)
from fittok.slurp import (
    count_tokens,
    pagerank,
    tfidf_scores,
    query_graph,
    _compute_combined_scores,
    _select_nodes,
    _select_neighbors,
    format_subgraph,
)


@pytest.fixture
def sample_graph():
    """Build a small knowledge graph for testing."""
    nodes = [
        GraphNode(
            id="file:auth.py",
            type=NodeType.FILE,
            name="auth.py",
            file="auth.py",
            content="",
        ),
        GraphNode(
            id="function:auth.py:login:0",
            type=NodeType.FUNCTION,
            name="login",
            file="auth.py",
            line_start=1,
            line_end=10,
            content="def login(username, password):\n    user = authenticate(username, password)\n    if user:\n        create_session(user)\n        return token\n    return None",
            token_count=30,
        ),
        GraphNode(
            id="function:auth.py:authenticate:12",
            type=NodeType.FUNCTION,
            name="authenticate",
            file="auth.py",
            line_start=12,
            line_end=20,
            content="def authenticate(username, password):\n    user = db.find_user(username)\n    if user and user.check_password(password):\n        return user\n    return None",
            token_count=28,
        ),
        GraphNode(
            id="function:auth.py:create_session:22",
            type=NodeType.FUNCTION,
            name="create_session",
            file="auth.py",
            line_start=22,
            line_end=30,
            content="def create_session(user):\n    session = Session(user=user)\n    session.save()\n    return session.token",
            token_count=22,
        ),
        GraphNode(
            id="class:auth.py:User:32",
            type=NodeType.CLASS,
            name="User",
            file="auth.py",
            line_start=32,
            line_end=50,
            content="class User:\n    def __init__(self, username, email):\n        self.username = username\n        self.email = email\n\n    def check_password(self, password):\n        return bcrypt.checkpw(password, self.hash)",
            token_count=35,
        ),
    ]
    edges = [
        GraphEdge(source="file:auth.py", target="function:auth.py:login:0", type=EdgeType.CONTAINS),
        GraphEdge(source="file:auth.py", target="function:auth.py:authenticate:12", type=EdgeType.CONTAINS),
        GraphEdge(source="file:auth.py", target="function:auth.py:create_session:22", type=EdgeType.CONTAINS),
        GraphEdge(source="file:auth.py", target="class:auth.py:User:32", type=EdgeType.CONTAINS),
        GraphEdge(source="function:auth.py:login:0", target="function:auth.py:authenticate:12", type=EdgeType.CALLS),
        GraphEdge(source="function:auth.py:login:0", target="function:auth.py:create_session:22", type=EdgeType.CALLS),
        GraphEdge(source="function:auth.py:authenticate:12", target="class:auth.py:User:32", type=EdgeType.REFERENCES),
    ]
    metadata = GraphMetadata(root="/test", total_nodes=5, total_edges=7)
    return KnowledgeGraph(nodes=nodes, edges=edges, metadata=metadata)


class TestTokenCounting:
    def test_count_tokens_basic(self):
        assert count_tokens("hello world") > 0

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0


class TestPageRank:
    def test_pagerank_basic(self, sample_graph):
        scores = pagerank(sample_graph.nodes, sample_graph.edges)
        assert len(scores) == len(sample_graph.nodes)
        assert all(v >= 0 for v in scores.values())

    def test_pagerank_sums_to_one(self, sample_graph):
        scores = pagerank(sample_graph.nodes, sample_graph.edges)
        total = sum(scores.values())
        assert abs(total - 1.0) < 0.01

    def test_pagerank_empty_graph(self):
        scores = pagerank([], [])
        assert scores == {}


class TestTfIdf:
    def test_tfidf_scores_basic(self, sample_graph):
        content_nodes = [n for n in sample_graph.nodes if n.content]
        scores = tfidf_scores(content_nodes, "authentication login")
        assert len(scores) == len(content_nodes)

    def test_tfidf_relevance(self, sample_graph):
        content_nodes = [n for n in sample_graph.nodes if n.content]
        scores = tfidf_scores(content_nodes, "login password")

        # login function should score higher than create_session for this query
        login_score = scores.get("function:auth.py:login:0", 0)
        session_score = scores.get("function:auth.py:create_session:22", 0)
        assert login_score > session_score

    def test_tfidf_empty_nodes(self):
        scores = tfidf_scores([], "query")
        assert scores == {}


class TestCombinedScores:
    def test_combined_scores(self, sample_graph):
        content_nodes = [n for n in sample_graph.nodes if n.content]
        scores = _compute_combined_scores(content_nodes, sample_graph.edges, "login")
        assert len(scores) == len(content_nodes)
        assert all(v >= 0 for v in scores.values())


class TestSelectNodes:
    def test_select_within_budget(self, sample_graph):
        content_nodes = [n for n in sample_graph.nodes if n.content]
        scores = _compute_combined_scores(content_nodes, sample_graph.edges, "login")
        selected = _select_nodes(content_nodes, sample_graph.edges, scores, token_budget=1000)

        assert len(selected) > 0
        assert len(selected) <= len(content_nodes)

    def test_select_tiny_budget(self, sample_graph):
        content_nodes = [n for n in sample_graph.nodes if n.content]
        scores = _compute_combined_scores(content_nodes, sample_graph.edges, "login")
        selected = _select_nodes(content_nodes, sample_graph.edges, scores, token_budget=10)

        # Should select fewer nodes with a tiny budget
        assert len(selected) <= 2


class TestFormatSubgraph:
    def test_format_output(self, sample_graph):
        content_nodes = [n for n in sample_graph.nodes if n.content]
        md = format_subgraph(content_nodes, 4000)
        assert "Relevant code" in md
        assert "login" in md
        assert "```" in md


class TestQueryGraph:
    def test_query_basic(self, sample_graph):
        md, count, tokens = query_graph(sample_graph, "how does login work", 4000)
        assert count > 0
        assert tokens > 0
        assert "login" in md

    def test_query_empty_graph(self):
        graph = KnowledgeGraph(
            nodes=[],
            edges=[],
            metadata=GraphMetadata(root="/test"),
        )
        md, count, tokens = query_graph(graph, "anything", 4000)
        assert count == 0
        assert "No nodes" in md


class TestNeighborExpansion:
    """The core fix for the re-read problem: definitions referenced by selected
    code (a constant like MODEL_NAME, or a called helper) are surfaced as
    compact dependencies, exempt from the relevance cliff but token-capped."""

    @pytest.fixture
    def match_graph(self):
        nodes = [
            GraphNode(
                id="function:m.py:do_match:0", type=NodeType.FUNCTION, name="do_match",
                file="m.py", line_start=1, line_end=5,
                content="def do_match():\n    model = MODEL_NAME\n    return helper(model)",
                token_count=20,
            ),
            GraphNode(
                id="constant:m.py:MODEL_NAME:7", type=NodeType.CONSTANT, name="MODEL_NAME",
                file="m.py", line_start=7, line_end=7,
                content='MODEL_NAME = "llama-3.3-70b"', token_count=10,
            ),
            GraphNode(
                id="function:m.py:helper:9", type=NodeType.FUNCTION, name="helper",
                file="m.py", line_start=9, line_end=12,
                content="def helper(x):\n    return x.strip()", token_count=12,
            ),
            GraphNode(  # defined but never referenced → must NOT be pulled in
                id="function:m.py:unused:14", type=NodeType.FUNCTION, name="unused",
                file="m.py", line_start=14, line_end=16,
                content="def unused():\n    return 0", token_count=8,
            ),
        ]
        return nodes

    def test_pulls_in_referenced_constant_and_helper(self, match_graph):
        selected = [match_graph[0]]
        chosen, md = _select_neighbors(
            selected, match_graph, token_cap=300,
            selected_ids={selected[0].id},
        )
        names = {n.name for n in chosen}
        assert "MODEL_NAME" in names       # the exact symbol that triggered the Copilot re-read
        assert "helper" in names           # called function resolved by name
        assert "unused" not in names       # not referenced → excluded
        assert "Referenced dependencies" in md
        assert "llama-3.3-70b" in md       # constant's value actually surfaced

    def test_token_cap_bounds_output(self, match_graph):
        selected = [match_graph[0]]
        # Cap smaller than even the section header → nothing fits.
        chosen, md = _select_neighbors(
            selected, match_graph, token_cap=2,
            selected_ids={selected[0].id},
        )
        assert chosen == []
        assert md == ""

    def test_respects_exclude_ids(self, match_graph):
        """Previously-returned nodes (cross-call dedup) aren't re-sent."""
        selected = [match_graph[0]]
        chosen, _ = _select_neighbors(
            selected, match_graph, token_cap=300,
            selected_ids={selected[0].id},
            exclude_ids={"constant:m.py:MODEL_NAME:7"},
        )
        names = {n.name for n in chosen}
        assert "MODEL_NAME" not in names   # excluded
        assert "helper" in names           # unaffected

    def test_end_to_end_surfaces_dependencies(self, match_graph):
        """Full query_graph path: a relevant function's referenced constant
        lands in the output under 'Referenced dependencies'."""
        graph = KnowledgeGraph(
            nodes=match_graph,
            edges=[
                GraphEdge(source="file:m.py", target=n.id, type=EdgeType.CONTAINS)
                for n in match_graph
            ],
            metadata=GraphMetadata(root="/test", total_nodes=4, total_edges=4),
        )
        md, _count, _tokens = query_graph(graph, "how does match work", 4000)
        assert "do_match" in md
        assert "Referenced dependencies" in md
        assert "MODEL_NAME" in md
