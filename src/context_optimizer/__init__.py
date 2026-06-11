"""Context Optimizer — relevant-code retrieval within a token budget.

Usable three ways, each standalone:
  - As a library:   ``from context_optimizer import optimize``
  - As a CLI:       ``context-optimizer query <path> "<question>"``
  - As an MCP server: ``python -m context_optimizer`` (for AI clients)
"""

__version__ = "0.3.0"

__all__ = ["optimize", "index", "__version__"]


def optimize(codebase_path: str, query: str, token_budget: int = 0) -> dict:
    """Return the most relevant source code for *query*, within *token_budget*.

    token_budget=0 → adaptive sizing. No MCP client required. Returns a dict with
    ``optimized_context``, ``graph_stats``, ``slurp_stats`` and ``savings``.
    """
    from .server import optimize_context_tool
    return optimize_context_tool(codebase_path, query, token_budget)


def index(codebase_path: str) -> dict:
    """Pre-build + cache the graph and embeddings for a codebase."""
    from .indexer import index_codebase
    return index_codebase(codebase_path)
