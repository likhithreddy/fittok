"""Data models for the context optimizer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Graph models ──────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    METHOD = "method"
    IMPORT = "import"
    # Module-level assignments (e.g. ``MODEL_NAME = "..."``). Made first-class so
    # the optimizer can surface a referenced constant instead of forcing a re-read.
    CONSTANT = "constant"


class EdgeType(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    REFERENCES = "references"
    CONTAINS = "contains"
    INHERITS = "inherits"


class GraphNode(BaseModel):
    id: str
    type: NodeType
    name: str
    file: str
    line_start: int = 0
    line_end: int = 0
    content: str = Field(default="", description="Truncated source content")
    token_count: int = 0


class GraphEdge(BaseModel):
    source: str
    target: str
    type: EdgeType


class GraphMetadata(BaseModel):
    root: str
    total_nodes: int = 0
    total_edges: int = 0
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    metadata: GraphMetadata


# ── Tool input models ─────────────────────────────────────────────────────────

class ParseCodebaseInput(BaseModel):
    path: str = Field(description="Root directory of the codebase")


class ParseCodebaseOutput(BaseModel):
    graph_json_path: str
    total_nodes: int
    total_edges: int


class QueryGraphInput(BaseModel):
    graph_path: str = Field(description="Path to graph.json")
    query: str = Field(description="Natural language query")
    token_budget: int = Field(default=4000, description="Max tokens for output")


class QueryGraphOutput(BaseModel):
    subgraph_markdown: str
    selected_node_count: int
    tokens_used: int


class CompressContextInput(BaseModel):
    context: str = Field(description="Text to compress")
    question: str = Field(description="Guiding question for compression")
    target_tokens: int = Field(description="Target output token count")
    rate: Optional[float] = Field(
        default=None, description="Compression ratio override"
    )


class CompressContextOutput(BaseModel):
    compressed: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float


class OptimizeContextInput(BaseModel):
    codebase_path: str = Field(description="Root directory")
    query: str = Field(description="User question")
    token_budget: int = Field(default=500, description="Final target tokens")


class OptimizeContextOutput(BaseModel):
    optimized_context: str
    graph_stats: dict = Field(default_factory=dict)
    slurp_stats: dict = Field(default_factory=dict)
    compression_stats: dict = Field(default_factory=dict)
