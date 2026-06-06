# Context Optimizer

A standalone MCP server that filters and compresses context before it reaches the LLM, reducing token consumption by 80–90%.

## How It Works

```
[User Query + Files]
        │
        ▼
┌──────────────────────────────────────┐
│     MCP Server: context-optimizer    │
│                                      │
│  ┌──────────┐                        │
│  │ Graphify │ → parse code into      │
│  └────┬─────┘   knowledge graph      │
│       │                               │
│  ┌────▼─────┐                        │
│  │  slurp   │ → select relevant      │
│  └────┬─────┘   subgraph (budget)    │
│       │                               │
│  ┌────▼──────┐                       │
│  │LLMLingua │ → compress to target   │
│  └──────────┘   token count          │
└──────────────────────────────────────┘
        │
        ▼
[Compressed Context] → Send to LLM
```

**Graphify** parses code into a knowledge graph (nodes = functions/classes/files, edges = imports/calls/references).

**slurp** queries the graph using PageRank + TF-IDF to select the most relevant nodes within a token budget.

**LLMLingua** performs final compression using a small local model.

## Installation

```bash
# From source
git clone <repo-url> context-optimizer
cd context-optimizer
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- 4GB RAM minimum
- 8GB VRAM optional (for GPU-accelerated LLMLingua compression)

### Model Configuration

LLMLingua defaults to `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (~500MB, CPU-friendly).

Override via environment variable:

```bash
export CONTEXT_OPTIMIZER_MODEL="microsoft/phi-2"  # GPU recommended
python -m context_optimizer
```

Or programmatically:

```python
from context_optimizer.llmlingua_wrapper import compress_context
result = compress_context(text, "question", target_tokens=500, model="microsoft/phi-2")
```

## Usage

### As an MCP Server (Claude Code, etc.)

Add to your Claude Code `settings.json`:

```json
{
  "mcpServers": {
    "context-optimizer": {
      "command": "python",
      "args": ["-m", "context_optimizer"]
    }
  }
}
```

Or run standalone:

```bash
python -m context_optimizer
```

Or run standalone:

```bash
python -m context_optimizer.server
```

### As a Python Library

```python
from context_optimizer.graphify import parse_codebase, save_graph
from context_optimizer.slurp import query_graph
from context_optimizer.llmlingua_wrapper import compress_context

# Step 1: Parse codebase
graph = parse_codebase("/path/to/code")
save_graph(graph, "graph.json")

# Step 2: Query for relevant subgraph
markdown, count, tokens = query_graph(graph, "How does auth work?", token_budget=4000)

# Step 3: Compress
result = compress_context(markdown, "How does auth work?", target_tokens=500)
print(result["compressed"])
```

### One-call Pipeline

```python
from context_optimizer.server import optimize_context_tool

result = optimize_context_tool(
    codebase_path="/path/to/code",
    query="How does authentication work?",
    token_budget=500,
)
print(result["optimized_context"])
```

## MCP Tools

### `parse_codebase`

Parse all code files in a directory into a knowledge graph.

| Parameter | Type   | Description                  |
|-----------|--------|------------------------------|
| `path`    | string | Root directory of the codebase |

Returns: `graph_json_path`, `total_nodes`, `total_edges`

### `query_graph`

Query the graph for the most relevant subgraph within a token budget.

| Parameter     | Type   | Default | Description                     |
|---------------|--------|---------|---------------------------------|
| `graph_path`  | string | —       | Path to `graph.json`            |
| `query`       | string | —       | Natural language query          |
| `token_budget`| int    | 4000    | Max tokens for output subgraph  |

Returns: `subgraph_markdown`, `selected_node_count`, `tokens_used`

### `compress_context`

Compress text using LLMLingua with a local model.

| Parameter       | Type        | Default | Description                    |
|-----------------|-------------|---------|--------------------------------|
| `context`       | string      | —       | Text to compress               |
| `question`      | string      | —       | Guiding question               |
| `target_tokens` | int         | 500     | Target output token count      |
| `rate`          | float/null  | null    | Compression ratio override     |

Returns: `compressed`, `original_tokens`, `compressed_tokens`, `compression_ratio`

### `optimize_context`

Full pipeline: parse → query → compress in one call.

| Parameter       | Type   | Default | Description               |
|-----------------|--------|---------|---------------------------|
| `codebase_path` | string | —       | Root directory            |
| `query`         | string | —       | User question             |
| `token_budget`  | int    | 500     | Final target tokens       |

Returns: `optimized_context` + stats for each stage

## Supported Languages

- Python
- JavaScript / TypeScript
- Java
- Go
- Rust

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Architecture

```
context-optimizer/
├── pyproject.toml
├── src/context_optimizer/
│   ├── __init__.py
│   ├── server.py              # MCP server (FastMCP)
│   ├── graphify.py            # Code → knowledge graph
│   ├── slurp.py               # Graph query engine
│   ├── llmlingua_wrapper.py   # Compression wrapper
│   └── models.py              # Pydantic data models
├── tests/
│   ├── test_graphify.py
│   ├── test_slurp.py
│   ├── test_llmlingua.py
│   └── test_server.py
├── examples/
│   └── usage.py
├── README.md
└── LICENSE
```

## Future Enhancements

- [ ] Streaming support
- [ ] Incremental graph updates (watch mode)
- [ ] Batch neighbor boosting in slurp (avoid O(n²) re-sort per selection)
- [ ] Chunked/streamed parsing for large codebases (10k+ files)
- [ ] More language support
- [ ] GPU-accelerated compression
- [ ] Caching layer for repeated queries
- [ ] Web UI for visualization

## License

MIT
