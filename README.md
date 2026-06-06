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

## What's New in v0.2.0

| Feature | Description |
|---------|-------------|
| **Streaming Output** | Stage-by-stage progress events via `optimize_context_stream` |
| **Watch Mode** | Incremental graph updates with file watcher (`watch_start` / `watch_stop`) |
| **Batch Boosting** | O(n log n) neighbor selection in slurp (3-phase approach) |
| **Chunked Parsing** | Batch file parsing with progress events for large codebases |
| **GPU Acceleration** | CUDA support with auto-detection for LLMLingua compression |
| **3-Level Cache** | Persistent graph/query/compression cache via diskcache |
| **Web UI** | Interactive graph visualization with Gradio + pyvis |
| **Graph Diffing** | Compare two graphs to see structural changes |
| **Multi-Query** | One parse, many queries with `optimize_context_batch` |
| **PII Scrubbing** | Detect and redact secrets, API keys, emails before processing |
| **Structured Output** | JSON output mode with supporting nodes and relevance scores |

## Installation

```bash
# From source
pip install -e .

# With all extras
pip install -e ".[dev,gpu,ui]"
```

### Requirements

- Python 3.9+
- 4GB RAM minimum
- 8GB VRAM optional (for GPU-accelerated compression)

### Model Configuration

LLMLingua defaults to `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (~500MB, CPU-friendly).

Override via environment variable:

```bash
export CONTEXT_OPTIMIZER_MODEL="microsoft/phi-2"  # GPU recommended
export CONTEXT_OPTIMIZER_DEVICE=auto  # auto | cuda | cpu
python -m context_optimizer
```

Or programmatically:

```python
from context_optimizer.llmlingua_wrapper import compress_context
result = compress_context(text, "question", target_tokens=500, model="microsoft/phi-2")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_OPTIMIZER_MODEL` | bert-base-multilingual | CPU model name |
| `CONTEXT_OPTIMIZER_MODEL_GPU` | bert-base-multilingual | GPU model name |
| `CONTEXT_OPTIMIZER_DEVICE` | auto | Device: auto, cuda, cpu |
| `CONTEXT_OPTIMIZER_SCRUB` | false | Enable PII scrubbing in pipeline |
| `CONTEXT_OPTIMIZER_CACHE_DIR` | ~/.cache/fittok | Cache directory |
| `CONTEXT_OPTIMIZER_CACHE_MAX_MB` | 500 | Max cache size in MB |

## Usage

### As an MCP Server (Claude Code, etc.)

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

### Multi-Query Batching

```python
from context_optimizer.server import optimize_context_batch

result = optimize_context_batch(
    codebase_path="/path/to/code",
    queries=["How does auth work?", "What is the entry point?"],
    token_budget=500,
)
for r in result["results"]:
    print(f"Q: {r['query']}\nA: {r['optimized_context']}\n")
```

### PII Scrubbing

```python
from context_optimizer.pii_scrubber import scrub_text

result = scrub_text("Contact admin@company.com with key AKIAIOSFODNN7EXAMPLE")
print(result["scrubbed"])
# "Contact [REDACTED_EMAIL] with key [REDACTED_AWS_ACCESS_KEY]"
```

## MCP Tools (v0.1.0)

| Tool | Description |
|------|-------------|
| `parse_codebase` | Parse code into a knowledge graph |
| `query_graph` | Query graph for relevant subgraph |
| `compress_context` | Compress text using LLMLingua |
| `optimize_context` | Full pipeline: parse → query → compress |

## MCP Tools (v0.2.0 — new)

| Tool | Description |
|------|-------------|
| `optimize_context_stream` | Streaming pipeline with stage-by-stage progress |
| `optimize_context_batch` | One parse, many queries |
| `optimize_context_structured` | JSON structured output with supporting nodes |
| `parse_codebase_stream` | Chunked parsing with progress events |
| `watch_start` / `watch_stop` | Incremental graph updates via file watcher |
| `get_graph_stats` | Graph metadata and type distribution |
| `reset_graph` | Force full re-parse, ignoring cache |
| `diff_graph` | Compare two knowledge graphs |
| `scrub_text` / `scrub_file` | PII detection and redaction |
| `list_pii_patterns` / `add_pii_pattern` | Manage PII patterns |
| `clear_cache` / `cache_stats` | Cache management |
| `launch_ui` | Launch web visualization dashboard |

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
│   ├── llmlingua_wrapper.py   # Compression wrapper (CPU + GPU)
│   ├── models.py              # Pydantic data models
│   ├── tokens.py              # Shared token counting
│   ├── cache.py               # 3-level persistent cache
│   ├── diff.py                # Graph diffing
│   ├── pii_scrubber.py        # PII detection & redaction
│   ├── watcher.py             # File watcher for incremental updates
│   └── ui.py                  # Web visualization (Gradio + pyvis)
├── tests/
│   ├── test_graphify.py
│   ├── test_slurp.py
│   ├── test_llmlingua.py
│   ├── test_server.py
│   ├── test_server_v2.py
│   ├── test_cache.py
│   ├── test_diff.py
│   └── test_pii_scrubber.py
├── examples/
│   └── usage.py
├── README.md
└── LICENSE
```

## License

MIT
