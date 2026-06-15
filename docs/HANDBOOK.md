# fittok Command Reference

> **fittok** retrieves only the *relevant* source code for a question about a codebase,
> within a token budget, so an LLM can answer from a focused slice instead of reading
> whole files.

This handbook documents every subcommand, argument, environment variable, and exit
code. For a quick-start, see the [README](../README.md).

---

## Table of Contents

1. [Synopsis](#synopsis)
2. [Prerequisites](#prerequisites)
3. [Subcommands](#subcommands)
   - [fittok serve](#fittok-serve)
   - [fittok index](#fittok-index)
   - [fittok query](#fittok-query)
   - [fittok graph](#fittok-graph)
4. [Environment Variables](#environment-variables)
5. [LLM Provider Selection](#llm-provider-selection)
6. [Caching](#caching)
7. [Exit Codes](#exit-codes)
8. [Examples](#examples)

---

## Synopsis

```
fittok <subcommand> [options]

uvx fittok <subcommand> [options]   # zero-install form (recommended)
```

When invoked with no subcommand, fittok defaults to `serve` so that MCP client
registrations (`claude mcp add fittok -- uvx fittok`) continue to work without
modification.

---

## Prerequisites

fittok is distributed on PyPI and runs through
[`uv`](https://docs.astral.sh/uv/) — no manual `pip install` required.

**Install `uv` once:**

```bash
brew install uv                                    # macOS
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux / Windows (WSL)
```

After that, every `uvx fittok …` command fetches the latest release from PyPI
automatically and caches it locally. No activation, no virtualenv management.

**Optional extras:**

| Extra | Installs | When you need it |
|---|---|---|
| `fittok[ui]` | `pyvis`, `gradio` | `fittok graph` — interactive browser visualization |
| `fittok[gpu]` | `torch` (CUDA) | GPU-accelerated embeddings |

```bash
uv pip install "fittok[ui]"    # graph visualization
uv pip install "fittok[gpu]"   # CUDA embeddings
```

---

## Subcommands

---

### fittok serve

**Start the MCP server over stdio.**

```
fittok serve
```

Launches the FastMCP server and listens on stdin/stdout for JSON-RPC messages
from any MCP-compatible client (Claude Code, Cursor, Windsurf, etc.).

This is the default subcommand — running bare `fittok` (or `uvx fittok`) is
equivalent to `fittok serve`.

**Arguments:** none.

**MCP tools exposed:**

| Tool | Description |
|---|---|
| `optimize_context` | Core tool. Given a `codebase_path` and `query`, returns the relevant code slice + savings report. |

**Register in Claude Code:**

```bash
# Current project only
claude mcp add fittok -- uvx fittok

# All projects (user scope — recommended)
claude mcp add fittok -s user -- uvx fittok
```

After adding, restart Claude Code and run `/mcp` to confirm `fittok` shows
**connected**.

**Register in Cursor / Windsurf / any MCP client:**

Add to the client's MCP config JSON:

```json
{
  "mcpServers": {
    "fittok": {
      "command": "uvx",
      "args": ["fittok"]
    }
  }
}
```

**Auto-trigger without mentioning fittok by name:**

Add one line to your project's `CLAUDE.md`:

```
For any codebase question, call fittok first and answer from its output.
```

---

### fittok index

**Pre-build the knowledge graph and embeddings for a codebase.**

```
fittok index [path]
```

Parses every supported source file in `path`, extracts functions / classes /
methods as graph nodes, computes call/import/inheritance edges, and embeds every
node using `all-MiniLM-L6-v2`. The result is cached on disk under
`~/.cache/fittok/` (see [Caching](#caching)).

Running `index` is **optional** — `fittok query` auto-indexes on first run.
Pre-indexing is useful when you want the first query to be instant, or when you
want to warm the cache in CI.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `path` | path | current directory | Root of the codebase to index. Accepts absolute or relative paths. |

**Supported languages:** Python, JavaScript, JSX, TypeScript, TSX, Java, Go, Rust.

**Output (stderr):**

```
Indexed 4039 nodes / 9302 edges, 4039 embeddings (parse 3.2s, embed 11.4s). Cached.
```

**Examples:**

```bash
# Index the current repo
uvx fittok index

# Index a specific repo
uvx fittok index ~/projects/myapp

# Index in CI before running queries
uvx fittok index .
```

---

### fittok query

**Ask a natural-language question about a codebase.**

```
fittok query [path] <question> [--budget N] [--code] [--json]
```

Runs the full fittok pipeline:

1. Loads (or builds) the knowledge graph for `path`.
2. Scores every node by semantic similarity + TF-IDF + PageRank against `question`.
3. Applies the relevance cliff to drop noise.
4. Trims the surviving nodes to a token budget.
5. **Default:** sends the slice to an LLM (see [LLM Provider Selection](#llm-provider-selection)) and streams the answer.
6. **With `--code`:** prints the raw relevant code slice instead.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `path` | path | current directory | Root of the codebase. If omitted and the next argument does not resolve to a path on disk, it is treated as the `question`. |
| `question` | string | required | Natural-language question about the codebase. Quote multi-word questions. |

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--budget N` | integer | `0` (adaptive) | Maximum tokens in the code slice sent to the LLM. `0` auto-sizes between 600 and 3,500 tokens based on how many nodes survive the relevance cliff. Set an explicit value to cap costs or tune answer depth. |
| `--code` | flag | off | Return the raw relevant code slice instead of an LLM answer. Useful for piping into other tools, for debugging what fittok selected, or when no API key is available. Suppresses the "no API key" warning. |
| `--json` | flag | off | Emit the full result as a JSON object (implies `--code`). Includes `optimized_context`, `slurp_stats`, `savings`, and `graph_stats`. |

**Stderr output (always):**

```
# 14 nodes · 2750 tokens · confidence medium · Sent 2750 tokens vs ~13519 (79.7% less)
```

**Stdout (default — LLM answer):**

The answer is streamed token-by-token to stdout so it can be piped or redirected:

```bash
uvx fittok query "how does auth work" > answer.md
```

**Stdout (`--code`):**

A Markdown-formatted block with the relevant source, ordered by relevance:

```
Relevant code

  claimItem() (src/store/board-store.ts:288-337)
  ...
```

**LLM fallback behavior:**

If no LLM API key is set, fittok prints a warning to stderr and falls back to
`--code` output automatically. Set `--code` explicitly to suppress the warning.

```
⚠  No LLM API key found — returning raw relevant code instead.
   To get an LLM answer, set one of:
     export ANTHROPIC_API_KEY='sk-ant-...'
     export OPENAI_API_KEY='sk-...'
   (Use --code to suppress this message and always get raw code.)
```

**Examples:**

```bash
# Ask a question about the current repo (LLM answers)
uvx fittok query "how does authentication work"

# Ask about a specific repo
uvx fittok query ~/projects/myapp "how does the payment flow work"

# Limit the slice to 1,500 tokens before sending to the LLM
uvx fittok query "how does auth work" --budget 1500

# Get raw relevant code instead of an LLM answer
uvx fittok query "how does auth work" --code

# Machine-readable full result
uvx fittok query "how does auth work" --json | jq .savings

# Pipe code into another tool
uvx fittok query "how does auth work" --code | pbcopy
```

---

### fittok graph

**Open an interactive knowledge graph of the codebase in the browser.**

```
fittok graph [path] [--query "..."]
```

Builds (or loads from cache) the knowledge graph for `path` and renders it as
an interactive HTML visualization using `pyvis` / `vis-network`. The file is
written to a system temp directory and opened with `webbrowser.open()`.

Requires the `ui` extra:

```bash
uv pip install "fittok[ui]"
```

If `pyvis` is not installed, fittok prints a clear install hint and exits with
code 1.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `path` | path | current directory | Root of the codebase to visualize. |

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--query TEXT` | string | none | Highlight nodes relevant to this query in green. Uses the same semantic scoring as `fittok query` — the top ~5% of nodes by relevance score are highlighted. Useful for visually tracing how a feature is implemented across the graph. |

**Node color legend:**

| Color | Node type |
|---|---|
| Indigo `#6366f1` | Function / arrow component |
| Orange `#FF9800` | Class |
| Purple `#9C27B0` | Method |
| Gray `#607D8B` | File / module |
| Green `#22c55e` | Relevant to `--query` (if provided) |

**Interaction:**

- **Pan:** click and drag the background.
- **Zoom:** scroll wheel.
- **Hover a node:** shows `type: name · file:line` tooltip.
- **Drag a node:** repositions it; the physics simulation re-stabilizes.

**Examples:**

```bash
# Visualize the current repo
uvx fittok graph

# Visualize a specific repo
uvx fittok graph ~/projects/myapp

# Highlight nodes relevant to authentication
uvx fittok graph --query "how does auth work"

# Combine path and query
uvx fittok graph ~/projects/myapp --query "payment processing"
```

---

## Environment Variables

All variables are read at startup. Set them in your shell profile
(`~/.zshrc`, `~/.bashrc`) or pass them inline:

```bash
FITTOK_SHOW_SAVINGS=false uvx fittok query "..."
```

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key. When set, `fittok query` uses `claude-haiku-4-5` to answer questions. Takes priority over `OPENAI_API_KEY`. |
| `OPENAI_API_KEY` | — | OpenAI API key. Used when `ANTHROPIC_API_KEY` is not set. `fittok query` uses `gpt-4o-mini`. |
| `FITTOK_SHOW_SAVINGS` | `true` | Append a `🪙 saved X%` footer to MCP answers. Set to `false` to disable. |
| `FITTOK_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model used for semantic embeddings. Change only if you have a specific model in mind — the default is fast and accurate for code. |
| `FITTOK_DEVICE` | `auto` | Embedding compute device. `auto` tries CUDA → Apple MPS → CPU. Override with `cuda`, `mps`, or `cpu`. |
| `FITTOK_CACHE_DIR` | `~/.cache/fittok` | Root directory for all cached graphs and embeddings. Override to put the cache on a faster disk or a shared network path. |
| `FITTOK_MAX_NODE_CHARS` | `8000` | Maximum characters of source stored per graph node. Nodes larger than this are truncated before embedding. |

---

## LLM Provider Selection

`fittok query` (without `--code`) sends the relevant code slice to an LLM and
streams the answer. The provider is selected automatically:

```
ANTHROPIC_API_KEY set  →  claude-haiku-4-5   (fast, cheap, default)
OPENAI_API_KEY set     →  gpt-4o-mini
neither set            →  falls back to --code + prints setup hint
```

**Setup (one-time):**

```bash
# Add to ~/.zshrc or ~/.bashrc
export ANTHROPIC_API_KEY="sk-ant-..."   # recommended — Claude Code users already have this
# or
export OPENAI_API_KEY="sk-..."
```

Users of Claude Code typically already have `ANTHROPIC_API_KEY` set in their
shell environment from Claude Code's own configuration — no extra step needed.

The LLM receives:
- A system prompt instructing it to answer from the provided code only.
- The optimized code slice as context.
- The original question.

The answer is streamed to stdout. Slice stats (nodes, tokens, savings) are always
printed to stderr so they don't pollute piped output.

---

## Caching

fittok caches aggressively so repeated queries on the same repo are instant.

| Cache type | Location | Key | TTL |
|---|---|---|---|
| Knowledge graph | `~/.cache/fittok/graphs/<name>-<hash>.json` | Repo path hash | Until invalidated |
| Embeddings | In-process dict (SHA-256 of node content) | Content hash | Process lifetime |

The graph is **content-keyed** — if a file changes, only the nodes from that
file are re-parsed and re-embedded on the next query. Unchanged files reuse
cached embeddings.

**Clear the cache:**

```bash
rm -rf ~/.cache/fittok      # full reset
uv cache clean fittok       # clear uvx tool cache (forces re-download of fittok itself)
```

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Error — message printed to stderr (parse failure, missing `pyvis`, API error, etc.) |

---

## Examples

### Ask a question and get an LLM answer

```bash
cd ~/projects/myapp
uvx fittok query "how does the websocket connection stay alive"
```

### Pre-warm then query (fastest cold start)

```bash
uvx fittok index ~/projects/myapp
uvx fittok query ~/projects/myapp "how does rate limiting work"
```

### Inspect what fittok selected before trusting the answer

```bash
uvx fittok query "how does auth work" --code | less
```

### Limit cost on a large codebase

```bash
uvx fittok query "walk me through the payment flow" --budget 2000
```

### Visualize the full graph

```bash
uv pip install "fittok[ui]"
uvx fittok graph
```

### Visualize and highlight a feature

```bash
uvx fittok graph --query "real-time synchronization"
```

### Machine-readable output for scripting

```bash
uvx fittok query "how does auth work" --json \
  | jq '{nodes: .slurp_stats.selected_nodes, saved: .savings.reduction_pct}'
```

### Use fittok as a Python library

```python
from fittok import optimize

result = optimize("/path/to/repo", "how does authentication work", token_budget=1500)
print(result["optimized_context"])   # the code slice
print(result["savings"])             # reduction stats
```
