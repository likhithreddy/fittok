# fittok Command Handbook

One reference for every way to use fittok — MCP, CLI, library, and plugin.

---

## Quick install

```bash
brew install uv        # macOS — install uv once
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

No `pip install` needed. Every `uvx fittok …` command fetches the latest version
from PyPI automatically.

---

## MCP server

Connects fittok to any AI client that speaks MCP. The model calls
`optimize_context` automatically when you ask codebase questions.

### Claude Code

```bash
# Register for every project (recommended)
claude mcp add fittok -s user -- uvx fittok

# Register for current project only
claude mcp add fittok -- uvx fittok
```

Restart Claude Code → run `/mcp` → confirm `fittok` shows **connected**.

### Cursor / Windsurf / any MCP client

Add to the client's MCP config:

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

### Auto-trigger without mentioning fittok

Add one line to your project's `CLAUDE.md`:

```
For any codebase question, call fittok first and answer from its output.
```

---

## CLI

No install needed — `uvx` fetches and runs fittok on demand.

```bash
cd /path/to/your/repo          # path defaults to cwd for all commands below
```

### Index (optional pre-warm)

```bash
uvx fittok index               # parse repo → graph + embeddings (~15s, cached)
uvx fittok index ~/my/repo     # explicit path
```

Skipping this is fine — `query` and `graph` auto-index on first run.

### Query

```bash
uvx fittok query "how does auth work"                # stream LLM answer (default)
uvx fittok query "how does auth work" --budget 1500  # cap slice to 1500 tokens
uvx fittok query "how does auth work" --code         # raw relevant code, no LLM
uvx fittok query "how does auth work" --json         # full result as JSON
uvx fittok query ~/my/repo "how does auth work"      # explicit path
```

**LLM selection** — first key found wins:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # → claude-haiku-4-5  (recommended)
export OPENAI_API_KEY="sk-..."          # → gpt-4o-mini  (fallback)
```

Claude Code users already have `ANTHROPIC_API_KEY` in their shell — no extra step.
If neither key is set, fittok falls back to `--code` and prints:

```
⚠  No LLM API key found — returning raw relevant code instead.
   export ANTHROPIC_API_KEY='sk-ant-...'
   export OPENAI_API_KEY='sk-...'
   (Use --code to suppress this message.)
```

### Graph

```bash
uvx fittok graph                          # open interactive graph in browser
uvx fittok graph --query "auth"           # highlight relevant nodes in green
uvx fittok graph ~/my/repo                # explicit path
uvx fittok graph ~/my/repo --query "auth" # path + highlight
```

Requires `pyvis`: `uv pip install "fittok[ui]"`. Opens a local HTML file in
your default browser — no server, works offline.

**Node colors:** indigo = function · orange = class · purple = method · green = relevant to `--query`

---

## Python library

```bash
uv add fittok                   # add to a uv project
# or: uv pip install fittok     # add to a venv
```

```python
from fittok import optimize, index

# Ask a question — returns the relevant code slice + stats
result = optimize("/path/to/repo", "how does auth work")
print(result["optimized_context"])   # relevant source code
print(result["savings"])             # token reduction stats

# Pre-index a repo (optional)
index("/path/to/repo")

# With explicit token budget
result = optimize("/path/to/repo", "how does auth work", token_budget=1500)
```

---

## Claude Code plugin (auto-inject)

The plugin injects the relevant context *before* Claude sees your prompt — no
MCP call needed, works on every codebase question automatically.

```bash
# Install the plugin
claude plugin install plugin/
```

The plugin hooks `UserPromptSubmit` → runs `fittok query` → injects the relevant
code as additional context. If the question doesn't look like a code question, it
no-ops silently.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables LLM answers via `claude-haiku-4-5` |
| `OPENAI_API_KEY` | — | Fallback LLM via `gpt-4o-mini` |
| `FITTOK_SHOW_SAVINGS` | `true` | `🪙 saved X%` footer on MCP answers |
| `FITTOK_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `FITTOK_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `FITTOK_CACHE_DIR` | `~/.cache/fittok` | Graph + embedding cache location |

---

## Optional extras

```bash
uv pip install "fittok[ui]"    # fittok graph — pyvis browser visualization
uv pip install "fittok[gpu]"   # GPU-accelerated embeddings (CUDA)
```

---

## Cache management

```bash
rm -rf ~/.cache/fittok         # clear graph + embedding cache
uv cache clean fittok          # force re-download of fittok itself
```
