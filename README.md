# fittok

**Retrieve only the relevant source code for a question — instead of the model
reading whole files — so an LLM answers codebase questions on a small, focused
slice of context.** Less input = fewer tokens, lower cost, faster answers.

Works three ways from one install: an **MCP server**, a **CLI**, and a **Python
library** — plus a **Claude Code plugin** that injects context automatically.

📖 **[Full command reference → docs/HANDBOOK.md](docs/HANDBOOK.md)**

---

## How it works

```
codebase ──▶ graphify ──▶ slurp ──▶ readable slice ──▶ LLM answers
             (parse)      (select)   (trim to budget)
```

1. **graphify** — parses the repo with tree-sitter into a knowledge graph of
   functions / classes / methods (Python, JS, JSX, TS, TSX, Java, Go, Rust).
2. **slurp** — scores every node against the question with **semantic embeddings
   + TF-IDF + PageRank**, then selects *only* the genuinely relevant nodes via a
   relevance cliff (no budget-padding with noise).
3. **readable output** — returns the **actual source code** of those nodes,
   top-ranked in full and the supporting tail as signatures, trimmed to a budget.
   The model answers directly from it.

Graphs and embeddings are cached on disk (`~/.cache/fittok`), keyed by content —
so after a code change only the changed functions re-embed.

---

## Install & use

fittok runs through **[`uv`](https://docs.astral.sh/uv/)** — one tool for everything
below, with no manual `pip install`. Install it once:

```bash
brew install uv                                  # macOS
# or any OS:  curl -LsSf https://astral.sh/uv/install.sh | sh
```

### MCP server — Claude Code / Cursor / Windsurf

**Claude Code:**
```bash
claude mcp add fittok -s user -- uvx fittok
```
Restart Claude Code → `/mcp` → confirm `fittok` is **connected**. Then ask
codebase questions normally — fittok fires automatically.

**Cursor / Windsurf / any MCP client:**
```json
{ "mcpServers": { "fittok": { "command": "uvx", "args": ["fittok"] } } }
```

To make fittok trigger **without mentioning it by name**, add to `CLAUDE.md`:
> *"For any codebase question, call fittok first and answer from its output."*

### CLI

```bash
cd /path/to/your/repo

uvx fittok index                                      # optional pre-warm (~15s, cached)
uvx fittok query "how does auth work"                 # LLM answers from relevant code
uvx fittok query "how does auth work" --budget 1500   # cap the slice at 1500 tokens
uvx fittok query "how does auth work" --code          # raw relevant code, no LLM
uvx fittok graph                                      # interactive browser graph
uvx fittok graph --query "auth"                       # graph with relevant nodes highlighted
```

`query` sends the relevant code slice to an LLM and streams the answer.
Set one key in your shell and it just works:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # → claude-haiku-4-5  (recommended)
export OPENAI_API_KEY="sk-..."          # → gpt-4o-mini  (fallback)
```

Users of Claude Code already have `ANTHROPIC_API_KEY` set — no extra step needed.
If neither key is set, fittok falls back to `--code` and prints a setup hint.

`graph` requires `pyvis`: `uv pip install "fittok[ui]"`.

### Python library

```bash
uv add fittok            # in a uv project   (or:  uv pip install fittok  in a venv)
```
```python
from fittok import optimize

result = optimize("/path/to/repo", "how does authentication work", token_budget=1500)
print(result["optimized_context"])   # the relevant code slice
print(result["savings"])             # token reduction stats
```

---

## Token savings — honest numbers

On a real Next.js/TS repo (~5k functions), fittok returns a **~1.5–3.5k-token
slice** instead of the model reading **15–20k+ tokens** of files — an **~80–90%
reduction on input**, deterministic and reported in the `savings` footer.

On Opus 4.8, a broad question cost **~84k total tokens without fittok vs ~27k
with it** — because fittok replaced a 58k-token Explore subagent with one tool call.

**How to measure it honestly:**
- Use the **`🪙 saved X%` footer** or your **API bill** (total tokens).
- Do *not* judge by Claude Code's `/context` Messages number — it excludes
  subagent tokens and is dominated by model reasoning, which fittok doesn't touch.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables LLM answers via `claude-haiku-4-5` |
| `OPENAI_API_KEY` | — | Fallback LLM via `gpt-4o-mini` |
| `FITTOK_SHOW_SAVINGS` | `true` | `🪙 saved X%` footer on MCP answers; set `false` to disable |
| `FITTOK_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `FITTOK_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `FITTOK_CACHE_DIR` | `~/.cache/fittok` | Cache location |

Full reference: **[docs/HANDBOOK.md](docs/HANDBOOK.md)**

---

## Requirements

Python ≥ 3.10. First run downloads a ~90 MB embedding model. Optional extras:
- `uv pip install "fittok[ui]"` — graph visualizer (`fittok graph`)
- `uv pip install "fittok[gpu]"` — torch/CUDA for GPU-accelerated embeddings

## License

MIT
