# fittok

**Retrieve only the relevant source code for a question — instead of the model
reading whole files — so an LLM answers codebase questions on a small, focused
slice of context.** Less input = fewer tokens, lower cost, faster answers.

Works three ways from one install: an **MCP server**, a **CLI**, and a **Python
library** — plus a **Claude Code plugin** that injects context automatically.

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

> Note: an earlier design compressed the slice with LLMLingua, but that produced
> unreadable token-salad the model ignored (then re-read the files). fittok
> returns **real, readable code** instead. LLMLingua remains available only as the
> standalone `compress_context` tool.

Graphs and embeddings are cached on disk (`~/.cache/fittok`), keyed by content —
so after a code change only the changed functions re-embed.

---

## Install & use

### As an MCP server (recommended — for Claude Code / Cursor)
Add one entry to your client's MCP config:
```json
{ "mcpServers": { "fittok": { "command": "uvx", "args": ["fittok"] } } }
```
Then ask codebase questions normally. To make it trigger **without mentioning it**,
add one line to your client's `CLAUDE.md`:
> *"For any codebase question, call fittok first and answer from its output."*

### As a CLI (no MCP needed)
```bash
pipx install fittok                       # recommended global install (macOS/Homebrew-safe)
# or, inside a venv:   python3 -m venv .venv && .venv/bin/pip install fittok
# or, plain pip on an unmanaged Python:   pip install fittok
fittok index <repo>                       # optional one-time pre-warm
fittok query <repo> "how does auth work"  # prints the relevant code slice
```
> On Homebrew / newer Debian **system** Python, a global `pip install` is blocked
> (PEP 668). Use `pipx`, a venv, or `uvx fittok` (no install) — not bare `pip`.

### As a library
```python
from fittok import optimize
result = optimize("/path/to/repo", "how does authentication work")
print(result["optimized_context"])
```

First query on a repo auto-indexes (~15s once, cached); after that it's instant.

---

## Token savings — honest numbers

fittok cuts the **input/exploration cost** of a codebase question. On a real
Next.js/TS repo (~5k functions) it returns a **~1.5–3.5k-token slice** instead of
the model reading **15–20k+ tokens** of files — an **~80–90% reduction on input**,
deterministic and reported in the tool's `savings` footer.

**How to measure it honestly:**
- ✅ Use the **`savings` footer** (e.g. `84% — 2,494 vs 15,631 tokens`) or your
  **API bill** (total tokens — which counts the subagent crawls fittok avoids).
- ⚠️ Do **not** judge by Claude Code's `/context` "Messages" number — it excludes
  subagent tokens and is dominated by the model's own reasoning, which fittok
  doesn't touch. On thorough models the real saving (e.g. ~84k → ~27k total
  tokens, by avoiding an Explore subagent) is invisible there but clear on the bill.

**Where it shines:** broad / multi-file questions, large files, unfamiliar repos,
and thorough models that would otherwise explore heavily. On a tiny question a
capable model can answer from one small file, so the win is marginal there.

---

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `FITTOK_SHOW_SAVINGS` | `false` | Append a `🪙 saved X%` footer to answers |
| `CONTEXT_OPTIMIZER_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `CONTEXT_OPTIMIZER_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `CONTEXT_OPTIMIZER_CACHE_DIR` | `~/.cache/fittok` | Cache location |

## Requirements
Python ≥ 3.10. First run downloads a ~90 MB embedding model. Optional extras:
`pip install "fittok[ui]"` (graph visualizer), `"fittok[gpu]"` (torch/CUDA).

## License
MIT.
