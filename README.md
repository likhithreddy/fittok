# fittok

**Retrieve only the relevant source code for a question — instead of the model
reading whole files — so an LLM answers codebase questions on a small, focused
slice of context.** Less input = fewer tokens, lower cost, faster answers.

Works three ways from one install: an **MCP server**, a **CLI**, and a **Python
library** — plus a **Claude Code plugin** that injects context automatically.

📖 **[Full command reference → docs/HANDBOOK.md](https://github.com/likhithreddy/fittok/blob/main/docs/HANDBOOK.md)**

<!-- MCP Registry: PyPI ownership verification — do not remove this line -->
mcp-name: io.github.likhithreddy/fittok

---

## How it works

```
codebase ──▶ graphify ──▶ slurp ──▶ readable slice ──▶ LLM answers
             (parse)      (select)   (trim to budget)
```

1. **graphify** — parses the repo with tree-sitter into a knowledge graph of
   functions / classes / methods (Python, JS, JSX, TS, TSX, Java, Go, Rust).
   Supports multi-language call/import/reference edges.
2. **slurp** — scores every node against the question using a **4-signal hybrid**:
   - **Semantic embeddings** (all-MiniLM-L6-v2) — meaning-based matching
   - **Content-BM25** (with camelCase/snake_case splitting) — keyword matching
   - **Summary-BM25** (node name + file + callers + callees) — structural matching
   - **PageRank** — graph centrality / hub detection

   Signals are fused via **Reciprocal Rank Fusion (RRF)** — rank-based, no score
   calibration issues. Nodes are selected via **round-robin directory interleaving**
   (guarantees facet coverage on multi-aspect queries — one node from each code
   area before any gets a second) with a **per-node token cap** (25% of budget, so
   large components don't crowd out smaller functions). A **relevance cliff**
   (semantic OR BM25 OR summary-BM25 threshold) excludes noise.
3. **readable output** — returns the **actual source code** of selected nodes,
   plus a **codebase map** (table of contents with docstrings, inspired by
   Karpathy's LLM Wiki / Google's OKF) so the model can route follow-up calls
   precisely. The model answers directly from it — no file reads needed.

As you edit, a file watcher (auto-started on first query) updates the graph
**incrementally** — only changed files are re-parsed and merged, and only
changed functions re-embed. Graphs and embeddings are cached on disk
(`~/.cache/fittok`). Set `FITTOK_AUTOWATCH=false` to disable the watcher, in
which case an edit triggers a full re-parse on the next query.

---

## Getting the best results (and known limitations)

### Ask focused questions

fittok ranks code against your question using a **4-signal hybrid** (semantic +
BM25 + structural + PageRank, fused via RRF) with **round-robin directory
diversity** — so multi-facet questions surface code from multiple areas (UI,
server, database) instead of clustering in one dominant area. It's most accurate
with **focused, specific questions** — ideally one concern each, and naming the
function/component/route when you can. Multi-facet questions are supported via
**decomposition** (the tool description tells the model to call once per aspect)
and the **codebase map** (a table of contents prepended to every response).

- ✅ *"How does `runSandboxQuery` execute and isolate a SQL query?"* → surfaces the exact function + its isolation code.
- ✅ *"How does the querydle client submit a query and render results?"* → surfaces the UI component.
- ✅ *"Trace the full lifecycle: UI submission, sandbox execution, data isolation"* → decomposition + round-robin diversity covers multiple facets; the codebase map routes the model to any missed files.

**Rule of thumb:** one concern per question (or 2–3 facets max). For "explain
the whole feature," split it into a few focused questions instead of one mega-query.

### Known limitations

- **GitHub Copilot Chat truncates large MCP outputs (the big one).** Copilot caches MCP tool results above ~7 KB to a `content.json` file, where the entire markdown collapses to ONE physical JSON line (newlines escaped) — and its Read tool truncates any line at ~2,000 characters. So an output over ~7 KB is effectively chopped to ~2,000 chars *regardless of total size*; the model can't see most of the code and falls back to reading source files directly. This is Copilot's delivery layer, not fittok — **every MCP server hits this wall.** By default (0.10.0+) fittok returns *all* relevant code uncapped, which is correct for clients that deliver inline but will be truncated by Copilot. Two workarounds:
  - **Cap the output for Copilot** so it's delivered inline (under the ~7 KB threshold). Set `FITTOK_MAX_BUDGET=1200` in your MCP server's env:
    ```json
    { "servers": { "fittok": { "command": "uvx", "args": ["fittok"], "env": { "FITTOK_MAX_BUDGET": "1200" } } } }
    ```
  - **Use Claude Code or the CLI** for multi-file questions. They deliver MCP output inline with no truncation — which is where fittok's complete (uncapped) results and anti-re-read design actually pay off.
- **Vocabulary gap on abstract queries.** When the query uses words that don't appear in the code (e.g. "isolation" → `REVOKE`/`DENY`), neither semantic nor BM25 can bridge it. The codebase map (file names + docstrings) and round-robin diversity help; naming the function/file routes the model to it.
- **Incremental edge-loss:** editing a file can drop call/import edges *into* it from unchanged files until a full re-parse. fittok auto-recovers on restart or `reset_graph`.
- **Token counts are approximate:** counts use `cl100k_base`, so real usage drifts ~10–20% vs Claude's tokenizer (only matters when you opt into a `FITTOK_MAX_BUDGET` cap).
- **Very large repos:** PageRank is not yet vectorized — fine through low-thousands of nodes, slower beyond that.

---

## Installation

fittok ships as an **MCP server**, a **CLI**, and a **Python library**. It uses
`torch` for embeddings, so a Python runtime must be present. **Pick one runtime**
below, then follow the section for your client.

> Every config below launches fittok as `uvx fittok`. If you chose Python or
> `pipx`, swap that for `python -m fittok` or `pipx run fittok` respectively.

### Prerequisites — choose a runtime (one of)

**A. `uv` — recommended (no Python needed on the machine)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux / macOS
winget install astral-sh.uv                        # Windows
brew install uv                                    # macOS (Homebrew)
```
Launch command: `uvx fittok` — `uv` provisions its own Python + all deps in
isolation. One static binary, so it's deployable org-wide via MDM/Intune/winget.

**B. Python 3.10+ (already on the machine)**

```bash
python -m pip install fittok      # Linux / macOS
py    -m pip install fittok       # Windows
```
Launch command: `python -m fittok` (Windows: `py -m fittok`).
> Managed Linux may reject `pip install` with PEP 668
> ("externally-managed-environment") — use option A to avoid it.

**C. `pipx` — isolated, no global install**

```bash
brew install pipx                               # macOS
pip install --user pipx && pipx ensurepath      # Linux / Windows
```
Launch command: `pipx run fittok`.

### MCP server — Claude Code

```bash
claude mcp add fittok -s user -- uvx fittok
```
Restart Claude Code → `/mcp` → confirm `fittok` is **connected**, then ask
codebase questions normally.

### MCP server — VS Code / GitHub Copilot Chat

```bash
code --add-mcp '{"name":"fittok","command":"uvx","args":["fittok"]}'
```
Or paste into `.vscode/mcp.json` (workspace) or your user `mcp.json`:
```json
{ "servers": { "fittok": { "type": "stdio", "command": "uvx", "args": ["fittok"] } } }
```
Then in Copilot Chat: **Agent** mode → enable fittok's tools (*Configure Tools*).

### MCP server — GitHub Copilot CLI

```bash
copilot mcp add fittok -- uvx fittok
copilot mcp get fittok          # verify status + tools
```

### MCP server — Cursor / Windsurf / any MCP client

```json
{ "mcpServers": { "fittok": { "command": "uvx", "args": ["fittok"] } } }
```

### Auto-trigger (optional, every MCP client)

To make fittok fire on **every** codebase question — without naming it — **and**
stop your client from re-reading files fittok already returned (which would
discard the savings), add this one line to your client's instructions file:

> *"For any codebase question, call fittok first and answer from its output —
> don't re-read files it already returned code from."*

The first half triggers fittok; the second keeps the client from opening the
same files afterward. They reinforce each other — one shapes *strategy* (use
fittok), the other stops the *double-read*. For a stronger, more explicit block:

> For any codebase question ("how does X work", "where is Y"):
> 1. Call the fittok MCP tool first, once.
> 2. Answer directly from its `optimized_context` — it is the real, authoritative
>    source for that question.
> 3. Do NOT read or grep the files fittok already returned code from. That
>    discards the token savings fittok exists to provide.

For the strongest effect, put it in your **user-global** instructions so it
applies to every repo, not just one:

| Client | Instructions file |
|---|---|
| Claude Code | `CLAUDE.md` (repo) or `~/.claude/CLAUDE.md` (user-global) |
| GitHub Copilot | `.github/copilot-instructions.md` or Copilot user instructions |
| Cursor | `.cursor/rules/*.mdc` (or `.cursorrules`) |
| Windsurf | `.windsurfrules` |

> fittok also bakes this rule into every response (an "answer from this, don't
> re-read" line above the code), so it works even without the snippet above —
> the snippet just makes it the client's default across all questions.

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

### Upgrading

`uvx` caches the environment, so a new `fittok` release isn't picked up
automatically — server restarts reuse the cached version. Upgrade with one
command (no need to re-register the MCP server):

```bash
uvx --refresh fittok        # re-resolve from PyPI → latest version
```

Then restart the MCP server (reload the window in VS Code, or restart the
Copilot CLI) so it boots the new version. For other runtimes:

- **pip:** `python -m pip install --upgrade fittok`
- **pipx:** `pipx upgrade fittok`

---

## Why tree-sitter, not LSP?

fittok uses **tree-sitter** (fast, syntactic AST parsing) instead of **LSP** (Language Server Protocol — semantic analysis with types, cross-file references, go-to-definition). This is a deliberate tradeoff:

| | tree-sitter (fittok) | LSP (e.g. Serena MCP) |
|---|---|---|
| **What it returns** | Actual source code — the model answers directly | Symbol metadata (names, refs, types) — the model must still Read files |
| **Setup** | Zero config, works on any directory | Needs language servers installed + project config (tsconfig, pyproject, etc.) |
| **Languages** | 8 out of the box (Python, JS/TS/TSX, Java, Go, Rust) | Only as many as LSP servers you've installed |
| **Startup** | ~15s (parse + embed) | Minutes (full project indexing per language) |
| **Memory** | ~100 MB (graph + embeddings) | 500 MB+ per language server |
| **Model calls per question** | 1–5 (one-shot retrieval) | 5–20+ (iterative symbol navigation) |
| **Token cost** | ~2,500 tokens (code delivered directly) | ~15,000+ tokens (metadata + file reads) |

**fittok's USP is token savings.** It returns the actual code in one call so the model doesn't need to read files. LSP-based tools return metadata (symbol names, reference lists) — precise, but the model still has to open the files to see the implementation. More round-trips, more tokens.

**The tradeoff:** tree-sitter can't resolve cross-file references as accurately as LSP (a `fetch("/api/run")` call in a .tsx file won't perfectly link to the route handler). fittok compensates with 4-signal retrieval (semantic + content-BM25 + structural summary-BM25 + PageRank, fused via RRF) and round-robin directory diversity — which cover the gap in practice.

**Complementary, not competitive:** LSP-based tools like [Serena](https://github.com/oraios/serena) excel at symbol-level navigation ("find all callers of `runSandboxQuery`"). fittok excels at semantic retrieval ("how does SQL execution work?"). Install both — the model picks the right tool per task.

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
| `FITTOK_MAX_BUDGET` | `0` (unlimited) | Code-token cap. `0` = return ALL relevant code in full (default — complete results in Claude Code / CLI). Set `1200` for **GitHub Copilot**, which truncates MCP outputs >~7 KB (see Known limitations). |
| `FITTOK_AUTOWATCH` | `true` | Auto-start the file watcher so graph updates are incremental (only changed files re-parse); set `false` to fall back to full re-parse on edits |
| `FITTOK_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `FITTOK_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `FITTOK_CACHE_DIR` | `~/.cache/fittok` | Cache location |

Full reference: **[docs/HANDBOOK.md](https://github.com/likhithreddy/fittok/blob/main/docs/HANDBOOK.md)**

---

## Requirements

Python ≥ 3.10. First run downloads a ~90 MB embedding model. Graph visualization
(`fittok graph`) is included by default. Optional extras:
- `uv pip install "fittok[ui]"` — Gradio web dashboard (`launch_ui` tool)
- `uv pip install "fittok[gpu]"` — torch/CUDA for GPU-accelerated embeddings

## License

MIT
