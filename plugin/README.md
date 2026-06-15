# fittok — Claude Code plugin

Bundles two layers so token-optimized context "just happens" in Claude Code:

1. **MCP server** (`mcpServers.fittok` → `uvx fittok`) — model-invoked. The model
   can call `optimize_context` for codebase questions.
2. **`UserPromptSubmit` hook** (`hooks/fittok_hook.py`) — **deterministic**. On
   every codebase question it auto-injects the relevant, token-optimized code
   *before the model decides*, so optimization happens whether or not the model
   chooses to call the tool.

## Install

Requires `fittok` reachable by the hook. The default uses `uvx fittok` (it's on
PyPI); for a local checkout, point the hook at it via `FITTOK_BIN`.

```bash
# Local plugin (no marketplace):
claude --plugin-dir /path/to/fittok/plugin
# or copy this `plugin/` dir under ~/.claude/plugins/ and enable it.
```

## Pre-warm (recommended)

The first codebase question triggers a one-time index (parse + embeddings, ~15s)
that the hook runs synchronously. Pre-warm once to avoid that first-prompt delay:

```bash
uvx fittok index /path/to/your/repo
```

## Tuning (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `FITTOK_BIN` | `fittok` | CLI the hook invokes (set to a venv path for local installs) |
| `FITTOK_HOOK_TIMEOUT` | `25` | Seconds before the hook gives up and no-ops |
| `FITTOK_HOOK_BUDGET` | `0` | Token budget (0 = adaptive) |
| `FITTOK_SHOW_SAVINGS` | `false` | Footer with token savings on each answer |

## Honest tradeoffs

- The hook fires on **every** prompt and injects on every *codebase question*
  (relevance-gated by keywords + a code-dir check). It is **fail-safe**: any
  error/timeout/non-code prompt is a silent no-op that never blocks the session.
- Injection costs ~the budget (default adaptive, ~1.5–8k tokens) **per matched
  prompt**, even if the model didn't strictly need it. That's the price of the
  deterministic guarantee. Turn the hook off and rely on the MCP alone if you'd
  rather the model decide.
- Injecting the relevant code makes file-reading far less likely, but the host
  model can still choose to read more — no hook can fully prevent that.
- Cross-HTTP-boundary connectors (e.g. an API route reached via `fetch('/api/x')`)
  may not be in the slice; broad "explain everything" questions can still need
  a file read or two. Focused questions are tight.
