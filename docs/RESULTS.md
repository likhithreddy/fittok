# fittok — How it works & token-savings results

> **fittok** retrieves only the *relevant* source code for a question — instead of
> the model reading whole files — so an LLM answers codebase questions on a small,
> focused slice of context. Less input = fewer tokens, lower cost, faster answers.

---

## TL;DR

- On a real Next.js/TypeScript codebase (≈5k functions), fittok answers a question
  from a **~1.5–3.5k-token slice instead of the ~13–20k tokens** the model would
  otherwise read — an **~80–90% reduction in input context** (deterministic, in the
  tool's `savings` footer), same answer.
- On a thorough model (Opus 4.8) a broad question cost **~84k total tokens without
  fittok vs ~27k with it** — because fittok let it answer in one tool call instead
  of spawning a 58k-token file-exploration subagent.
- It works three ways from one install — **MCP server, CLI, and Python library** —
  plus a **Claude Code plugin** that injects the context automatically.

---

## 1. The problem

To answer "how does X work in this codebase?", an AI agent typically **reads many
whole files** (Grep → Read → Read …). Most of those tokens are irrelevant — config,
imports, unrelated functions. You pay for all of them on every question.

## 2. How it works — a 3-stage pipeline

```
codebase ──▶ graphify ──▶ slurp ──▶ readable slice ──▶ LLM answers
             (parse)      (select)   (trim to budget)
```

1. **graphify** — parses the repo with tree-sitter into a knowledge graph of
   functions/classes/methods (Python, JS, JSX, TS, TSX, Java, Go, Rust).
2. **slurp** — scores every node against the question with **semantic embeddings +
   TF-IDF + PageRank**, and selects only the most relevant nodes within a token
   budget (auto-sized by default).
3. **readable output** — returns the *actual source code* of those nodes (trimmed to
   budget), so the model can answer directly. (An earlier design compressed the text
   with LLMLingua, but that produced unreadable token-salad the model ignored — so
   fittok returns real, readable code instead.)

Embeddings are cached on disk and keyed by content, so re-indexing after a code
change only re-embeds what changed.

## 3. Interfaces (one install, four front doors)

| Interface | Who uses it | How |
|---|---|---|
| **MCP server** | AI clients (Claude Code, Cursor) | `uvx fittok` registered as an MCP; the model calls `optimize_context` |
| **Claude Code plugin** | Claude Code users who want it automatic | `UserPromptSubmit` hook auto-injects the relevant context every codebase question |
| **CLI** | scripts / CI / verification | `fittok query <repo> "<question>"` |
| **Python library** | custom pipelines | `from fittok import optimize` |

## 4. Install & usage

**Recommended (MCP via uvx):** add to your client's MCP config —
```json
{ "mcpServers": { "fittok": { "command": "uvx", "args": ["fittok"] } } }
```
**Auto-trigger without mentioning it:** add one line to your client's `CLAUDE.md` —
> *"For any codebase question, call fittok first and answer from its output."*

**CLI (no install):** `uvx fittok query <repo> "<q>"`.
**Library:** `uv add fittok` (or `uv pip install fittok`), then
`from fittok import optimize; optimize("<repo>", "<q>")`.

---

## 5. Token-savings results

### 5a. Engine savings — deterministic (no model in the loop)

Measured directly via `fittok query` on the `mira` repo (adaptive budget). "Baseline"
= total tokens of the files the answer lives in (what the model would otherwise read):

| Question | Baseline (files) | fittok sent | **Reduction** |
|---|---:|---:|---:|
| How does authentication & login work | 13,178 | 1,200 | **90.9%** |
| How does silence detection end a turn | 9,782 | 3,500 | **64.2%** |
| How does the AI gateway route & rotate keys | 14,041 | 3,500 | **75.1%** |
| How are interview questions generated from the resume | 19,668 | 3,500 | **82.2%** |

This number is **deterministic** — same every run, independent of the host model.

### 5b. End-to-end, in a real client — and how to measure it honestly

A **broad** pipeline question on `mira`, same prompt both sides, answer length held
constant ("~5 sentences"):

| Model | Without fittok (total tokens) | With fittok | fittok footer |
|---|---|---|---|
| Opus 4.8 | **~84k** (26k messages + **58.4k Explore subagent**) | **~27k** (1 call, 0 reads) | 84% (2,494 vs 15,631) |
| glm-5.1 | ~21k messages (~5 file reads) | ~17.5k messages | 86.6% |

**Measure total token spend (your API bill) or the `savings` footer — NOT Claude
Code's `/context` "Messages" number.** `/context` Messages:
- **excludes subagent tokens** — Opus's 58.4k Explore crawl never appears there, so
  the without-fittok cost is massively under-reported;
- **is dominated by the model's own reasoning**, which fittok doesn't touch and which
  varies wildly run-to-run.

So on `/context` Messages alone the two columns can look equal, while the **real
billed cost differs ~3×**. The footer and the bill tell the truth; `/context` hides it.

### 5c. Selectivity proof

On a synthetic repo of **1,010 functions across 10 unrelated domains** (auth,
payment, email, geometry, weather, …), the query *"how does authentication and login
work"* selected **68 nodes, 100% from `auth.py`** — zero leakage from the other 9
domains.

---

## 6. What "noise" looks like (and why narrow questions mislead)

The same broad question was run across two models. The takeaway: the **win scales
with how much the model would otherwise explore**, and the noise is the model's
*reasoning*, which fittok doesn't change.

- **Narrow question** ("how does silence detection end a turn"): the answer lives in
  ~2 small files. A capable model reads them cheaply, and its reasoning (~8–24k,
  varying) dwarfs the tiny input difference — so `/context` shows a near-tie even
  though fittok still sent less. **Small questions are fittok's worst case.**
- **Broad question** ("walk me through the whole pipeline"): the model must read
  10–17 files (or spawn a subagent). Here fittok's slice (~2.5k) replaces 15–58k of
  exploration — a large, real win, biggest on thorough models (see §5b).

**Bottom line:** judge fittok by the **deterministic footer** and the **total token
bill**, on **broad/multi-file questions** — that's where it's designed to help and
where the saving is unambiguous.

---

## 7. Reproduce it yourself

```bash
uvx fittok index <your-repo>   # one-time pre-warm (parse + embeddings, cached)
uvx fittok query <your-repo> "how does <feature> work"   # prints the slice + savings on stderr
```
The stderr line shows `Sent X tokens instead of Y (Z% reduction)` — the deterministic proof.

---

## 8. Honest limitations

- **Focused questions are where it shines** (tight, zero file reads). **Broad
  "explain the entire flow" questions** can miss a pivotal connector function (it may
  rank low on vocabulary, or sit across an HTTP boundary a code graph can't cross), so
  the model may still read 1–2 files. The win is smaller there, not absent.
- **The MCP can't force the model** to use fittok or to stop reading files — it can
  only make it the easy, obvious path. The **plugin hook** is the deterministic
  guarantee (it injects the context before the model decides), at the cost of running
  on every matched prompt.
- Token savings depend on the model trusting the slice; readable output makes that
  far more likely, but the host model always has final say.
