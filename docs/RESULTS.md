# fittok — How it works & token-savings results

> **fittok** retrieves only the *relevant* source code for a question — instead of
> the model reading whole files — so an LLM answers codebase questions on a small,
> focused slice of context. Less input = fewer tokens, lower cost, faster answers.

---

## TL;DR

- On a real Next.js/TypeScript codebase (≈5k functions), fittok answers a focused
  question from a **~1.2–3.5k-token slice instead of the ~9–20k tokens** the model
  would otherwise read — a **60–91% reduction in input context**, same answer.
- In a live Claude Code session, the same question consumed **~18.9k tokens without
  fittok vs ~10.7k with it** (one tool call, zero file reads).
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

**CLI / library:** `pip install fittok`, then `fittok query <repo> "<q>"` or
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

### 5b. End-to-end session — with vs without (live `/context`)

Same focused question, same repo, in Claude Code:

| | Without fittok | With fittok |
|---|---|---|
| fittok calls | 0 | 1 |
| Files the model read | several | 0 |
| Context consumed (`Messages`) | **~18.9k** | **~10.7k** |

> _Note: a fixed ~5.3k of "MCP tools" overhead is present in both and cancels out;
> the comparison is the `Messages` delta._

### 5c. Selectivity proof

On a synthetic repo of **1,010 functions across 10 unrelated domains** (auth,
payment, email, geometry, weather, …), the query *"how does authentication and login
work"* selected **68 nodes, 100% from `auth.py`** — zero leakage from the other 9
domains.

---

## 6. Live demo — screenshots (fill in from your run)

> Replace each placeholder with your screenshot and the measured numbers.

**Question used:** `How does silence detection end the candidate's turn?`

### Without fittok  ✅ measured
- `/context` before — Messages: **1.1k** (total 37.4k)
- `/context` after — Messages: **11.5k** (total 46.9k)
- Files read directly: **~5** (`silenceDetector.ts`, `useSpeechRecognition.ts`,
  `speechRecognitionWrapper.ts`, + grep/cat of `submitAnswer.ts`, `constants.ts`)
- **Context consumed (delta): ≈10.4k tokens**
- _[screenshot ①: `/context` before] · [screenshot ②: `/context` after]_

### With fittok  ⏳ pending
- `/context` before — Messages: `____`
- `/context` after — Messages: `____`
- fittok calls: `____` (target 1) · files read: `____` (target 0)
- fittok's own `savings`: `____`% (`____` vs `____` tokens)
- _[screenshot ④: before] · [⑤: after] · [⑥: tool call + savings] · [⑦: answer + 🪙 footer]_

### Result
> Same focused question, same repo. **Without fittok** the model ran ~5 reads/greps
> and consumed **≈10.4k tokens** of context. **With fittok** it answered from one
> `~__k`-token slice — **`__`% less context, same answer.**

---

## 7. Reproduce it yourself

```bash
pip install fittok          # or: uvx fittok ...
fittok index   <your-repo>  # one-time pre-warm (parse + embeddings, cached)
fittok query   <your-repo> "how does <feature> work"   # prints the slice + savings on stderr
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
