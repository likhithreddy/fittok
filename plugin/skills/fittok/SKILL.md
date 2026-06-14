---
name: fittok
description: Use when answering "how does X work" / "where is Y" / "explain Z" questions about THIS codebase. Retrieves the token-optimized relevant source via the fittok MCP instead of reading whole files.
---

# fittok — token-optimized codebase context

For any question about how the current codebase works, call the **fittok**
MCP tool `optimize_context` ONCE with the repo path and the question, then
answer directly from its `optimized_context` output.

Do **not** separately read or grep the files it came from — that re-reads the
same code at full cost and negates the token savings. The returned slice is the
relevant code, already ranked and trimmed to a budget.

If the `UserPromptSubmit` hook has already injected a "Relevant code for this
question" block, answer from that and skip the tool call entirely.
