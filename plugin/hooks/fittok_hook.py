#!/usr/bin/env python3
"""fittok UserPromptSubmit hook — deterministic context injection.

Claude Code runs this on every prompt BEFORE the model sees it. We:
  1. read the prompt + cwd from stdin (JSON),
  2. relevance-gate: only act on codebase questions in a code directory,
  3. run `fittok query <cwd> "<prompt>"` to get the token-optimized context,
  4. inject it via hookSpecificOutput.additionalContext so the model answers
     from it instead of reading files.

It is FAIL-SAFE: any error / timeout / non-code prompt → no-op (empty output,
exit 0). It never blocks the session. Tune via env:
  FITTOK_BIN          (default "fittok")   — the CLI to invoke
  FITTOK_HOOK_TIMEOUT (default "25")       — seconds before giving up
  FITTOK_HOOK_BUDGET  (default "0")        — token budget (0 = adaptive)
"""

import json
import os
import subprocess
import sys


def _noop():
    # Minimal valid UserPromptSubmit output that injects nothing.
    json.dump({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit"}}, sys.stdout)
    sys.exit(0)


# A prompt is "about the codebase" if it reads like a how/where/what question.
_CODE_HINTS = (
    "how does", "how do", "how is", "how are", "where is", "where does", "where are",
    "what does", "what is", "explain", "walk me through", "trace", "why does",
    "implement", "add a", "add support", "fix", "refactor", "debug", "architecture",
    "flow", "where's", "how's", "find the", "which file", "what happens when",
)


def _looks_like_code_question(prompt: str) -> bool:
    p = prompt.strip().lower()
    if len(p) < 12:
        return False
    return any(h in p for h in _CODE_HINTS)


def _is_code_dir(cwd: str) -> bool:
    markers = (".git", "package.json", "pyproject.toml", "go.mod", "Cargo.toml", "pom.xml")
    try:
        return any(os.path.exists(os.path.join(cwd, m)) for m in markers)
    except OSError:
        return False


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        _noop()

    prompt = (data.get("prompt") or "").strip()
    cwd = data.get("cwd") or os.getcwd()

    if not _looks_like_code_question(prompt) or not _is_code_dir(cwd):
        _noop()

    fittok_bin = os.environ.get("FITTOK_BIN", "fittok")
    budget = os.environ.get("FITTOK_HOOK_BUDGET", "0")
    timeout = float(os.environ.get("FITTOK_HOOK_TIMEOUT", "25"))

    try:
        proc = subprocess.run(
            [fittok_bin, "query", cwd, prompt, "--budget", budget],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        _noop()  # cold index too slow, fittok not installed, etc. — degrade silently

    context = (proc.stdout or "").strip()
    if proc.returncode != 0 or not context:
        _noop()

    injected = (
        "## Relevant code for this question (retrieved by fittok, token-optimized)\n\n"
        "Answer from the code below. It is the relevant slice of the codebase for "
        "this question — prefer it over reading whole files, which wastes tokens.\n\n"
        + context
    )
    json.dump(
        {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": injected}},
        sys.stdout,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
