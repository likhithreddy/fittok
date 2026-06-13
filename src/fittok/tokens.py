"""Shared token-counting utility."""

from __future__ import annotations

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding."""
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most *max_tokens* tokens (token-exact)."""
    if max_tokens <= 0:
        return ""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])
