"""Shared token-counting utility.

NOTE on accuracy: counts use OpenAI's ``cl100k_base`` BPE as a stable,
dependency-light approximation of model context size. It is NOT Claude's exact
tokenizer, so real budgets can land ~10-20% off in either direction. The
budget constants in slurp.py carry headroom for this. If you need exact Claude
counts, swap the encoding here for the Anthropic tokenizer.
"""

from __future__ import annotations

_ENCODING = None


def _get_enc():
    """Lazily load the cl100k_base encoding.

    Done lazily (not at import) so a tiktoken download/encoding error doesn't
    brick the whole package import.
    """
    global _ENCODING
    if _ENCODING is None:
        import tiktoken
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding (approximation — see module doc)."""
    return len(_get_enc().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most *max_tokens* tokens (token-exact)."""
    if max_tokens <= 0:
        return ""
    enc = _get_enc()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
