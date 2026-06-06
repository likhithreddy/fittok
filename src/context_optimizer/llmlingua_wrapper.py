"""LLMLingua wrapper — compress context using a local model.

The model can be configured via:
  - The ``CONTEXT_OPTIMIZER_MODEL`` environment variable
  - The ``model`` parameter on ``compress_context()``
  - Default: ``microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank``
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .tokens import count_tokens as _count_tokens

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Cache keyed by model name
_compressors: dict[str, object] = {}


def _get_compressor(model_name: str | None = None) -> object:
    """Lazily initialize and cache the LLMLingua PromptCompressor."""
    name = model_name or os.environ.get("CONTEXT_OPTIMIZER_MODEL", DEFAULT_MODEL)

    if name in _compressors:
        return _compressors[name]

    try:
        from llmlingua import PromptCompressor
    except ImportError:
        raise ImportError(
            "llmlingua is not installed. Install it with: pip install llmlingua"
        )

    logger.info("Initializing LLMLingua PromptCompressor(model=%s) ...", name)
    compressor = PromptCompressor(model_name=name)
    _compressors[name] = compressor
    return compressor


def compress_context(
    context: str,
    question: str,
    target_tokens: int = 500,
    rate: float | None = None,
    model: Optional[str] = None,
) -> dict:
    """Compress context text using LLMLingua.

    Args:
        context: The text to compress.
        question: A guiding question to steer compression.
        target_tokens: Target output token count.
        rate: Optional compression ratio override (0.0–1.0).
        model: Optional model name override (or set CONTEXT_OPTIMIZER_MODEL env var).

    Returns:
        Dict with keys: compressed, original_tokens, compressed_tokens, compression_ratio
    """
    original_tokens = _count_tokens(context)

    if original_tokens == 0:
        return {
            "compressed": "",
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compression_ratio": 0.0,
        }

    if original_tokens <= target_tokens:
        return {
            "compressed": context,
            "original_tokens": original_tokens,
            "compressed_tokens": original_tokens,
            "compression_ratio": 1.0,
        }

    compressor = _get_compressor(model)

    effective_rate: float = rate if rate is not None else target_tokens / original_tokens
    effective_rate = max(0.1, min(effective_rate, 1.0))

    try:
        result = compressor.compress_prompt(
            context,
            question=question,
            rate=effective_rate,
        )
        compressed = result.get("compressed_prompt", context)

        if isinstance(compressed, list):
            compressed = " ".join(compressed)

    except Exception:
        logger.warning("LLMLingua compression failed, falling back to truncation", exc_info=True)
        chars_per_token = len(context) / max(original_tokens, 1)
        target_chars = int(target_tokens * chars_per_token)
        compressed = context[:target_chars]

    compressed_tokens = _count_tokens(compressed)
    compression_ratio = compressed_tokens / max(original_tokens, 1)

    return {
        "compressed": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": compression_ratio,
    }
