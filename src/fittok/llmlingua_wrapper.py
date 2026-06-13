"""LLMLingua wrapper — compress context using a local model.

Supports CPU and GPU (CUDA) modes, configurable via:
  - Environment variables:
      FITTOK_MODEL — CPU model name
      FITTOK_MODEL_GPU — GPU model name
      FITTOK_DEVICE — "auto" | "cuda" | "cpu"
  - Function parameters on compress_context()
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .tokens import count_tokens as _count_tokens
from .tokens import truncate_to_tokens as _truncate_to_tokens

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
DEFAULT_GPU_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Cache keyed by model name
_compressors: dict[str, object] = {}


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_mps() -> bool:
    """Check if Apple Silicon MPS is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def _resolve_device(device: str | None = None) -> str:
    """Resolve the target device, preferring CUDA, then MPS, then CPU."""
    dev = device or os.environ.get("FITTOK_DEVICE", "auto")
    if dev == "auto":
        if _has_cuda():
            return "cuda"
        if _has_mps():
            return "mps"
        return "cpu"
    return dev


def _is_llmlingua2_model(name: str) -> bool:
    """LLMLingua-2 models are token-classification models loaded differently."""
    return "llmlingua-2" in name.lower()


def _resolve_model(device: str, model_name: str | None = None) -> str:
    """Pick the right model for the device."""
    if model_name:
        return model_name
    if device == "cuda":
        return os.environ.get("FITTOK_MODEL_GPU", DEFAULT_GPU_MODEL)
    return os.environ.get("FITTOK_MODEL", DEFAULT_MODEL)


def _get_compressor(model_name: str | None = None, device: str | None = None) -> object:
    """Lazily initialize and cache the LLMLingua PromptCompressor."""
    resolved_device = _resolve_device(device)
    name = _resolve_model(resolved_device, model_name)

    if name in _compressors:
        return _compressors[name]

    try:
        from llmlingua import PromptCompressor
    except ImportError:
        raise ImportError(
            "llmlingua is not installed. Install it with: pip install llmlingua"
        )

    use_llmlingua2 = _is_llmlingua2_model(name)
    logger.info(
        "Initializing LLMLingua PromptCompressor(model=%s, device=%s, use_llmlingua2=%s) ...",
        name, resolved_device, use_llmlingua2,
    )
    compressor = PromptCompressor(
        model_name=name,
        device_map=resolved_device,
        use_llmlingua2=use_llmlingua2,
    )
    _compressors[name] = compressor
    return compressor


def compress_context(
    context: str,
    question: str,
    target_tokens: int = 500,
    rate: float | None = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """Compress context text using LLMLingua.

    Args:
        context: The text to compress.
        question: A guiding question to steer compression.
        target_tokens: Target output token count.
        rate: Optional compression ratio override (0.0–1.0).
        model: Optional model name override.
        device: Optional device override ("auto" | "cuda" | "cpu").

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

    compressor = _get_compressor(model, device)

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
        compressed = _truncate_to_tokens(context, target_tokens)

    # Hard ceiling: LLMLingua's `rate` is approximate, so the model can overshoot
    # the requested budget. Guarantee the result never exceeds target_tokens.
    if _count_tokens(compressed) > target_tokens:
        logger.info("Compressed output exceeded budget; truncating to %d tokens", target_tokens)
        compressed = _truncate_to_tokens(compressed, target_tokens)

    compressed_tokens = _count_tokens(compressed)
    compression_ratio = compressed_tokens / max(original_tokens, 1)

    return {
        "compressed": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": compression_ratio,
    }
