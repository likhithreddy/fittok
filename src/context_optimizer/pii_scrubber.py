"""PII scrubbing module — detect and redact sensitive data before processing."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PII_PATTERNS: dict[str, str] = {
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "aws_secret_key": r"(?i)aws_secret_access_key\s*[=:]\s*\S+",
    "github_token": r"gh[pousr]_[A-Za-z0-9_]{36,}",
    "generic_api_key": r"(?i)(api[_-]?key|apikey|access[_-]?token)\s*[=:]\s*['\"]?[A-Za-z0-9\-_.]{20,}['\"]?",
    "bearer_token": r"Bearer\s+[A-Za-z0-9._\-]+",
    "private_key": r"-----BEGIN\s+(RSA|DSA|EC|OPENSSH)?\s*PRIVATE KEY-----",
    "email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "jwt": r"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+",
    "connection_string": r"(?i)(mongodb|postgres|mysql|redis)://\S+",
}

# Compile patterns
_compiled: dict[str, re.Pattern] = {}


def _get_compiled() -> dict[str, re.Pattern]:
    if not _compiled:
        for name, pattern in PII_PATTERNS.items():
            _compiled[name] = re.compile(pattern)
    return _compiled


def scrub_text(text: str, custom_patterns: dict | None = None) -> dict:
    """Scrub PII from text, replacing with redacted placeholders.

    Args:
        text: Input text to scrub.
        custom_patterns: Optional dict of {name: regex} to add/override.

    Returns:
        Dict with scrubbed text, findings list, and count.
    """
    patterns = dict(_get_compiled())
    if custom_patterns:
        for name, regex in custom_patterns.items():
            patterns[name] = re.compile(regex)

    findings: list[dict] = []
    scrubbed = text

    for name, pattern in patterns.items():
        for match in pattern.finditer(text):
            findings.append({
                "type": name,
                "start": match.start(),
                "end": match.end(),
                "preview": _preview(match.group()),
            })

    # Replace all matches (order-independent — use original text positions)
    scrubbed = text
    for name, pattern in patterns.items():
        scrubbed = pattern.sub(f"[REDACTED_{name.upper()}]", scrubbed)

    return {
        "scrubbed": scrubbed,
        "findings": findings,
        "count": len(findings),
    }


def scrub_file(path: str, output_path: str | None = None) -> dict:
    """Scrub PII from a file.

    Args:
        path: Path to the file to scrub.
        output_path: Optional output path. Defaults to <path>.scrubbed.

    Returns:
        Dict with scrubbed content, findings, and output path.
    """
    filepath = Path(path)
    if not filepath.is_file():
        return {"error": f"File not found: {path}"}

    content = filepath.read_text(encoding="utf-8", errors="replace")
    result = scrub_text(content)

    if output_path is None:
        output_path = str(filepath) + ".scrubbed"

    Path(output_path).write_text(result["scrubbed"], encoding="utf-8")

    return {
        **result,
        "output_path": output_path,
        "source": path,
    }


def list_pii_patterns() -> dict:
    """Return all registered PII patterns."""
    return {
        "patterns": {name: pattern for name, pattern in PII_PATTERNS.items()},
        "count": len(PII_PATTERNS),
    }


def add_pii_pattern(name: str, regex: str) -> dict:
    """Add or override a PII pattern at runtime.

    Args:
        name: Pattern name (e.g. "slack_token").
        regex: Regular expression string.

    Returns:
        Dict with success status and pattern info.
    """
    try:
        compiled = re.compile(regex)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    PII_PATTERNS[name] = regex
    _compiled[name] = compiled

    return {
        "added": True,
        "name": name,
        "pattern": regex,
        "total_patterns": len(PII_PATTERNS),
    }


def scrub_graph_content(graph) -> None:
    """Scrub PII from all node content in a KnowledgeGraph (in-place)."""
    for node in graph.nodes:
        if node.content:
            result = scrub_text(node.content)
            if result["count"] > 0:
                node.content = result["scrubbed"]


def _preview(value: str, max_len: int = 20) -> str:
    """Create a safe preview of a PII match."""
    if len(value) <= max_len:
        return value[:3] + "..." + value[-3:]
    return value[:3] + "..." + value[-3:]
