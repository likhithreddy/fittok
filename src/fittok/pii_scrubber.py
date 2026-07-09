"""PII scrubbing module — detect and redact sensitive data before processing."""

from __future__ import annotations

import logging
import re
import threading
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
        # Default to the cache dir, NOT next to the source file (a .scrubbed file
        # dropped into the user's tree is surprising and can get committed).
        import os
        from .cache import CACHE_DIR
        out_dir = os.path.join(CACHE_DIR, "scrubbed")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, filepath.name + ".scrubbed")

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


_PII_LOCK = threading.Lock()


def _looks_re_dos(regex: str) -> bool:
    """Heuristic guard: reject regexes that are the classic ReDoS footgun
    (nested quantifiers like ``(a+)+`` / ``(a*)*``, or very long patterns). Not
    a complete defense — Python's stdlib ``re`` has no timeout — but it blocks
    the common case where a crafted client pattern could hang the server."""
    if len(regex) > 500:
        return True
    return bool(re.search(r"\([^)]*[+*?][^)]*\)[+*?]", regex))


def add_pii_pattern(name: str, regex: str) -> dict:
    """Add or override a PII pattern at runtime.

    Args:
        name: Pattern name (e.g. "slack_token").
        regex: Regular expression string.

    Returns:
        Dict with success status and pattern info.
    """
    if _looks_re_dos(regex):
        return {"error": "Pattern rejected: possible ReDoS risk (nested quantifiers or overly long). Simplify it."}
    try:
        compiled = re.compile(regex)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    with _PII_LOCK:
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
    """Create a safe preview of a PII match.

    Never echoes real characters of the secret — the old impl leaked the first
    and last 3 chars of every match into the findings list. Report only length
    and a fully-masked form.
    """
    n = len(value)
    return f"({'*' * min(n, 16)})[len={n}]"
