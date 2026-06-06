"""Graphify — parse code files into a knowledge graph using tree-sitter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from tree_sitter import Language, Parser, Node

from .models import EdgeType, GraphEdge, GraphMetadata, GraphNode, KnowledgeGraph, NodeType
from .tokens import count_tokens

logger = logging.getLogger(__name__)

# ── Language support ──────────────────────────────────────────────────────────

_LANGUAGE_MAP: dict[str, Language] = {}
_PARSER_CACHE: dict[str, Parser] = {}

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}

# Tree-sitter query patterns per language
_QUERY_TEMPLATES: dict[str, list[tuple[NodeType, str]]] = {
    "python": [
        (NodeType.FUNCTION, "function_definition"),
        (NodeType.CLASS, "class_definition"),
    ],
    "javascript": [
        (NodeType.FUNCTION, "function_declaration"),
        (NodeType.FUNCTION, "arrow_function"),
        (NodeType.FUNCTION, "function_expression"),
        (NodeType.CLASS, "class_declaration"),
        (NodeType.CLASS, "class_expression"),
        (NodeType.METHOD, "method_definition"),
    ],
    "typescript": [
        (NodeType.FUNCTION, "function_declaration"),
        (NodeType.FUNCTION, "arrow_function"),
        (NodeType.CLASS, "class_declaration"),
        (NodeType.METHOD, "method_definition"),
        (NodeType.METHOD, "public_field_definition"),
    ],
    "java": [
        (NodeType.FUNCTION, "method_declaration"),
        (NodeType.CLASS, "class_declaration"),
        (NodeType.CLASS, "interface_declaration"),
    ],
    "go": [
        (NodeType.FUNCTION, "function_declaration"),
        (NodeType.FUNCTION, "method_declaration"),
        (NodeType.CLASS, "type_declaration"),
    ],
    "rust": [
        (NodeType.FUNCTION, "function_item"),
        (NodeType.FUNCTION, "impl_item"),
        (NodeType.CLASS, "struct_item"),
        (NodeType.CLASS, "enum_item"),
        (NodeType.CLASS, "trait_item"),
    ],
}


# ── Language / parser setup ───────────────────────────────────────────────────

def _get_language(lang_name: str) -> Optional[Language]:
    """Load a tree-sitter Language for the given language name.

    Handles tree-sitter >= 0.22 where language packages expose a ``language()``
    function that returns a PyCapsule that must be wrapped in ``Language()``.
    """
    if lang_name in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[lang_name]

    try:
        mod = __import__(f"tree_sitter_{lang_name}")
    except ImportError:
        logger.warning("tree-sitter language package not found: %s", lang_name)
        return None

    lang_attr = getattr(mod, "language", None)
    if lang_attr is None:
        logger.warning("tree_sitter_%s has no 'language' attribute", lang_name)
        return None

    # tree-sitter >= 0.22: language() returns a PyCapsule → wrap in Language()
    if isinstance(lang_attr, Language):
        lang_obj = lang_attr
    elif callable(lang_attr):
        lang_obj = Language(lang_attr())
    else:
        lang_obj = Language(lang_attr)

    _LANGUAGE_MAP[lang_name] = lang_obj
    return lang_obj


def _get_parser(lang_name: str) -> Optional[Parser]:
    """Return a cached Parser for *lang_name*, or None if unavailable."""
    if lang_name in _PARSER_CACHE:
        return _PARSER_CACHE[lang_name]
    language = _get_language(lang_name)
    if language is None:
        return None
    parser = Parser(language)
    _PARSER_CACHE[lang_name] = parser
    return parser


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node_name(node: Node, source_bytes: bytes) -> str:
    """Extract a human-readable name from a definition node."""
    for child in node.children:
        if child.type in ("identifier", "name"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    return text.split("\n")[0][:80]


def _safe_content(source_bytes: bytes, node: Node, max_chars: int = 2000) -> str:
    """Extract node content, truncated for storage."""
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n// ... [truncated]"
    return text


def _extract_imports(node: Node, source_bytes: bytes, _lang: str) -> list[str]:
    """Extract imported module/package names from a top-level import node."""
    imports: list[str] = []
    for child in node.children:
        if child.type in ("dotted_name", "identifier", "string", "module_name"):
            name = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip('"\'')
            if name:
                imports.append(name)
        if child.type == "aliased_import":
            for sub in child.children:
                if sub.type in ("dotted_name", "identifier"):
                    imports.append(source_bytes[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace"))
    return imports


def _extract_calls(node: Node, source_bytes: bytes) -> list[str]:
    """Extract function/method call names from within a node."""
    calls: list[str] = []
    _walk_calls(node, source_bytes, calls)
    return calls


def _walk_calls(node: Node, source_bytes: bytes, calls: list[str]) -> None:
    if node.type == "call":
        callee = node.child_by_field_name("function")
        if callee is None and node.children:
            callee = node.children[0]
        if callee:
            name = source_bytes[callee.start_byte:callee.end_byte].decode("utf-8", errors="replace")
            calls.append(name)
    for child in node.children:
        _walk_calls(child, source_bytes, calls)


# ── Import resolution ─────────────────────────────────────────────────────────

def _resolve_import_target(import_name: str, file_nodes: list[GraphNode]) -> list[str]:
    """Resolve an import name to file node IDs using path-based matching.

    Tries multiple strategies:
      1. Exact file path match (e.g. ``utils`` → ``utils.py``)
      2. Dotted→slash conversion (e.g. ``a.b`` → ``a/b.py`` or ``a/b/__init__.py``)
      3. Substring match on file paths
    """
    targets: list[str] = []

    # Strategy 1 & 2: convert dotted import to file-system path
    slash_path = import_name.replace(".", "/")
    candidates = [
        slash_path + ".py",
        slash_path + ".js",
        slash_path + ".ts",
        slash_path + "/index.js",
        slash_path + "/index.ts",
        slash_path + "/__init__.py",
        import_name + ".py",  # handle bare names like "utils"
        import_name + ".js",
    ]

    for n in file_nodes:
        if n.type != NodeType.FILE:
            continue
        for candidate in candidates:
            if n.file == candidate or n.file.endswith("/" + candidate):
                targets.append(n.id)
                break
        if targets:
            return targets[:1]

    # Strategy 3: substring match as last resort
    for n in file_nodes:
        if n.type != NodeType.FILE:
            continue
        # Match the tail of the import against the file name
        parts = import_name.split(".")
        tail = parts[-1]
        if n.name.startswith(tail):
            targets.append(n.id)
            break

    return targets[:1]


# ── Main parsing ──────────────────────────────────────────────────────────────

def parse_file(filepath: Path, root: Path) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Parse a single file into graph nodes and edges."""
    lang_name = _EXT_TO_LANG.get(filepath.suffix)
    if lang_name is None:
        return [], []

    parser = _get_parser(lang_name)
    if parser is None:
        return [], []

    source_bytes = filepath.read_bytes()
    tree = parser.parse(source_bytes)
    if tree.root_node is None:
        return [], []

    rel_path = str(filepath.relative_to(root))
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    definitions: dict[str, GraphNode] = {}

    # File node
    file_id = f"file:{rel_path}"
    file_node = GraphNode(
        id=file_id,
        type=NodeType.FILE,
        name=filepath.name,
        file=rel_path,
        line_start=1,
        line_end=source_bytes.count(b"\n") + 1,
        content="",
        token_count=0,
    )
    nodes.append(file_node)

    # Walk top-level statements for imports
    import_targets: list[str] = []
    import_node_types = {
        "python": {"import_statement", "import_from_statement"},
        "javascript": {"import_statement", "require_clause"},
        "typescript": {"import_statement", "require_clause"},
        "java": {"import_declaration"},
        "go": {"import_declaration"},
        "rust": {"use_declaration"},
    }

    for child in tree.root_node.children:
        if child.type in import_node_types.get(lang_name, set()):
            imported = _extract_imports(child, source_bytes, lang_name)
            import_targets.extend(imported)

    # Extract definitions
    templates = _QUERY_TEMPLATES.get(lang_name, [])
    for target_type, ts_type in templates:
        for ts_node in _find_nodes_by_type(tree.root_node, ts_type):
            name = _node_name(ts_node, source_bytes)
            content = _safe_content(source_bytes, ts_node)
            node_id = f"{target_type.value}:{rel_path}:{name}:{ts_node.start_point[0]}"

            graph_node = GraphNode(
                id=node_id,
                type=NodeType(target_type),
                name=name,
                file=rel_path,
                line_start=ts_node.start_point[0] + 1,
                line_end=ts_node.end_point[0] + 1,
                content=content,
                token_count=count_tokens(content),
            )
            nodes.append(graph_node)
            definitions[name] = graph_node

            edges.append(GraphEdge(source=file_id, target=node_id, type=EdgeType.CONTAINS))

            calls = _extract_calls(ts_node, source_bytes)
            for call_name in calls:
                edges.append(GraphEdge(
                    source=node_id,
                    target=call_name,  # placeholder, resolved later
                    type=EdgeType.CALLS,
                ))

    # Import edges (placeholders)
    for imp in import_targets:
        edges.append(GraphEdge(
            source=file_id,
            target=imp,  # placeholder
            type=EdgeType.IMPORTS,
        ))

    return nodes, edges


def _find_nodes_by_type(node: Node, type_name: str) -> list[Node]:
    """Recursively find all descendant nodes of a given type."""
    results: list[Node] = []
    if node.type == type_name:
        results.append(node)
    for child in node.children:
        results.extend(_find_nodes_by_type(child, type_name))
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def parse_codebase(root_path: str) -> KnowledgeGraph:
    """Parse an entire codebase into a KnowledgeGraph.

    Recursively walks the directory, parses each code file with tree-sitter,
    and assembles a graph of nodes (files, functions, classes) and edges
    (imports, calls, references, contains).
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []

    skip_dirs = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".next", ".nuxt", "target", "vendor", ".gradle",
    }

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            filepath = Path(dirpath) / filename
            if filepath.suffix not in _EXT_TO_LANG:
                continue
            try:
                file_nodes, file_edges = parse_file(filepath, root)
                all_nodes.extend(file_nodes)
                all_edges.extend(file_edges)
            except Exception:
                logger.warning("Failed to parse %s", filepath, exc_info=True)

    # ── Resolve placeholder edge targets ─────────────────────────────────────

    # Build lookup indices
    name_to_ids: dict[str, list[str]] = {}
    for n in all_nodes:
        name_to_ids.setdefault(n.name, []).append(n.id)

    file_nodes = [n for n in all_nodes if n.type == NodeType.FILE]

    resolved_edges: list[GraphEdge] = []
    for edge in all_edges:
        if edge.type in (EdgeType.CALLS, EdgeType.REFERENCES):
            # Match by definition name
            targets = name_to_ids.get(edge.target, [])
            for tid in targets[:3]:
                resolved_edges.append(GraphEdge(
                    source=edge.source, target=tid, type=edge.type
                ))

        elif edge.type == EdgeType.IMPORTS:
            # Use path-based resolution for imports
            targets = _resolve_import_target(edge.target, file_nodes)
            if not targets:
                # Fallback: try name-based match
                targets = name_to_ids.get(edge.target, [])
            for tid in targets[:1]:
                resolved_edges.append(GraphEdge(
                    source=edge.source, target=tid, type=edge.type
                ))

        else:
            # CONTAINS edges already have proper IDs
            resolved_edges.append(edge)

    graph = KnowledgeGraph(
        nodes=all_nodes,
        edges=resolved_edges,
        metadata=GraphMetadata(
            root=str(root),
            total_nodes=len(all_nodes),
            total_edges=len(resolved_edges),
        ),
    )
    return graph


# ── Chunked streaming parser ─────────────────────────────────────────────────

_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".next", ".nuxt", "target", "vendor", ".gradle",
}


def _get_code_files(root_path: str) -> list[Path]:
    """Collect all parseable code files under root."""
    root = Path(root_path).resolve()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix in _EXT_TO_LANG:
                files.append(p)
    return files


def _resolve_edges(all_nodes: list[GraphNode], all_edges: list[GraphEdge]) -> list[GraphEdge]:
    """Resolve placeholder edge targets to node IDs."""
    name_to_ids: dict[str, list[str]] = {}
    for n in all_nodes:
        name_to_ids.setdefault(n.name, []).append(n.id)

    file_nodes = [n for n in all_nodes if n.type == NodeType.FILE]

    resolved: list[GraphEdge] = []
    for edge in all_edges:
        if edge.type in (EdgeType.CALLS, EdgeType.REFERENCES):
            targets = name_to_ids.get(edge.target, [])
            for tid in targets[:3]:
                resolved.append(GraphEdge(source=edge.source, target=tid, type=edge.type))
        elif edge.type == EdgeType.IMPORTS:
            targets = _resolve_import_target(edge.target, file_nodes)
            if not targets:
                targets = name_to_ids.get(edge.target, [])
            for tid in targets[:1]:
                resolved.append(GraphEdge(source=edge.source, target=tid, type=edge.type))
        else:
            resolved.append(edge)
    return resolved


async def parse_codebase_stream(root_path: str, batch_size: int = 50):
    """Async generator that parses in batches, yielding progress.

    Yields dicts with progress info, final yield contains the graph.
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    files = _get_code_files(root_path)
    total = len(files)
    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []

    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        for fp in batch:
            try:
                n, e = parse_file(fp, root)
                all_nodes.extend(n)
                all_edges.extend(e)
            except Exception:
                logger.warning("Failed to parse %s", fp, exc_info=True)

        yield {
            "stage": "parsing",
            "progress": min((i + batch_size) / total, 1.0),
            "files_parsed": min(i + batch_size, total),
            "total_files": total,
            "nodes_so_far": len(all_nodes),
        }

    resolved = _resolve_edges(all_nodes, all_edges)
    graph = KnowledgeGraph(
        nodes=all_nodes,
        edges=resolved,
        metadata=GraphMetadata(
            root=str(root),
            total_nodes=len(all_nodes),
            total_edges=len(resolved),
        ),
    )

    yield {
        "stage": "parsing",
        "status": "done",
        "total_nodes": len(all_nodes),
        "total_edges": len(resolved),
        "graph": graph,
    }


# ── Incremental graph updates ────────────────────────────────────────────────

def update_graph(graph: KnowledgeGraph, root_path: str, changed_files: list[str]) -> KnowledgeGraph:
    """Re-parse only changed files and merge into existing graph."""
    root = Path(root_path).resolve()

    # Remove old nodes/edges for changed files
    changed_rel = set()
    for cf in changed_files:
        try:
            changed_rel.add(str(Path(cf).relative_to(root)))
        except ValueError:
            changed_rel.add(cf)

    # Filter out old data for changed files
    kept_nodes = [n for n in graph.nodes if n.file not in changed_rel]
    kept_node_ids = {n.id for n in kept_nodes}

    # Remove edges where source OR target belongs to a changed file
    kept_edges = [
        e for e in graph.edges
        if e.source in kept_node_ids and e.target in kept_node_ids
    ]

    # Re-parse changed files
    new_nodes: list[GraphNode] = []
    new_edges: list[GraphEdge] = []
    for cf in changed_files:
        fp = Path(cf)
        if not fp.is_file() or fp.suffix not in _EXT_TO_LANG:
            continue
        try:
            n, e = parse_file(fp, root)
            new_nodes.extend(n)
            new_edges.extend(e)
        except Exception:
            logger.warning("Failed to re-parse %s", fp, exc_info=True)

    # Merge
    merged_nodes = kept_nodes + new_nodes
    merged_edges = kept_edges + new_edges

    # Re-resolve edges
    resolved = _resolve_edges(merged_nodes, merged_edges)

    return KnowledgeGraph(
        nodes=merged_nodes,
        edges=resolved,
        metadata=GraphMetadata(
            root=str(root),
            total_nodes=len(merged_nodes),
            total_edges=len(resolved),
        ),
    )


def save_graph(graph: KnowledgeGraph, output_path: str | None = None) -> str:
    """Save a KnowledgeGraph to JSON. Returns the output path."""
    if output_path is None:
        output_path = os.path.join(graph.metadata.root, "graph.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(graph.model_dump_json(indent=2))
    logger.info("Graph saved to %s (%d nodes, %d edges)", output_path, len(graph.nodes), len(graph.edges))
    return output_path


def load_graph(path: str) -> KnowledgeGraph:
    """Load a KnowledgeGraph from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return KnowledgeGraph.model_validate(data)
