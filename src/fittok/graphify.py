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
    ".tsx": "tsx",       # tsx needs the dedicated tree-sitter-typescript TSX grammar
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}

# Maps internal language name -> (pip module, attribute returning the grammar).
# tree-sitter-typescript is the notable case: it exposes language_typescript /
# language_tsx (NOT a bare `language`), so .ts/.tsx files parsed to nothing
# before this mapping existed.
_LANG_MODULE: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
}

# Directories that never contain hand-written source worth indexing.
# Single source of truth — used by every file-walking entry point below.
_SKIP_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".next", ".nuxt", "target", "vendor", ".gradle",
    # Test-coverage and other generated report output (the usual noise culprits)
    "coverage", "htmlcov", ".nyc_output", "coverage-report",
    "out", ".cache", ".turbo", ".parcel-cache", "storybook-static",
    "bower_components", "site-packages", "__snapshots__",
}

# Generated / bundled files that are technically valid JS/TS but are not
# meaningful source (minified bundles, sourcemap shims, etc.).
_GENERATED_SUFFIXES: tuple[str, ...] = (
    ".min.js", ".min.ts", ".bundle.js", ".chunk.js", ".min.jsx", ".min.tsx",
)


def _is_generated_file(path: Path) -> bool:
    """True for minified/bundled/generated files that pollute the graph."""
    name = path.name.lower()
    return name.endswith(_GENERATED_SUFFIXES) or ".min." in name or name.endswith(".d.ts")

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
    "tsx": [
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


# Top-level statement types that bind a name to a value (module-scope constants).
# Captured as CONSTANT nodes so a referenced constant (MODEL_NAME, API_URL, …)
# can be surfaced as a dependency instead of forcing a file re-read. JS/TS
# declarators whose value is a function/arrow/class are skipped here — they are
# already captured as FUNCTION/CLASS nodes by the templates above.
_CONST_STMT_TYPES: dict[str, set[str]] = {
    "python": {"assignment"},
    "javascript": {"lexical_declaration", "variable_declaration"},
    "typescript": {"lexical_declaration", "variable_declaration"},
    "tsx": {"lexical_declaration", "variable_declaration"},
}


# ── Language / parser setup ───────────────────────────────────────────────────

def _get_language(lang_name: str) -> Optional[Language]:
    """Load a tree-sitter Language for the given language name.

    Handles tree-sitter >= 0.22 where language packages expose a ``language()``
    function that returns a PyCapsule that must be wrapped in ``Language()``.
    """
    if lang_name in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[lang_name]

    module_name, attr_name = _LANG_MODULE.get(
        lang_name, (f"tree_sitter_{lang_name}", "language")
    )
    try:
        mod = __import__(module_name)
    except ImportError:
        logger.warning("tree-sitter language package not found: %s", module_name)
        return None

    # Prefer the mapped attribute, then fall back to a bare `language` or
    # `language_<name>` so grammars with either convention work.
    lang_attr = (
        getattr(mod, attr_name, None)
        or getattr(mod, "language", None)
        or getattr(mod, f"language_{lang_name}", None)
    )
    if lang_attr is None:
        logger.warning("%s exposes no usable language function (tried %r)", module_name, attr_name)
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

_TRIVIAL_CALLBACK_MAX_TOKENS = 30


def _is_trivial_callback(node: Node, source_bytes: bytes) -> bool:
    """True for small anonymous callbacks passed as call arguments.

    e.g. ``arr.map(x => x.id)``, ``.catch(() => {})``, ``(s) => s.foo`` Zustand
    selectors, ``report.answers.map((a, idx) => ({ ...a }))``. These are ~21% of
    nodes on a real React codebase and are pure retrieval noise. Judged by token
    count (not line span — trivial callbacks are often wrapped across 3-5 lines).
    Substantial callbacks (a real multi-line useEffect, >30 tokens) and
    named/assigned arrows (``const f = () =>``, parent is a declarator) are kept.
    """
    if node.type not in ("arrow_function", "function_expression"):
        return False
    parent = node.parent
    if parent is None or parent.type != "arguments":
        return False
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    return count_tokens(text) < _TRIVIAL_CALLBACK_MAX_TOKENS


def _decode(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_name(node: Node, source_bytes: bytes) -> str:
    """Extract a human-readable name from a definition node.

    Handles the modern JS/TS patterns the naive version missed:
      - ``const Foo = () => {}`` / ``const Foo = function(){}`` (name on the
        parent ``variable_declarator``)
      - object methods / assignments (``foo: () => {}``)
      - exported defaults
    Without this, anonymous arrow/function expressions get their *parameters*
    as a "name", which poisons both call-graph resolution and relevance scoring.
    """
    # 1. Explicit name field (function_definition, class_definition, method, ...)
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _decode(name_node, source_bytes)

    # 2. A direct identifier child
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
            return _decode(child, source_bytes)

    # 3. Anonymous function/arrow assigned to a variable, property, or field:
    #    pull the name from the enclosing declarator/pair.
    parent = node.parent
    if parent is not None and parent.type in (
        "variable_declarator", "pair", "assignment_expression",
        "public_field_definition", "field_definition", "property_signature",
    ):
        pn = parent.child_by_field_name("name") or parent.child_by_field_name("key") \
            or parent.child_by_field_name("left")
        if pn is not None:
            return _decode(pn, source_bytes).strip()
        for child in parent.children:
            if child.type in ("identifier", "property_identifier"):
                return _decode(child, source_bytes)

    # 3b. Anonymous function/arrow passed as a call argument (callbacks such as
    #     useEffect(...), arr.map(...), Zustand create((set) => ...)): name it
    #     after the enclosing call instead of the useless "() => {".
    if parent is not None and parent.type == "arguments":
        call = parent.parent
        if call is not None and call.type in ("call_expression", "new_expression"):
            fn = call.child_by_field_name("function")
            if fn is not None:
                callee = _decode(fn, source_bytes).split("\n")[0].strip()[:60]
                if callee:
                    return f"{callee}()callback"

    # 4. Last resort: first line, trimmed
    return _decode(node, source_bytes).split("\n")[0][:80]


_MAX_CONTENT_CHARS = int(os.environ.get("FITTOK_MAX_NODE_CHARS", "8000"))


def _safe_content(source_bytes: bytes, node: Node, max_chars: int | None = None) -> str:
    """Extract node content, truncated for storage.

    The cap is configurable via FITTOK_MAX_NODE_CHARS (default 8000,
    up from 2000) so large functions aren't gutted before relevance scoring.
    """
    limit = max_chars if max_chars is not None else _MAX_CONTENT_CHARS
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    if len(text) > limit:
        text = text[:limit] + "\n// ... [truncated]"
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
        "tsx": {"import_statement", "require_clause"},
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
            if _is_trivial_callback(ts_node, source_bytes):
                continue  # skip small anonymous callbacks (retrieval noise)
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

    # Module-level assignments → CONSTANT nodes (e.g. ``MODEL_NAME = "..."``).
    # Top-level ONLY (direct children of the module root): locals inside a
    # function/class body are retrieval noise. Making these first-class nodes is
    # what lets the optimizer return a referenced constant as a dependency
    # instead of leaving the model to re-open the file to see it.
    const_stmt_types = _CONST_STMT_TYPES.get(lang_name)
    if const_stmt_types:
        for child in tree.root_node.children:
            # Top-level statements only. Python wraps assignments in an
            # expression_statement node (grammar-version dependent); JS/TS
            # lexical/variable declarations are direct statements. Locals inside
            # function/class bodies live deeper (block > …) and are never reached.
            stmts = child.children if child.type == "expression_statement" else [child]
            for stmt in stmts:
                if stmt.type not in const_stmt_types:
                    continue
                # Build (name, content-bearing node) pairs for this statement.
                if lang_name == "python":
                    left = stmt.child_by_field_name("left")
                    if left is None or left.type != "identifier":
                        continue  # skip tuple unpacking, attribute/index targets
                    pairs = [(_decode(left, source_bytes), stmt)]
                else:  # JS/TS: one or more variable_declarator children
                    pairs = []
                    for dec in stmt.children:
                        if dec.type != "variable_declarator":
                            continue
                        nm = dec.child_by_field_name("name")
                        if nm is None or nm.type != "identifier":
                            continue
                        val = dec.child_by_field_name("value")
                        if val is not None and val.type in (
                            "arrow_function", "function", "function_expression",
                            "class", "class_declaration",
                        ):
                            continue  # already captured as a FUNCTION/CLASS node
                        pairs.append((_decode(nm, source_bytes), dec))
                for name, content_node in pairs:
                    if not name or name.startswith("_"):
                        continue  # skip private / dunder throwaways
                    content = _safe_content(source_bytes, content_node)
                    node_id = f"{NodeType.CONSTANT.value}:{rel_path}:{name}:{content_node.start_point[0]}"
                    const_node = GraphNode(
                        id=node_id,
                        type=NodeType.CONSTANT,
                        name=name,
                        file=rel_path,
                        line_start=content_node.start_point[0] + 1,
                        line_end=content_node.end_point[0] + 1,
                        content=content,
                        token_count=count_tokens(content),
                    )
                    nodes.append(const_node)
                    definitions[name] = const_node
                    edges.append(GraphEdge(source=file_id, target=node_id, type=EdgeType.CONTAINS))

    # Import edges (placeholders)
    for imp in import_targets:
        edges.append(GraphEdge(
            source=file_id,
            target=imp,  # placeholder
            type=EdgeType.IMPORTS,
        ))

    # Fallback: if the file produced no definition nodes (config-like files,
    # JSX-heavy components, or constructs we don't extract), index the file
    # body itself so the file is never invisible to retrieval/scoring.
    # (Generated stubs like *.d.ts are excluded earlier, at the walk level.)
    if len(nodes) == 1:  # only the file node was added
        body = _safe_content(source_bytes, tree.root_node)
        file_node.content = body
        file_node.token_count = count_tokens(body)

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

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in filenames:
            filepath = Path(dirpath) / filename
            if filepath.suffix not in _EXT_TO_LANG:
                continue
            if _is_generated_file(filepath):
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


def _get_code_files(root_path: str) -> list[Path]:
    """Collect all parseable code files under root."""
    root = Path(root_path).resolve()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix in _EXT_TO_LANG and not _is_generated_file(p):
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
