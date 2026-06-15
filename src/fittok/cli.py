"""Command-line interface — lets the package do its job WITHOUT an MCP client.

Subcommands:
    fittok serve                    Run the MCP server (stdio) — for AI clients.
    fittok index [path]             Pre-build graph + embeddings (default: cwd).
    fittok query "<q>"              Ask a question; answer via LLM if API key is set.
    fittok query "<q>" --code       Return the raw relevant code slice instead.
    fittok query "<q>" --budget N   Limit the slice to N tokens before sending to LLM.

LLM selection (first key found wins):
    ANTHROPIC_API_KEY  →  claude-haiku-4-5  (recommended)
    OPENAI_API_KEY     →  gpt-4o-mini
    neither set        →  falls back to --code behavior with a hint

With no subcommand, defaults to `serve` so existing MCP registrations keep working.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _detect_llm() -> tuple[str, str] | None:
    """Return (provider, model) for the first available API key, or None."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("anthropic", "claude-haiku-4-5-20251001")
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", "gpt-4o-mini")
    return None


def _ask_llm(provider: str, model: str, query: str, context: str) -> None:
    """Stream an LLM answer to stdout given the optimized code context."""
    system = (
        "You are a senior software engineer. "
        "Answer the user's question using ONLY the code context provided. "
        "Be concise and precise. Do not make up code that is not in the context."
    )
    prompt = f"Code context:\n\n{context}\n\nQuestion: {query}"

    if provider == "anthropic":
        try:
            import anthropic  # type: ignore
        except ImportError:
            print("Install anthropic: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic()
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        print()  # newline after stream

    elif provider == "openai":
        try:
            import openai  # type: ignore
        except ImportError:
            print("Install openai: pip install openai", file=sys.stderr)
            sys.exit(1)
        client = openai.OpenAI()
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            max_tokens=1024,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
        print()


# ── Graph command ─────────────────────────────────────────────────────────────

def _cmd_graph(args) -> None:
    try:
        from pyvis.network import Network  # noqa: F401
    except ImportError:
        print(
            "⚠  pyvis is required for graph visualization.\n"
            "   Install it with:  uv pip install 'fittok[ui]'\n"
            "   or:               pip install pyvis",
            file=sys.stderr,
        )
        sys.exit(1)

    import tempfile
    import webbrowser
    from pathlib import Path
    from .graphify import parse_codebase
    from .cache import get_cached_graph, set_cached_graph

    path = Path(args.path or os.getcwd()).resolve()

    # Load or build the graph
    graph = get_cached_graph(str(path))
    if graph is None:
        print(f"Indexing {path} …", file=sys.stderr)
        graph = parse_codebase(str(path))
        set_cached_graph(str(path), graph)

    total = len(graph.nodes)
    print(f"Graph: {total} nodes, {len(graph.edges)} edges", file=sys.stderr)

    # Compute relevance scores if --query given
    highlight_ids: set[str] = set()
    if args.query:
        from .slurp import score_nodes
        scores = score_nodes(graph, args.query)
        # top 5% or at least top 10 nodes
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        cutoff = max(10, len(sorted_scores) // 20)
        highlight_ids = {nid for nid, _ in sorted_scores[:cutoff]}
        print(f"Highlighted {len(highlight_ids)} nodes relevant to: {args.query!r}", file=sys.stderr)

    # Build pyvis network
    from pyvis.network import Network

    net = Network(height="100vh", width="100%", directed=True, notebook=False,
                  bgcolor="#07090f", font_color="#e2e8f0")
    net.barnes_hut(gravity=-8000, spring_length=120)

    color_map = {
        "file":     "#607D8B",
        "function": "#6366f1",
        "class":    "#FF9800",
        "method":   "#9C27B0",
        "module":   "#607D8B",
        "import":   "#795548",
    }

    for node in graph.nodes:
        is_highlighted = node.id in highlight_ids
        base_color = color_map.get(node.type.value if hasattr(node.type, "value") else str(node.type), "#6366f1")
        color = "#22c55e" if is_highlighted else base_color
        size  = 28 if is_highlighted else 12
        label = node.name if len(node.name) < 28 else node.name[:25] + "…"
        title = f"{node.type.value if hasattr(node.type, 'value') else node.type}: {node.name}\n{node.file}:{node.line_start}"
        net.add_node(node.id, label=label, color=color, title=title, size=size)

    for edge in graph.edges:
        net.add_edge(edge.source, edge.target, color="#1e293b")

    # Add a legend note to the page title
    query_note = f" — highlighting '{args.query}'" if args.query else ""
    net.set_options("""{
      "nodes": {"font": {"size": 11}},
      "edges": {"smooth": {"type": "dynamic"}},
      "physics": {"stabilization": {"iterations": 150}}
    }""")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        # Inject a simple legend bar into the HTML
        html = Path(f.name).read_text()
        legend = (
            f'<div style="position:fixed;top:10px;left:10px;background:#0d1117;'
            f'border:1px solid #1e293b;border-radius:8px;padding:10px 16px;'
            f'font-family:monospace;font-size:12px;color:#e2e8f0;z-index:999">'
            f'<b style="color:#6366f1">fittok graph</b> · {path.name}{query_note}<br>'
            f'<span style="color:#6366f1">● function</span> &nbsp;'
            f'<span style="color:#FF9800">● class</span> &nbsp;'
            f'<span style="color:#9C27B0">● method</span>'
            + (f' &nbsp;<span style="color:#22c55e">● relevant</span>' if highlight_ids else "")
            + f'<br><span style="color:#475569">{total} nodes · {len(graph.edges)} edges</span>'
            f'</div>'
        )
        html = html.replace("<body>", "<body>" + legend, 1)
        Path(f.name).write_text(html)
        out_path = f.name

    print(f"Opening graph in browser …", file=sys.stderr)
    webbrowser.open(f"file://{out_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fittok",
        description="Retrieve the most relevant source code for a query, within a token budget.",
    )
    from . import __version__
    parser.add_argument("--version", action="version", version=f"fittok {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="Run the MCP server over stdio (for AI clients).")

    p_index = sub.add_parser("index", help="Pre-build the knowledge graph + embeddings for a repo.")
    p_index.add_argument("path", nargs="?", default=None,
                         help="Path to the codebase (default: current directory).")

    p_graph = sub.add_parser("graph", help="Open an interactive knowledge graph in the browser.")
    p_graph.add_argument("path", nargs="?", default=None,
                         help="Path to the codebase (default: current directory).")
    p_graph.add_argument("--query", default=None,
                         help="Highlight nodes relevant to this query.")

    p_query = sub.add_parser("query", help="Ask a question about a codebase.")
    p_query.add_argument("path", nargs="?", default=None,
                         help="Path to the codebase (default: current directory).")
    p_query.add_argument("query", help="Natural-language question about the codebase.")
    p_query.add_argument("--budget", type=int, default=0,
                         help="Token budget for the code slice (0 = adaptive).")
    p_query.add_argument("--code", action="store_true",
                         help="Return the raw relevant code instead of an LLM answer.")
    p_query.add_argument("--json", action="store_true",
                         help="Emit the full result dict as JSON (implies --code).")

    args = parser.parse_args(argv)

    if args.cmd in (None, "serve"):
        from .server import main as serve_main
        serve_main()
        return

    import logging
    logging.disable(logging.INFO)

    if args.cmd == "index":
        from .indexer import index_codebase
        path = args.path or os.getcwd()
        r = index_codebase(path)
        print(f"Indexed {r['nodes']} nodes / {r['edges']} edges, {r['embedded']} embeddings "
              f"(parse {r['parse_s']}s, embed {r['embed_s']}s). Cached.")
        return

    if args.cmd == "graph":
        _cmd_graph(args)
        return

    if args.cmd == "query":
        from .server import optimize_context_tool

        # If path arg doesn't exist on disk, treat it as the query (path omitted).
        if args.path and not os.path.exists(args.path):
            args.query = args.path
            args.path = None
        path = args.path or os.getcwd()

        res = optimize_context_tool(path, args.query, args.budget)
        if "error" in res:
            print(f"Error: {res['error']}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(res, indent=2))
            return

        s, sv = res["slurp_stats"], res["savings"]
        # Always print slice stats to stderr so they don't pollute piped output.
        print(
            f"# {s['selected_nodes']} nodes · {s['tokens_sent']} tokens · "
            f"confidence {s['confidence_label']} · {sv['summary']}",
            file=sys.stderr,
        )

        if args.code:
            # Raw code mode — current default behaviour, now opt-in via --code.
            print(res["optimized_context"])
            return

        # ── LLM answer mode (default) ──────────────────────────────────────
        llm = _detect_llm()
        if llm is None:
            print(
                "\n⚠  No LLM API key found — returning raw relevant code instead.\n"
                "   To get an LLM answer, set one of:\n"
                "     export ANTHROPIC_API_KEY='sk-ant-...'\n"
                "     export OPENAI_API_KEY='sk-...'\n"
                "   Then re-run the same command.\n"
                "   (Use --code to suppress this message and always get raw code.)\n",
                file=sys.stderr,
            )
            print(res["optimized_context"])
            return

        provider, model = llm
        print(f"# Answering via {model} …", file=sys.stderr)
        _ask_llm(provider, model, args.query, res["optimized_context"])
        return


if __name__ == "__main__":
    main()
