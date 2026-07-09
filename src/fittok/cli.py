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

    def _ntype(t):
        return t.value if hasattr(t, "value") else str(t)

    # Node visual identity by type: (color, shape). Shape adds a second visual
    # dimension beyond color (and helps colour-blind readers).
    node_style = {
        "function": ("#6366f1", "dot"),
        "method":   ("#9C27B0", "triangle"),
        "class":    ("#FF9800", "box"),
        "constant": ("#06b6d4", "star"),
        "file":     ("#607D8B", "square"),
        "module":   ("#607D8B", "square"),
        "import":   ("#795548", "dot"),
    }

    # Edge colour by relationship type, so the graph reads at a glance instead
    # of looking like a uniform hairball.
    edge_color = {
        "calls":      "#3b82f6",   # blue
        "imports":    "#64748b",   # slate
        "references": "#a855f7",   # purple
        "contains":   "#334155",   # dim — structural containment (very common)
        "inherits":   "#f97316",   # orange
    }

    # Degree (connectivity) per node → size hubs bigger so they stand out.
    degree: dict[str, int] = {}
    for e in graph.edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1
    max_deg = max(degree.values()) if degree else 1

    # Short text tag prefixed on each label, e.g. [fn] optimize_context.
    type_tag = {
        "function": "fn", "method": "meth", "class": "class",
        "constant": "const", "file": "file", "module": "mod", "import": "imp",
    }

    for node in graph.nodes:
        nt = _ntype(node.type)
        color, shape = node_style.get(nt, ("#6366f1", "dot"))
        is_hi = node.id in highlight_ids
        deg = degree.get(node.id, 0)
        size = (9 + 16 * (deg / max_deg)) if deg else 7   # scales with connectivity
        if is_hi:
            size = max(size, 28)
        tag = type_tag.get(nt, "?")
        name = node.name if len(node.name) < 22 else node.name[:19] + "…"
        short_file = os.path.basename(node.file)
        label = f"[{tag}] {name}\n{short_file}"
        loc = f"{short_file}:{node.line_start}" + (
            f"-{node.line_end}" if node.line_end and node.line_end != node.line_start else "")
        # PLAIN-TEXT tooltip (rendered multi-line via the .vis-tooltip CSS injected
        # below). Plain text avoids the raw-HTML-tags bug in the popup.
        first_line = next((ln for ln in node.content.splitlines() if ln.strip()), "")[:100]
        meta_bits = [f"connections: {deg}", f"file: {short_file}"]
        if node.token_count:
            meta_bits.append(f"tokens: {node.token_count}")
        if is_hi:
            meta_bits.append("relevant ★")
        title = f"{nt} · {node.name}\n{loc}\n{' · '.join(meta_bits)}"
        if first_line:
            title += f"\n{first_line}"
        # Append directly so each node carries `type` + `file` for the JS filters.
        net.nodes.append({
            "id": node.id, "label": label, "title": title,
            "color": color, "shape": shape, "size": size,
            "borderWidth": (2 if is_hi else 1),
            "type": nt, "file": short_file, "rel": is_hi,
        })

    for edge in graph.edges:
        et = _ntype(edge.type)
        # Dashed for structural "contains" so it recedes vs. semantic edges.
        net.edges.append({
            "from": edge.source, "to": edge.target, "arrows": "to",
            "color": edge_color.get(et, "#1e293b"),
            "title": et, "dashes": (et == "contains"), "type": et,
        })

    # Legend (top-left) + filter controls (top-right)
    query_note = f" — highlighting '{args.query}'" if args.query else ""
    net.set_options("""{
      "nodes": {"font": {"size": 11}},
      "edges": {"smooth": {"type": "dynamic"}},
      "physics": {"stabilization": {"iterations": 150}}
    }""")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html = Path(f.name).read_text()
        legend = (
            f'<div style="position:fixed;top:10px;left:10px;background:#0d1117;'
            f'border:1px solid #1e293b;border-radius:8px;padding:10px 16px;'
            f'font-family:monospace;font-size:12px;color:#e2e8f0;z-index:999;'
            f'max-height:90vh;overflow:auto">'
            f'<b style="color:#6366f1">fittok graph</b> · {path.name}{query_note}<br>'
            f'<span style="color:#6366f1">● function</span> &nbsp;'
            f'<span style="color:#9C27B0">▲ method</span> &nbsp;'
            f'<span style="color:#FF9800">▣ class</span> &nbsp;'
            f'<span style="color:#06b6d4">★ constant</span><br>'
            f'<span style="color:#3b82f6">— calls</span> &nbsp;'
            f'<span style="color:#a855f7">— references</span> &nbsp;'
            f'<span style="color:#f97316">— inherits</span> &nbsp;'
            f'<span style="color:#64748b">— imports</span> &nbsp;'
            f'<span style="color:#334155">┄ contains</span>'
            + (f'<br><span style="color:#22c55e">● relevant (enlarged)</span>' if highlight_ids else "")
            + f'<br><span style="color:#6366f1;font-size:9px">●</span>'
            f'<span style="color:#475569"> few conn.</span> &nbsp;&nbsp;'
            f'<span style="color:#6366f1;font-size:16px">●</span>'
            f'<span style="color:#475569"> hub</span>'
            + f'<br><span style="color:#475569">{total} nodes · {len(graph.edges)} edges'
            f' · hover = details · click node = isolate</span>'
            f'</div>'
        )
        rel_cb = (
            '<label style="margin-right:8px"><input type="checkbox" id="ft-rel" '
            'style="width:auto;margin-right:4px;vertical-align:middle">relevant only</label>'
            if highlight_ids else ''
        )
        controls = (
            "<style>"
            ".vis-tooltip { white-space: pre-line !important; max-width: 440px; "
            "font-family: monospace; font-size: 12px; line-height: 1.5; }"
            "#ft-controls { position:fixed; top:10px; right:10px; background:#0d1117; "
            "border:1px solid #1e293b; border-radius:8px; padding:10px 14px; "
            "font-family:monospace; font-size:12px; color:#e2e8f0; z-index:999; max-width:270px; }"
            "#ft-controls input[type=text] { background:#0b0f17; color:#e2e8f0; "
            "border:1px solid #334155; border-radius:6px; padding:4px 6px; "
            "font-family:monospace; font-size:12px; width:100%; box-sizing:border-box; }"
            "#ft-controls select, #ft-controls button { background:#0b0f17; color:#e2e8f0; "
            "border:1px solid #334155; border-radius:6px; padding:3px 6px; "
            "font-family:monospace; font-size:12px; margin-left:4px; cursor:pointer; }"
            "#ft-controls label { color:#94a3b8; }"
            "#ft-controls .row { margin-top:6px; }"
            "#ft-controls .hint { color:#475569; margin-top:6px; font-size:11px; }"
            "</style>"
            "<div id=\"ft-controls\">"
            "<input id=\"ft-search\" type=\"text\" placeholder=\"search node name, press Enter\">"
            "<div class=\"row\"><label>type<select id=\"ft-type\"><option value=\"\">all</option></select></label>"
            "<label>file<select id=\"ft-file\"><option value=\"\">all</option></select></label></div>"
            "<div class=\"row\">" + rel_cb + "<button id=\"ft-reset\">reset</button></div>"
            "<div class=\"hint\">click node = isolate 1-hop · click background = restore</div>"
            "</div>"
            "<script>(function(){"
            "function init(){"
            "if(typeof network==='undefined'||!network||!network.body){setTimeout(init,60);return;}"
            "var ns=nodes.get();var types=[],files=[];"
            "ns.forEach(function(n){if(n.type&&types.indexOf(n.type)<0)types.push(n.type);"
            "if(n.file&&files.indexOf(n.file)<0)files.push(n.file);});"
            "types.sort();files.sort();"
            "var tSel=document.getElementById('ft-type'),fSel=document.getElementById('ft-file');"
            "types.forEach(function(t){var o=document.createElement('option');o.value=t;o.textContent=t;tSel.appendChild(o);});"
            "files.forEach(function(f){var o=document.createElement('option');o.value=f;o.textContent=f;fSel.appendChild(o);});"
            "var relCb=document.getElementById('ft-rel');"
            "var isolateSet=null;"
            "function vis(n){if(isolateSet&&!isolateSet.has(n.id))return false;"
            "if(relCb&&relCb.checked&&!n.rel)return false;"
            "if(tSel.value&&n.type!==tSel.value)return false;"
            "if(fSel.value&&n.file!==fSel.value)return false;return true;}"
            "function apply(){var v={};"
            "ns.forEach(function(n){v[n.id]=vis(n);});"
            "nodes.update(ns.map(function(n){return{id:n.id,hidden:!v[n.id]};}));"
            "var es=edges.get();"
            "edges.update(es.map(function(e){return{id:e.id,hidden:!(v[e.from]&&v[e.to])};}));}"
            "tSel.addEventListener('change',apply);fSel.addEventListener('change',apply);"
            "if(relCb)relCb.addEventListener('change',apply);"
            "document.getElementById('ft-reset').addEventListener('click',function(){tSel.value='';fSel.value='';if(relCb)relCb.checked=false;isolateSet=null;apply();});"
            "var search=document.getElementById('ft-search');"
            "search.addEventListener('keydown',function(ev){if(ev.key!=='Enter')return;"
            "var q=search.value.trim().toLowerCase();if(!q)return;"
            "var m=ns.filter(function(n){return((n.label||'').toLowerCase().indexOf(q)>=0)||((n.id||'').toLowerCase().indexOf(q)>=0);});"
            "if(m.length){isolateSet=null;apply();network.focus(m[0].id,{scale:1.3,animation:{duration:400,easingFunction:'easeInOutQuad'}});network.selectNodes([m[0].id]);}});"
            "network.on('click',function(params){"
            "if(params.nodes&&params.nodes.length>0){var start=params.nodes[0];var nb={};nb[start]=true;"
            "edges.get().forEach(function(e){if(e.from===start)nb[e.to]=true;if(e.to===start)nb[e.from]=true;});"
            "isolateSet=new Set(Object.keys(nb));apply();"
            "network.focus(start,{scale:1.2,animation:{duration:300,easingFunction:'easeInOutQuad'}});}"
            "else{isolateSet=null;apply();}});"
            "}init();})();</script>"
        )
        html = html.replace("<body>", "<body>" + legend + controls, 1)
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
