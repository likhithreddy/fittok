"""Web UI for visualizing the context optimizer pipeline.

Uses Gradio for the dashboard and pyvis for interactive graph visualization.
Gracefully degrades if either is not installed.
"""

from __future__ import annotations

import html as _html
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _iframe_wrap(doc_html: str, height: str = "600px") -> str:
    """Embed a full HTML document in an iframe via srcdoc.

    Gradio's gr.HTML injects content with innerHTML, where <script> tags never
    execute — so pyvis (which draws the graph via vis-network JS) renders an
    empty box. An iframe's srcdoc runs scripts normally, so the graph appears.
    """
    srcdoc = _html.escape(doc_html, quote=True)
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:{height};border:none;background:#fff;"></iframe>'
    )


def _build_graph_html(graph) -> str:
    """Generate an interactive graph visualization as HTML using pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis not installed. Install with: pip install pyvis</p>"

    net = Network(height="600px", directed=True, notebook=False)
    net.barnes_hut()

    color_map = {
        "file": "#4CAF50",
        "function": "#2196F3",
        "class": "#FF9800",
        "method": "#9C27B0",
        "module": "#607D8B",
        "import": "#795548",
    }

    for node in graph.nodes:
        color = color_map.get(node.type.value, "#999")
        label = node.name if len(node.name) < 30 else node.name[:27] + "..."
        title = f"{node.type.value}: {node.name}\n{node.file}:{node.line_start}"
        net.add_node(node.id, label=label, color=color, title=title, size=15)

    for edge in graph.edges:
        net.add_edge(edge.source, edge.target)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return _iframe_wrap(Path(f.name).read_text(), "600px")


def _run_query(graph, query: str, token_budget: int) -> tuple[str, str, dict]:
    """Run a query and return (markdown, graph_html, stats)."""
    from .slurp import query_graph as _query_graph

    markdown, node_count, tokens_used = _query_graph(graph, query, token_budget)
    stats = {
        "selected_nodes": node_count,
        "tokens_used": tokens_used,
        "budget": token_budget,
        "compression": f"{tokens_used / max(token_budget, 1):.1%}",
    }

    # Build subgraph HTML from selected nodes
    try:
        from pyvis.network import Network
        from .models import NodeType

        net = Network(height="400px", directed=True, notebook=False)
        net.barnes_hut()

        # Find nodes mentioned in markdown
        selected_ids = set()
        for node in graph.nodes:
            if node.name in markdown and node.type != NodeType.FILE:
                selected_ids.add(node.id)

        for nid in selected_ids:
            node = next((n for n in graph.nodes if n.id == nid), None)
            if node:
                net.add_node(node.id, label=node.name, color="#2196F3", size=20,
                             title=f"{node.file}:{node.line_start}")

        for edge in graph.edges:
            if edge.source in selected_ids or edge.target in selected_ids:
                net.add_edge(edge.source, edge.target)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            net.save_graph(f.name)
            graph_html = _iframe_wrap(Path(f.name).read_text(), "400px")
    except ImportError:
        graph_html = ""

    return markdown, graph_html, stats


def create_ui(graph=None):
    """Create the Gradio UI. Returns the Blocks app."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio not installed. Install with: pip install gradio")

    state = {"graph": graph}

    def do_parse(codebase_path: str):
        from .graphify import parse_codebase
        g = parse_codebase(codebase_path)
        state["graph"] = g
        html = _build_graph_html(g)
        stats = f"Nodes: {g.metadata.total_nodes} | Edges: {g.metadata.total_edges}"
        return html, stats

    def do_query(query: str, budget: int):
        g = state.get("graph")
        if g is None:
            return "Parse a codebase first", "", {}
        md, html, stats = _run_query(g, query, budget)
        return md, html, json.dumps(stats, indent=2)

    with gr.Blocks(title="Fittok Visualizer") as app:
        gr.Markdown("# Fittok Context Optimizer")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                codebase_path = gr.Textbox(label="Codebase Path", placeholder="/path/to/code")
                parse_btn = gr.Button("Parse Codebase")
                parse_stats = gr.Textbox(label="Graph Stats", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Graph View")
                graph_html = gr.HTML()

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(label="Query", placeholder="How does auth work?")
                token_budget = gr.Slider(100, 10000, value=4000, step=100, label="Token Budget")
                query_btn = gr.Button("Query Graph")

            with gr.Column(scale=2):
                gr.Markdown("### Selected Nodes")
                result_md = gr.Markdown()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Selection Graph")
                subgraph_html = gr.HTML()
            with gr.Column():
                gr.Markdown("### Stats")
                query_stats = gr.JSON()

        parse_btn.click(do_parse, inputs=[codebase_path], outputs=[graph_html, parse_stats])
        query_btn.click(do_query, inputs=[query_input, token_budget],
                        outputs=[result_md, subgraph_html, query_stats])

    return app


def launch_ui(port: int = 8765, open_browser: bool = True, graph=None) -> dict:
    """Launch the web visualization UI in a background thread.

    Args:
        port: Port to serve on (default 8765).
        open_browser: Whether to open browser automatically.
        graph: Optional pre-loaded graph.

    Returns:
        Dict with launch info.
    """
    import threading

    try:
        app = create_ui(graph)
        # Run Gradio in a daemon thread so it doesn't block the MCP server
        t = threading.Thread(
            target=lambda: app.launch(
                server_port=port, inbrowser=open_browser,
                show_error=True, prevent_thread_lock=True,
            ),
            daemon=True,
        )
        t.start()
        return {"launched": True, "port": port, "url": f"http://localhost:{port}"}
    except ImportError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to launch UI: {e}"}
