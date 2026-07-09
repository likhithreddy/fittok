"""Interactive knowledge-graph visualization (pyvis HTML).

Kept in its own module so the CLI (`fittok graph`) and the MCP `show_graph`
tool share ONE renderer — change the graph look here and both update. pyvis is
imported lazily inside `build_graph_html`, so importing this module does not
require pyvis (the caller checks/handles the optional dependency).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _ntype(t):
    return t.value if hasattr(t, "value") else str(t)


_NODE_STYLE = {
    "function": ("#6366f1", "dot"),
    "method":   ("#9C27B0", "triangle"),
    "class":    ("#FF9800", "box"),
    "constant": ("#06b6d4", "star"),
    "file":     ("#607D8B", "square"),
    "module":   ("#607D8B", "square"),
    "import":   ("#795548", "dot"),
}

_EDGE_COLOR = {
    "calls":      "#3b82f6",
    "imports":    "#64748b",
    "references": "#a855f7",
    "contains":   "#334155",
    "inherits":   "#f97316",
}

_TYPE_TAG = {
    "function": "fn", "method": "meth", "class": "class",
    "constant": "const", "file": "file", "module": "mod", "import": "imp",
}


def build_graph_html(graph, root_path: str, highlight_ids=None, query: str | None = None) -> str:
    """Render `graph` to an interactive pyvis HTML file and return its path.

    Caller is responsible for opening the returned path in a browser.
    """
    from pyvis.network import Network

    highlight_ids = highlight_ids or set()
    total = len(graph.nodes)

    net = Network(height="100vh", width="100%", directed=True, notebook=False,
                  bgcolor="#07090f", font_color="#e2e8f0")
    net.barnes_hut(gravity=-8000, spring_length=120)

    # Degree (connectivity) per node → size hubs bigger so they stand out.
    degree: dict[str, int] = {}
    for e in graph.edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1
    max_deg = max(degree.values()) if degree else 1

    for node in graph.nodes:
        nt = _ntype(node.type)
        color, shape = _NODE_STYLE.get(nt, ("#6366f1", "dot"))
        is_hi = node.id in highlight_ids
        deg = degree.get(node.id, 0)
        size = (9 + 16 * (deg / max_deg)) if deg else 7   # scales with connectivity
        if is_hi:
            size = max(size, 28)
        tag = _TYPE_TAG.get(nt, "?")
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
            "color": _EDGE_COLOR.get(et, "#1e293b"),
            "title": et, "dashes": (et == "contains"), "type": et,
        })

    name = os.path.basename(root_path.rstrip("/")) or root_path
    query_note = f" — highlighting '{query}'" if query else ""
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
            f'<b style="color:#6366f1">fittok graph</b> · {name}{query_note}<br>'
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
        return f.name
