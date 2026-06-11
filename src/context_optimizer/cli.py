"""Command-line interface — lets the package do its job WITHOUT an MCP client.

Subcommands:
    context-optimizer serve              Run the MCP server (stdio) — for AI clients.
    context-optimizer index <path>       Pre-build graph + embeddings for a repo.
    context-optimizer query <path> "<q>" Print the most relevant code for a query.

`query` is the standalone equivalent of the MCP `optimize_context` tool: it
returns the same readable, budget-bounded context, straight to your terminal.
With no subcommand, defaults to `serve` (so existing MCP registrations that
launch the bare `context-optimizer` command keep working).
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="context-optimizer",
        description="Retrieve the most relevant source code for a query, within a token budget.",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="Run the MCP server over stdio (for AI clients).")

    p_index = sub.add_parser("index", help="Pre-build the knowledge graph + embeddings for a repo.")
    p_index.add_argument("path", help="Path to the codebase to index.")

    p_query = sub.add_parser("query", help="Print the most relevant code for a query (no MCP needed).")
    p_query.add_argument("path", help="Path to the codebase.")
    p_query.add_argument("query", help="Natural-language question about the codebase.")
    p_query.add_argument("--budget", type=int, default=0,
                         help="Token budget (0 = adaptive, the default).")
    p_query.add_argument("--json", action="store_true",
                         help="Emit the full result dict as JSON.")

    args = parser.parse_args(argv)

    # Default (or `serve`) → run the MCP server.
    if args.cmd in (None, "serve"):
        from .server import main as serve_main
        serve_main()
        return

    # For human-facing commands, keep stderr clean (INFO logs are server-mode noise).
    import logging
    logging.disable(logging.INFO)

    if args.cmd == "index":
        from .indexer import index_codebase
        r = index_codebase(args.path)
        print(f"Indexed {r['nodes']} nodes / {r['edges']} edges, {r['embedded']} embeddings "
              f"(parse {r['parse_s']}s, embed {r['embed_s']}s). Cached.")
        return

    if args.cmd == "query":
        from .server import optimize_context_tool
        res = optimize_context_tool(args.path, args.query, args.budget)
        if "error" in res:
            print(f"Error: {res['error']}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(res, indent=2))
            return
        # Human mode: stats to stderr, the actual context to stdout (pipe-friendly).
        s, sv = res["slurp_stats"], res["savings"]
        print(
            f"# {s['selected_nodes']} nodes · {s['tokens_sent']} tokens · "
            f"confidence {s['confidence_label']} ({s['confidence']}) · {sv['summary']}",
            file=sys.stderr,
        )
        print(res["optimized_context"])
        return


if __name__ == "__main__":
    main()
