"""Usage example for context-optimizer.

Demonstrates all three approaches:
  1. Step-by-step pipeline
  2. One-call optimize_context
  3. MCP server mode
"""

import sys
import tempfile
from pathlib import Path


def create_sample_project(base_dir: str) -> str:
    """Create a small sample project for demonstration."""
    base = Path(base_dir)

    (base / "auth.py").write_text(
        '''"""Authentication module."""

import hashlib
import os


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Hash a password with a random salt."""
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return key.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, salt_hex: str) -> bool:
    """Verify a password against stored hash and salt."""
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return key.hex() == stored_hash


class UserAuth:
    """Handles user authentication."""

    def __init__(self, db_connection):
        self.db = db_connection

    def login(self, username: str, password: str) -> dict | None:
        """Authenticate a user and return session data."""
        user = self.db.find_user(username)
        if not user:
            return None
        if verify_password(password, user["password_hash"], user["salt"]):
            return {"user_id": user["id"], "token": self._create_token(user)}
        return None

    def _create_token(self, user: dict) -> str:
        """Create a session token."""
        import secrets
        return secrets.token_hex(32)
'''
    )

    (base / "app.py").write_text(
        '''"""Main application."""

from auth import UserAuth


class Application:
    def __init__(self, db):
        self.auth = UserAuth(db)

    def handle_login(self, request):
        username = request.get("username")
        password = request.get("password")
        session = self.auth.login(username, password)
        if session:
            return {"status": "ok", "token": session["token"]}
        return {"status": "error", "message": "Invalid credentials"}
'''
    )

    return str(base)


def demo_step_by_step(project_path: str):
    """Step-by-step pipeline: parse → query → compress."""
    print("=" * 60)
    print("Step-by-step Pipeline Demo")
    print("=" * 60)

    from context_optimizer.graphify import parse_codebase, save_graph
    from context_optimizer.slurp import query_graph
    from context_optimizer.llmlingua_wrapper import compress_context

    # Step 1: Parse
    print("\n1. Parsing codebase...")
    graph = parse_codebase(project_path)
    save_graph(graph, str(Path(project_path) / "graph.json"))
    print(f"   Nodes: {graph.metadata.total_nodes}, Edges: {graph.metadata.total_edges}")

    # Step 2: Query
    print("\n2. Querying graph...")
    query = "How does user authentication work?"
    markdown, node_count, tokens = query_graph(graph, query, token_budget=2000)
    print(f"   Selected {node_count} nodes ({tokens} tokens)")

    # Step 3: Compress
    print("\n3. Compressing...")
    result = compress_context(markdown, query, target_tokens=200)
    print(f"   {result['original_tokens']} → {result['compressed_tokens']} tokens "
          f"({result['compression_ratio']:.1%} ratio)")
    print(f"\n   Compressed output:\n   {'-' * 40}")
    for line in result["compressed"].split("\n"):
        print(f"   {line}")


def demo_one_call(project_path: str):
    """One-call optimize_context pipeline."""
    print("\n" + "=" * 60)
    print("One-call Pipeline Demo")
    print("=" * 60)

    from context_optimizer.server import optimize_context_tool

    result = optimize_context_tool(
        codebase_path=project_path,
        query="How does user authentication work?",
        token_budget=200,
    )

    print(f"\nGraph: {result['graph_stats']['total_nodes']} nodes, "
          f"{result['graph_stats']['total_edges']} edges")
    print(f"Slurp: {result['slurp_stats']['selected_nodes']} nodes selected")
    comp = result.get("compression_stats", {})
    if comp:
        print(f"Compression: {comp.get('original_tokens', '?')} → "
              f"{comp.get('compressed_tokens', '?')} tokens")

    if "optimized_context" in result:
        print(f"\nOptimized context:\n{'-' * 40}")
        print(result["optimized_context"][:500])


def main():
    # Create temp project
    with tempfile.TemporaryDirectory(prefix="ctx-opt-demo-") as tmp:
        project_path = create_sample_project(tmp)

        if len(sys.argv) > 1 and sys.argv[1] == "server":
            print("Starting MCP server on stdio...")
            from context_optimizer.server import main as server_main
            server_main()
        else:
            demo_step_by_step(project_path)
            demo_one_call(project_path)


if __name__ == "__main__":
    main()
