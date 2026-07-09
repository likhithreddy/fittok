"""File watcher for incremental graph updates.

Uses watchdog for cross-platform file watching. Gracefully degrades
if watchdog is not installed.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

from .graphify import _EXT_TO_LANG, update_graph as _update_graph
from .models import KnowledgeGraph

logger = logging.getLogger(__name__)

# Active watchers: root_path -> WatcherState
_watchers: dict[str, WatcherState] = {}

# Cap concurrent watchers so auto-watch (one per queried codebase) can't grow
# memory/threads without bound. Oldest is evicted when the cap is hit.
_MAX_WATCHERS = 8


class WatcherState:
    """Manages a background file watcher for a codebase."""

    def __init__(self, root_path: str, graph: KnowledgeGraph):
        self.root_path = root_path
        self.graph = graph
        self.observer: Optional[object] = None
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._changed_files: set[str] = set()
        self._lock = threading.Lock()
        self._generation = 0  # bumped on reset/replace; flush discards stale results

    def start(self) -> None:
        """Start watching for file changes."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
        except ImportError:
            logger.warning("watchdog not installed — watch mode unavailable")
            return

        root = Path(self.root_path).resolve()

        class Handler(FileSystemEventHandler):
            def __init__(self, state: WatcherState):
                self.state = state

            def on_modified(self, event):
                self._record(event.src_path)

            def on_created(self, event):
                self._record(event.src_path)

            def on_deleted(self, event):
                # Recording the path is enough: update_graph removes nodes whose
                # file is in the changed set, and the re-parse step skips files
                # that no longer exist. Without this, deleted files linger as
                # orphan nodes indefinitely.
                self._record(event.src_path)

            def on_moved(self, event):
                # Rename: drop the old path's nodes, parse the new path.
                self._record(event.src_path)
                self._record(event.dest_path)

            def _record(self, path: str):
                p = Path(path)
                if p.suffix in _EXT_TO_LANG:
                    with self.state._lock:
                        self.state._changed_files.add(str(p))

        self.observer = Observer()
        self.observer.schedule(Handler(self), str(root), recursive=True)
        self.observer.daemon = True
        self.observer.start()

        # Start flush thread
        self.thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.thread.start()

        logger.info("Watch started for %s", self.root_path)

    def stop(self) -> None:
        """Stop watching."""
        self._stop_event.set()
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
        logger.info("Watch stopped for %s", self.root_path)

    def _flush_loop(self) -> None:
        """Periodically flush changed files into the graph."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=2.0)
            self.flush()

    def flush(self) -> bool:
        """Process any pending changes immediately.

        Returns True if the graph was updated. Called by the background flush
        loop AND by the query path (via ``_live_graph``) so a query sees the
        latest edits without waiting for the 2s tick. The lock makes it safe
        to call from both threads — only one drains the pending set.
        """
        with self._lock:
            changed = list(self._changed_files)
            self._changed_files.clear()

        if not changed:
            return False
        gen_before = self._generation
        try:
            new_graph = _update_graph(self.graph, self.root_path, changed)
        except Exception:
            logger.warning("Failed to update graph", exc_info=True)
            return False
        with self._lock:
            # If a reset/replace happened while we were re-parsing, discard this
            # stale incremental result — persisting it would silently undo the reset.
            if self._generation != gen_before:
                return False
            self.graph = new_graph
            self._generation += 1
        logger.info("Updated graph: %d changed files, now %d nodes",
                    len(changed), len(self.graph.nodes))
        self._persist()
        return True

    def _persist(self) -> None:
        """Write the live graph back to disk + cache so the file-based tools
        (query_graph / get_graph_stats / diff_graph) and a server restart see
        the incremental update instead of a stale pre-edit snapshot."""
        try:
            from .cache import set_cached_graph, graph_output_path
            from .graphify import save_graph
            save_graph(self.graph, graph_output_path(self.root_path))
            set_cached_graph(self.root_path, self.graph)
        except Exception:
            logger.warning("Failed to persist incremental graph update", exc_info=True)

    def replace_graph(self, new_graph) -> None:
        """Swap the live graph (e.g. after a forced full re-parse via
        reset_graph), dropping pending changes and re-persisting."""
        with self._lock:
            self.graph = new_graph
            self._changed_files.clear()
            self._generation += 1  # invalidate any in-flight flush
        self._persist()

    def get_stats(self) -> dict:
        """Return watcher statistics."""
        return {
            "root_path": self.root_path,
            "active": self.observer is not None and self.observer.is_alive(),
            "pending_changes": len(self._changed_files),
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
        }


def start_watch(root_path: str, graph: KnowledgeGraph) -> dict:
    """Start watching a codebase for changes.

    Returns watcher info dict.
    """
    resolved = str(Path(root_path).resolve())
    if resolved in _watchers:
        return {"error": f"Already watching: {resolved}"}

    # Evict the oldest watcher if at capacity (auto-watch can otherwise spawn
    # one per queried codebase with no limit).
    while len(_watchers) >= _MAX_WATCHERS:
        oldest = next(iter(_watchers))
        evicted = _watchers.pop(oldest, None)
        if evicted is not None:
            try:
                evicted.stop()
            except Exception:
                logger.warning("Failed to stop evicted watcher for %s", oldest, exc_info=True)
            logger.info("Evicted oldest watcher (cap %d) to make room", _MAX_WATCHERS)

    state = WatcherState(resolved, graph)
    state.start()

    if state.observer is None:
        return {"error": "watchdog not installed. Install with: pip install watchdog"}

    _watchers[resolved] = state
    return {
        "watching": resolved,
        "active": True,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
    }


def stop_watch(root_path: str) -> dict:
    """Stop watching a codebase."""
    resolved = str(Path(root_path).resolve())
    state = _watchers.pop(resolved, None)
    if state is None:
        return {"error": f"Not watching: {resolved}"}
    state.stop()
    return {"stopped": resolved}


def get_watcher(root_path: str) -> Optional[WatcherState]:
    """Get the watcher state for a path."""
    return _watchers.get(str(Path(root_path).resolve()))
