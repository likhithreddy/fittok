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
            with self._lock:
                changed = list(self._changed_files)
                self._changed_files.clear()

            if changed:
                try:
                    self.graph = _update_graph(self.graph, self.root_path, changed)
                    logger.info("Updated graph: %d changed files, now %d nodes",
                                len(changed), len(self.graph.nodes))
                except Exception:
                    logger.warning("Failed to update graph", exc_info=True)

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
