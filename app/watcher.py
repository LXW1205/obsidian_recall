"""
watcher.py — File Watcher

Monitors the vault folder for file changes and automatically triggers re-indexing.
Runs as a background daemon thread so the Streamlit UI stays responsive.
"""

import logging
import os
import threading
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ingest import delete_file_chunks, index_file, reindex_file

logger = logging.getLogger("obsidian_recall.watcher")

VAULT_PATH = os.getenv("VAULT_PATH", "/app/notes")


class VaultEventHandler(FileSystemEventHandler):
    """Handles file system events in the Obsidian vault."""

    def __init__(self):
        self.last_modified = {}
        self.debounce_seconds = 2.0  # Wait before processing to avoid duplicate events

    def _is_markdown(self, path: str) -> bool:
        return path.endswith(".md")

    def _should_process(self, path: str) -> bool:
        """Debounce: only process if enough time has passed since last event."""
        now = time.time()
        last = self.last_modified.get(path, 0)
        if now - last < self.debounce_seconds:
            return False
        self.last_modified[path] = now
        return True

    def on_modified(self, event):
        if event.is_directory:
            return
        if not self._is_markdown(event.src_path):
            return
        if not self._should_process(event.src_path):
            return

        logger.info(f"File modified: {event.src_path}")
        try:
            result = reindex_file(event.src_path)
            if result.get("success"):
                logger.info(f"Re-indexed: {event.src_path}")
            else:
                logger.warning(f"Re-index failed: {event.src_path} — {result.get('error')}")
        except Exception as e:
            logger.error(f"Error re-indexing {event.src_path}: {e}")

    def on_created(self, event):
        if event.is_directory:
            return
        if not self._is_markdown(event.src_path):
            return
        if not self._should_process(event.src_path):
            return

        logger.info(f"File created: {event.src_path}")
        try:
            result = index_file(event.src_path, force=True)
            if result.get("success"):
                logger.info(f"Indexed new file: {event.src_path}")
            else:
                logger.warning(f"Index failed: {event.src_path} — {result.get('error')}")
        except Exception as e:
            logger.error(f"Error indexing {event.src_path}: {e}")

    def on_deleted(self, event):
        if event.is_directory:
            return
        if not self._is_markdown(event.src_path):
            return

        logger.info(f"File deleted: {event.src_path}")
        try:
            delete_file_chunks(event.src_path)
            logger.info(f"Removed chunks for: {event.src_path}")
        except Exception as e:
            logger.error(f"Error deleting chunks for {event.src_path}: {e}")


def start_watcher(vault_path: str = None) -> threading.Thread:
    """
    Start the file watcher as a background daemon thread.

    Returns the thread object (daemon thread, will exit when main process exits).
    """
    vault_path = vault_path or VAULT_PATH

    if not os.path.exists(vault_path):
        logger.warning(f"Vault path does not exist: {vault_path}. Watcher not started.")
        return None

    event_handler = VaultEventHandler()
    observer = Observer()
    observer.schedule(event_handler, vault_path, recursive=True)
    observer.start()

    def run_observer():
        logger.info(f"Watcher started on {vault_path}")
        try:
            while observer.is_alive():
                observer.join(timeout=1)
        except KeyboardInterrupt:
            logger.info("Watcher stopped by keyboard interrupt")
        finally:
            observer.stop()
            observer.join()

    thread = threading.Thread(target=run_observer, daemon=True)
    thread.start()
    return thread


def stop_watcher(observer: Observer):
    """Stop the file watcher."""
    if observer:
        observer.stop()
        observer.join()
        logger.info("Watcher stopped")
