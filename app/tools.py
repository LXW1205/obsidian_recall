"""
tools.py — File Operation Tools for Agent Mode

All write operations the agent can call. Each function returns a result dict:
{
    "success": bool,
    "action": "description of what was done",
    "filepath": "affected file path",
    "error": "error message if failed"
}
"""

import os
import re
import shutil
from pathlib import Path
from typing import Optional

from ingest import VAULT_PATH, delete_file_chunks, index_file, reindex_file, collection, get_embedding


def _resolve_path(filepath: str) -> str:
    """Resolve a relative filepath against VAULT_PATH."""
    if os.path.isabs(filepath):
        return filepath
    return os.path.join(VAULT_PATH, filepath)


def _relative_path(filepath: str) -> str:
    """Get path relative to VAULT_PATH."""
    return os.path.relpath(filepath, VAULT_PATH)


def read_note(filepath: str) -> dict:
    """Read content of a note."""
    try:
        full_path = _resolve_path(filepath)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "success": True,
            "action": f"Read note: {_relative_path(full_path)}",
            "filepath": _relative_path(full_path),
            "content": content,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Read note: {filepath}",
            "filepath": filepath,
            "error": str(e),
        }


def list_notes(folder: Optional[str] = None) -> dict:
    """List all notes, optionally filtered by folder."""
    try:
        search_path = _resolve_path(folder) if folder else VAULT_PATH
        if not os.path.exists(search_path):
            return {
                "success": False,
                "action": f"List notes in {folder or 'vault'}",
                "filepath": folder or VAULT_PATH,
                "error": f"Path does not exist: {search_path}",
                "notes": [],
            }

        notes = []
        for root, _, files in os.walk(search_path):
            for file in files:
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    notes.append(_relative_path(full_path))

        notes.sort()
        return {
            "success": True,
            "action": f"Listed {len(notes)} notes" + (f" in {folder}" if folder else ""),
            "filepath": folder or VAULT_PATH,
            "notes": notes,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"List notes in {folder or 'vault'}",
            "filepath": folder or VAULT_PATH,
            "error": str(e),
            "notes": [],
        }


def create_note(filepath: str, content: str) -> dict:
    """Create a new note."""
    try:
        full_path = _resolve_path(filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Index the new note
        index_file(full_path, force=True)

        return {
            "success": True,
            "action": f"Created note: {_relative_path(full_path)}",
            "filepath": _relative_path(full_path),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Create note: {filepath}",
            "filepath": filepath,
            "error": str(e),
        }


def edit_note(filepath: str, new_content: str) -> dict:
    """Overwrite a note's content."""
    try:
        full_path = _resolve_path(filepath)
        if not os.path.exists(full_path):
            return {
                "success": False,
                "action": f"Edit note: {filepath}",
                "filepath": filepath,
                "error": "File does not exist",
            }

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        # Re-index the edited note
        reindex_file(full_path)

        return {
            "success": True,
            "action": f"Edited note: {_relative_path(full_path)}",
            "filepath": _relative_path(full_path),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Edit note: {filepath}",
            "filepath": filepath,
            "error": str(e),
        }


def rename_note(old_path: str, new_name: str) -> dict:
    """Rename a note file. new_name should include .md extension."""
    try:
        old_full = _resolve_path(old_path)
        if not os.path.exists(old_full):
            return {
                "success": False,
                "action": f"Rename note: {old_path}",
                "filepath": old_path,
                "error": "File does not exist",
            }

        # Ensure new_name has .md extension
        if not new_name.endswith(".md"):
            new_name += ".md"

        parent = os.path.dirname(old_full)
        new_full = os.path.join(parent, new_name)

        if os.path.exists(new_full):
            return {
                "success": False,
                "action": f"Rename note: {old_path} -> {new_name}",
                "filepath": old_path,
                "error": f"Target already exists: {new_name}",
            }

        os.rename(old_full, new_full)

        # Delete old chunks, index new file
        delete_file_chunks(old_full)
        index_file(new_full, force=True)

        return {
            "success": True,
            "action": f"Renamed: {_relative_path(old_full)} -> {_relative_path(new_full)}",
            "filepath": _relative_path(new_full),
            "old_filepath": _relative_path(old_full),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Rename note: {old_path} -> {new_name}",
            "filepath": old_path,
            "error": str(e),
        }


def move_note(filepath: str, target_folder: str) -> dict:
    """Move a note to a different folder."""
    try:
        old_full = _resolve_path(filepath)
        if not os.path.exists(old_full):
            return {
                "success": False,
                "action": f"Move note: {filepath}",
                "filepath": filepath,
                "error": "File does not exist",
            }

        target_full = _resolve_path(target_folder)
        os.makedirs(target_full, exist_ok=True)

        filename = os.path.basename(old_full)
        new_full = os.path.join(target_full, filename)

        if os.path.exists(new_full):
            return {
                "success": False,
                "action": f"Move note: {filepath} -> {target_folder}",
                "filepath": filepath,
                "error": f"Target already exists: {new_full}",
            }

        shutil.move(old_full, new_full)

        # Delete old chunks, index new file
        delete_file_chunks(old_full)
        index_file(new_full, force=True)

        return {
            "success": True,
            "action": f"Moved: {_relative_path(old_full)} -> {_relative_path(new_full)}",
            "filepath": _relative_path(new_full),
            "old_filepath": _relative_path(old_full),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Move note: {filepath} -> {target_folder}",
            "filepath": filepath,
            "error": str(e),
        }


def add_tag(filepath: str, tag: str) -> dict:
    """Add a #tag to a note's frontmatter or top of file."""
    try:
        full_path = _resolve_path(filepath)
        if not os.path.exists(full_path):
            return {
                "success": False,
                "action": f"Add tag to: {filepath}",
                "filepath": filepath,
                "error": "File does not exist",
            }

        # Normalize tag
        if not tag.startswith("#"):
            tag = "#" + tag
        tag = tag.lower().replace(" ", "-")

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if tag already exists
        if re.search(rf"(?<!\w){re.escape(tag)}(?!\w)", content):
            return {
                "success": True,
                "action": f"Tag {tag} already exists in {_relative_path(full_path)}",
                "filepath": _relative_path(full_path),
                "error": None,
            }

        # Check if frontmatter exists
        frontmatter_match = re.match(r"^---\n([\s\S]*?)\n---\n", content)
        if frontmatter_match:
            # Insert tag into existing frontmatter
            tags_line = f"tags: [{tag}]"
            new_content = content.replace("---\n", f"---\n{tags_line}\n", 1)
        else:
            # Create new frontmatter with tag
            new_content = f"---\ntags: [{tag}]\n---\n\n{content}"

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        # Re-index
        reindex_file(full_path)

        return {
            "success": True,
            "action": f"Added tag {tag} to {_relative_path(full_path)}",
            "filepath": _relative_path(full_path),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Add tag to: {filepath}",
            "filepath": filepath,
            "error": str(e),
        }


def delete_note(filepath: str) -> dict:
    """Delete a note. High-risk action."""
    try:
        full_path = _resolve_path(filepath)
        if not os.path.exists(full_path):
            return {
                "success": False,
                "action": f"Delete note: {filepath}",
                "filepath": filepath,
                "error": "File does not exist",
            }

        os.remove(full_path)

        # Remove from ChromaDB
        delete_file_chunks(full_path)

        return {
            "success": True,
            "action": f"Deleted note: {_relative_path(full_path)}",
            "filepath": _relative_path(full_path),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Delete note: {filepath}",
            "filepath": filepath,
            "error": str(e),
        }


def merge_notes(source_paths: list[str], target_path: str) -> dict:
    """Merge multiple notes into one."""
    try:
        merged_content = []
        for src in source_paths:
            full_src = _resolve_path(src)
            if not os.path.exists(full_src):
                return {
                    "success": False,
                    "action": f"Merge notes into {target_path}",
                    "filepath": target_path,
                    "error": f"Source not found: {src}",
                }
            with open(full_src, "r", encoding="utf-8") as f:
                content = f.read()
            merged_content.append(f"## From: {os.path.basename(full_src)}\n\n{content}")

        full_target = _resolve_path(target_path)
        os.makedirs(os.path.dirname(full_target), exist_ok=True)

        with open(full_target, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(merged_content))

        # Delete old notes
        for src in source_paths:
            full_src = _resolve_path(src)
            os.remove(full_src)
            delete_file_chunks(full_src)

        # Index merged note
        index_file(full_target, force=True)

        return {
            "success": True,
            "action": f"Merged {len(source_paths)} notes into {_relative_path(full_target)}",
            "filepath": _relative_path(full_target),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Merge notes into {target_path}",
            "filepath": target_path,
            "error": str(e),
        }


def find_duplicates(threshold: float = 0.92) -> dict:
    """Find semantically similar notes via ChromaDB embedding similarity."""
    try:
        # Get all documents and their metadata
        all_docs = collection.get(include=["documents", "metadatas"])
        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"]
        ids = all_docs["ids"]

        if len(documents) < 2:
            return {
                "success": True,
                "action": "Find duplicates",
                "filepath": VAULT_PATH,
                "duplicates": [],
                "error": None,
            }

        # Group by source file
        file_chunks = {}
        for i, meta in enumerate(metadatas):
            source = meta.get("source", "unknown")
            if source not in file_chunks:
                file_chunks[source] = []
            file_chunks[source].append(i)

        # Get embeddings for each file's first chunk (representative)
        file_embeddings = {}
        for source, chunk_indices in file_chunks.items():
            # Use the first chunk as representative
            rep_idx = chunk_indices[0]
            result = collection.query(
                query_embeddings=[all_docs["embeddings"][rep_idx]],
                n_results=min(10, len(documents)),
                include=["distances", "metadatas"],
            )
            file_embeddings[source] = {
                "embedding": all_docs["embeddings"][rep_idx],
                "result": result,
            }

        # Find similar pairs
        duplicates = []
        checked = set()
        sources = list(file_embeddings.keys())

        for i, src_a in enumerate(sources):
            for src_b in sources[i + 1:]:
                pair_key = tuple(sorted([src_a, src_b]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                # Query ChromaDB to find similarity
                emb_a = file_embeddings[src_a]["embedding"]
                result = collection.query(
                    query_embeddings=[emb_a],
                    n_results=1,
                    include=["metadatas", "distances"],
                )

                if result["metadatas"][0]:
                    closest_meta = result["metadatas"][0][0]
                    closest_source = closest_meta.get("source", "")
                    if closest_source == src_b and result["distances"][0][0] < (1 - threshold):
                        duplicates.append({
                            "file_a": src_a,
                            "file_b": src_b,
                            "similarity": round(1 - result["distances"][0][0], 3),
                        })

        return {
            "success": True,
            "action": f"Found {len(duplicates)} potential duplicate pairs",
            "filepath": VAULT_PATH,
            "duplicates": duplicates,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": "Find duplicates",
            "filepath": VAULT_PATH,
            "error": str(e),
            "duplicates": [],
        }


def find_broken_links() -> dict:
    """Scan vault for broken [[wikilinks]]."""
    try:
        notes_result = list_notes()
        if not notes_result["success"]:
            return notes_result

        broken_links = []

        for note_path in notes_result["notes"]:
            read_result = read_note(note_path)
            if not read_result["success"]:
                continue

            content = read_result["content"]
            # Find all wikilinks
            wikilinks = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", content)

            for link in wikilinks:
                # Check if target exists
                link_clean = link.strip()
                if not link_clean.endswith(".md"):
                    link_clean += ".md"

                # Try to find the file
                target_found = False
                for existing_note in notes_result["notes"]:
                    if existing_note.endswith(link_clean) or os.path.basename(existing_note) == link_clean:
                        target_found = True
                        break

                if not target_found:
                    broken_links.append({
                        "source": note_path,
                        "link": link,
                        "target": link_clean,
                    })

        return {
            "success": True,
            "action": f"Found {len(broken_links)} broken links",
            "filepath": VAULT_PATH,
            "broken_links": broken_links,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": "Find broken links",
            "filepath": VAULT_PATH,
            "error": str(e),
            "broken_links": [],
        }


def search_notes(query: str, n_results: int = 5) -> dict:
    """Semantic search via ChromaDB."""
    try:
        embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            hits.append({
                "source": meta.get("source", "Unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "similarity": round(1 - dist, 3),
            })

        return {
            "success": True,
            "action": f"Semantic search: {query}",
            "filepath": VAULT_PATH,
            "results": hits,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": f"Semantic search: {query}",
            "filepath": VAULT_PATH,
            "error": str(e),
            "results": [],
        }
