"""
ingest.py — Ingestion Pipeline

Reads all .md files from the Obsidian vault, chunks them, embeds each chunk,
and stores in ChromaDB. Supports incremental indexing via MD5 hash tracking.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional

import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH", "/app/notes")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/chroma")
HASH_FILE = os.path.join(CHROMA_PATH, "index_hashes.json")

EMBEDDING_MODEL = "text-embedding-004"

# Lazy Google AI client initialization
_google_client = None


def get_google_client():
    """Get or create the Google AI client (lazy initialization)."""
    global _google_client
    if _google_client is None:
        _google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _google_client

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="obsidian_notes")


def compute_md5(filepath: str) -> str:
    """Compute MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_hashes() -> dict:
    """Load stored file hashes from JSON."""
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}


def save_hashes(hashes: dict):
    """Save file hashes to JSON."""
    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def strip_markdown(text: str) -> str:
    """Remove markdown syntax, keep plain text."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove wikilinks but keep text
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    # Remove headers markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove strikethrough
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # Remove blockquote markers
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\*{3,}$", "", text, flags=re.MULTILINE)
    # Remove list markers
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove frontmatter
    text = re.sub(r"^---\n[\s\S]*?\n---\n", "", text)
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(text: str) -> list[tuple[int, str]]:
    """Extract markdown headers with their character positions.
    
    Returns list of (char_position, header_text) tuples.
    """
    sections = []
    for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, flags=re.MULTILINE):
        sections.append((match.start(), match.group(2).strip()))
    return sections


def get_section_at_position(sections: list[tuple[int, str]], pos: int) -> str:
    """Get the current section header for a given character position."""
    current_section = ""
    for section_pos, section_text in sections:
        if section_pos <= pos:
            current_section = section_text
        else:
            break
    return current_section


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[tuple[str, str]]:
    """Split text into overlapping chunks with section tracking.
    
    Returns list of (chunk_text, section) tuples.
    """
    if not text:
        return []

    sections = extract_sections(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                chunk = chunk[: break_point + 1].strip()
                end = start + break_point + 1
        if chunk.strip():
            section = get_section_at_position(sections, start)
            chunks.append((chunk.strip(), section))
        start = end - overlap if end < len(text) else len(text)
    return chunks


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a text chunk using Google text-embedding-004."""
    client = get_google_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def delete_file_chunks(filepath: str):
    """Delete all chunks for a specific file from ChromaDB."""
    relative_path = os.path.relpath(filepath, VAULT_PATH)
    collection.delete(where={"source": relative_path})


def index_file(filepath: str, force: bool = False) -> dict:
    """
    Index a single file into ChromaDB.

    Returns dict with status info.
    """
    relative_path = os.path.relpath(filepath, VAULT_PATH)

    # Read file content
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        return {"success": False, "error": f"Could not read {filepath}: {e}"}

    if not raw_text.strip():
        return {"success": True, "skipped": True, "reason": "Empty file"}

    # Check if file has changed
    current_hash = compute_md5(filepath)
    hashes = load_hashes()

    if not force and hashes.get(relative_path) == current_hash:
        return {"success": True, "skipped": True, "reason": "No changes"}

    # Clean and chunk
    clean_text = strip_markdown(raw_text)
    chunks_with_sections = chunk_text(clean_text)

    if not chunks_with_sections:
        return {"success": True, "skipped": True, "reason": "No content after cleaning"}

    # Delete old chunks for this file
    delete_file_chunks(filepath)

    # Generate structured chunk_ids and metadata
    filename_base = os.path.splitext(os.path.basename(filepath))[0]
    chunk_ids = []
    chunk_texts = []
    metadatas = []

    for i, (chunk_text_content, section) in enumerate(chunks_with_sections):
        chunk_id = f"chunk_{filename_base}_{i}"
        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk_text_content)
        metadatas.append({
            "source": relative_path,
            "filename": os.path.basename(filepath),
            "chunk_id": chunk_id,
            "chunk_index": i,
            "section": section,
        })

    # Generate embeddings
    embeddings = []
    for chunk_text_content in chunk_texts:
        embeddings.append(get_embedding(chunk_text_content))

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas,
    )

    # Update hash
    hashes[relative_path] = current_hash
    save_hashes(hashes)

    return {
        "success": True,
        "chunks": len(chunks),
        "filepath": relative_path,
    }


def reindex_file(filepath: str) -> dict:
    """Re-index a single file (called by watcher)."""
    return index_file(filepath, force=True)


def full_reindex(force: bool = False) -> dict:
    """
    Index all .md files in the vault.

    Returns summary dict.
    """
    if not os.path.exists(VAULT_PATH):
        return {"success": False, "error": f"Vault path not found: {VAULT_PATH}"}

    results = {"indexed": 0, "skipped": 0, "errors": 0, "total_chunks": 0, "files": []}

    md_files = []
    for root, _, files in os.walk(VAULT_PATH):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))

    for filepath in md_files:
        result = index_file(filepath, force=force)
        if result.get("success"):
            if result.get("skipped"):
                results["skipped"] += 1
            else:
                results["indexed"] += 1
                results["total_chunks"] += result.get("chunks", 0)
                results["files"].append(result.get("filepath", filepath))
        else:
            results["errors"] += 1
            results["files"].append(
                {"filepath": filepath, "error": result.get("error")}
            )

    results["total_files"] = len(md_files)
    return results


def get_stats() -> dict:
    """Get current indexing statistics."""
    hashes = load_hashes()
    return {
        "total_files_indexed": len(hashes),
        "total_chunks": collection.count(),
    }
