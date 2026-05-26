"""Shared fixtures for the test suite."""

import sys
from pathlib import Path

# Add the app directory to sys.path so tests can import modules
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import pytest


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk_0",
            "doc_title": "RAG.md",
            "source": "RAG.md",
            "section": "Introduction",
            "text": (
                "Retrieval Augmented Generation (RAG) is a technique that "
                "combines retrieval from a knowledge base with text generation. "
                "It is used to improve the accuracy of LLM responses."
            ),
        },
        {
            "chunk_id": "chunk_1",
            "doc_title": "Chunking.md",
            "source": "Chunking.md",
            "section": "Methods",
            "text": (
                "Chunking splits documents into smaller pieces for indexing. "
                "Common strategies include sentence splitting, paragraph splitting, "
                "and semantic chunking."
            ),
        },
        {
            "chunk_id": "chunk_2",
            "doc_title": "Embeddings.md",
            "source": "Embeddings/Embeddings.md",
            "section": "Overview",
            "text": (
                "Embeddings are vector representations of text. "
                "They capture semantic meaning and enable similarity search. "
                "Popular models include text-embedding-004 and sentence-transformers."
            ),
        },
    ]
