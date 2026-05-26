"""Tests for RAG evaluation metrics."""

from evaluate import compute_recall_at_k, compute_precision_at_k, compute_faithfulness


def test_recall_all_found():
    assert compute_recall_at_k(["RAG.md", "Chunking.md"], ["RAG.md", "Chunking.md"], k=5) == 1.0


def test_recall_partial():
    assert compute_recall_at_k(["RAG.md", "Embeddings.md"], ["RAG.md", "Chunking.md"], k=5) == 0.5


def test_recall_none_found():
    assert compute_recall_at_k(["Python.md", "Java.md"], ["RAG.md"], k=5) == 0.0


def test_recall_no_expectations():
    assert compute_recall_at_k(["RAG.md"], [], k=5) == 1.0


def test_recall_respects_k():
    assert compute_recall_at_k(["RAG.md", "Chunking.md"], ["Chunking.md"], k=1) == 0.0


def test_recall_substring_match():
    assert compute_recall_at_k(["Notes/RAG.md", "notes/Chunking.md"], ["RAG.md"], k=5) == 1.0


def test_precision_all_relevant():
    assert compute_precision_at_k(["RAG.md", "Chunking.md"], ["RAG.md", "Chunking.md"], k=5) == 1.0


def test_precision_partial():
    assert compute_precision_at_k(["RAG.md", "Python.md"], ["RAG.md"], k=5) == 0.5


def test_precision_empty_retrieved():
    assert compute_precision_at_k([], ["RAG.md"], k=5) == 0.0


def test_precision_no_expectations():
    assert compute_precision_at_k(["RAG.md"], [], k=5) == 0.5


def test_faithfulness_high(sample_chunks):
    score = compute_faithfulness("RAG combines retrieval with generation to improve accuracy.", sample_chunks)
    assert score > 0.5


def test_faithfulness_low(sample_chunks):
    score = compute_faithfulness("The weather today is sunny and warm.", sample_chunks)
    assert score < 0.3


def test_faithfulness_empty_answer():
    assert compute_faithfulness("", [{"text": "something"}]) == 0.0


def test_faithfulness_no_chunks():
    assert compute_faithfulness("RAG is useful.", []) == 0.0


def test_faithfulness_citations_stripped(sample_chunks):
    score = compute_faithfulness("RAG improves accuracy [RAG.md §chunk_0].", sample_chunks)
    assert score > 0
