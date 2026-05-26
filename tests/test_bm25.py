"""Tests for BM25 keyword scoring index."""

from bm25 import BM25Index


def test_empty_index():
    idx = BM25Index()
    idx.build()
    assert idx.size == 0
    import pytest
    with pytest.raises(RuntimeError, match="not built"):
        idx.score("anything")


def test_single_document():
    idx = BM25Index()
    idx.add_document("doc1", "the cat sat on the mat")
    idx.build()
    assert idx.size == 1
    scores = idx.score("cat mat")
    assert len(scores) == 1
    assert scores[0][0] == 0
    assert scores[0][1] > 0


def test_multiple_documents():
    idx = BM25Index()
    idx.add_document("doc1", "python programming language")
    idx.add_document("doc2", "java programming language")
    idx.add_document("doc3", "machine learning algorithms")
    idx.build()
    scores = idx.score("python programming")
    assert idx.get_doc_id(scores[0][0]) == "doc1"


def test_query_no_match():
    idx = BM25Index()
    idx.add_document("doc1", "the quick brown fox")
    idx.add_document("doc2", "jumps over the lazy dog")
    idx.build()
    scores = idx.score("zzzzzzz")
    assert all(score == 0 for _, score in scores)


def test_build_not_called():
    idx = BM25Index()
    idx.add_document("doc1", "some text")
    import pytest
    with pytest.raises(RuntimeError, match="not built"):
        idx.score("text")


def test_get_doc_id():
    idx = BM25Index()
    idx.add_document("doc_a", "first document")
    idx.add_document("doc_b", "second document")
    assert idx.get_doc_id(0) == "doc_a"
    assert idx.get_doc_id(1) == "doc_b"


def test_tokenization():
    idx = BM25Index()
    t = idx._tokenize
    assert t("Hello World!") == ["hello", "world"]
    assert t("RAG-is-great") == ["rag", "is", "great"]
    assert t("Note 1: Introduction") == ["note", "1", "introduction"]
    assert t("") == []
    assert t("UPPERCASE") == ["uppercase"]


def test_relevance_ordering():
    idx = BM25Index()
    idx.add_document("doc1", "python is a programming language used for web development data science and automation")
    idx.add_document("doc2", "javascript is a programming language used for web development")
    idx.add_document("doc3", "machine learning is a subset of artificial intelligence")
    idx.build()
    scores = idx.score("python programming")
    assert idx.get_doc_id(scores[0][0]) == "doc1"
    scores2 = idx.score("machine learning artificial intelligence")
    assert idx.get_doc_id(scores2[0][0]) == "doc3"
