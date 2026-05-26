"""Tests for citation grounding validation."""

from grounding import extract_keywords, check_grounding, ALLOWED_ANSWER_LABELS


def test_allowed_labels():
    assert "grounded_answer" in ALLOWED_ANSWER_LABELS
    assert "insufficient_context" in ALLOWED_ANSWER_LABELS
    assert "conflicting_context" in ALLOWED_ANSWER_LABELS


def test_extract_keywords():
    kw = extract_keywords("RAG combines retrieval with generation")
    assert "rag" in kw
    assert "retrieval" in kw
    assert "generation" in kw
    assert "the" not in kw
    assert "a" not in kw


def test_extract_keywords_empty():
    assert extract_keywords("") == set()
    assert extract_keywords("the a an is") == set()


def test_check_grounding_valid_citation(sample_chunks):
    citations = ["[RAG.md §chunk_0]"]
    answer = "RAG combines retrieval with generation to improve LLM accuracy [RAG.md §chunk_0]."
    result = check_grounding(answer, citations, sample_chunks)
    assert result["all_citations_valid"] is True
    assert result["grounding_score"] > 0
    assert result["citation_checks"][0]["valid"] is True


def test_check_grounding_nonexistent_chunk(sample_chunks):
    citations = ["[Ghost.md §chunk_999]"]
    result = check_grounding("text", citations, sample_chunks)
    assert result["all_citations_valid"] is False
    assert result["citation_checks"][0]["valid"] is False
    assert result["citation_checks"][0]["chunk_exists"] is False


def test_check_grounding_no_citations(sample_chunks):
    result = check_grounding("Some answer.", [], sample_chunks)
    assert result["all_citations_valid"] is True
    assert result["grounding_score"] == 0.0


def test_check_grounding_malformed_citation(sample_chunks):
    citations = ["[bad citation no chunk id]"]
    result = check_grounding("text", citations, sample_chunks)
    assert result["all_citations_valid"] is False


def test_check_grounding_multiple_citations_mixed(sample_chunks):
    citations = ["[RAG.md §chunk_0]", "[Ghost.md §ghost_chunk]"]
    answer = "RAG improves accuracy [RAG.md §chunk_0] but ghost is missing [Ghost.md §ghost_chunk]."
    result = check_grounding(answer, citations, sample_chunks)
    assert result["all_citations_valid"] is False
    assert result["citation_checks"][0]["valid"] is True
    assert result["citation_checks"][1]["valid"] is False
