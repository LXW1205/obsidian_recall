"""Tests for query module — prompt building and answer parsing."""

from query import build_prompt, parse_answer_response


def test_build_prompt_contains_question(sample_chunks):
    prompt = build_prompt("What is RAG?", sample_chunks[:1])
    assert "QUESTION: What is RAG?" in prompt


def test_build_prompt_contains_context(sample_chunks):
    prompt = build_prompt("What is RAG?", sample_chunks[:1])
    assert "CONTEXT:" in prompt
    assert "Retrieval Augmented Generation" in prompt


def test_build_prompt_includes_citation(sample_chunks):
    prompt = build_prompt("test", sample_chunks[:1])
    assert "[RAG.md §chunk_0]" in prompt


def test_build_prompt_includes_section(sample_chunks):
    prompt = build_prompt("test", sample_chunks[:1])
    assert "Section: Introduction" in prompt


def test_build_prompt_multiple_chunks(sample_chunks):
    prompt = build_prompt("test", sample_chunks)
    assert "[RAG.md §chunk_0]" in prompt
    assert "[Chunking.md §chunk_1]" in prompt
    assert "[Embeddings.md §chunk_2]" in prompt


def test_build_prompt_has_instructions(sample_chunks):
    prompt = build_prompt("test", sample_chunks[:1])
    assert "INSTRUCTIONS" in prompt
    assert "inline citation" in prompt


def test_parse_grounded_answer(sample_chunks):
    answer = (
        "RAG is a technique that combines retrieval with generation "
        "to improve LLM accuracy [RAG.md §chunk_0]."
    )
    result = parse_answer_response("What is RAG?", answer, sample_chunks)
    assert result["answer_label"] == "grounded_answer"
    assert result["citations"] == ["[RAG.md §chunk_0]"]
    assert result["used_chunk_ids"] == ["chunk_0"]


def test_parse_insufficient_context_keyword(sample_chunks):
    result = parse_answer_response("What is Python?", "I couldn't find this in your notes.", sample_chunks)
    assert result["answer_label"] == "insufficient_context"
    assert result["citations"] == []


def test_parse_insufficient_context_no_citations(sample_chunks):
    answer = "RAG is a technique that combines retrieval with generation."
    result = parse_answer_response("What is RAG?", answer, sample_chunks)
    assert result["answer_label"] == "insufficient_context"
    assert result["citations"] == []


def test_parse_wrong_chunk_id(sample_chunks):
    answer = "RAG is great [RAG.md §nonexistent_chunk]."
    result = parse_answer_response("What is RAG?", answer, sample_chunks)
    assert result["answer_label"] == "insufficient_context"


def test_parse_skips_duplicate_ids(sample_chunks):
    answer = (
        "RAG is a technique [RAG.md §chunk_0]. "
        "It combines retrieval with generation [RAG.md §chunk_0]."
    )
    result = parse_answer_response("What is RAG?", answer, sample_chunks[:1])
    assert len(result["citations"]) == 2
    assert result["used_chunk_ids"] == ["chunk_0"]


def test_parse_multiple_chunks(sample_chunks):
    answer = (
        "RAG is a technique [RAG.md §chunk_0]. "
        "Chunking splits text [Chunking.md §chunk_1]."
    )
    result = parse_answer_response("What are RAG and chunking?", answer, sample_chunks[:2])
    assert result["answer_label"] == "grounded_answer"
    assert len(result["citations"]) == 2
