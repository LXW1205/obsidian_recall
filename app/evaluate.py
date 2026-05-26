"""
evaluate.py — RAG evaluation with Recall@K, Precision@K, and Faithfulness.

Runs a test suite of queries against the vault and produces a report:
- Recall@K: Are the expected sources in top K results?
- Precision@K: What % of retrieved sources are relevant?
- Faithfulness: Does the answer align with retrieved context?

Usage:
    from evaluate import run_evaluation
    results = run_evaluation(queries)
"""

import os
import re
from typing import Any

from grounding import extract_keywords
from ingest import VAULT_PATH
from query import ask, get_retriever


def compute_recall_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    k: int = 5,
) -> float:
    """
    Compute Recall@K: fraction of expected sources found in top K results.
    
    If expected_sources is empty, returns 1.0 (no expectations to check).
    """
    if not expected_sources:
        return 1.0
    
    retrieved_top_k = retrieved_sources[:k]
    matched = sum(1 for src in expected_sources if any(src.lower() in r.lower() for r in retrieved_top_k))
    return matched / len(expected_sources)


def compute_precision_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    k: int = 5,
) -> float:
    """
    Compute Precision@K: fraction of top K retrieved sources that are relevant.
    
    If expected_sources is empty, returns 0.5 (neutral).
    """
    if not expected_sources:
        return 0.5
    
    retrieved_top_k = retrieved_sources[:k]
    if not retrieved_top_k:
        return 0.0
    relevant = sum(1 for r in retrieved_top_k if any(src.lower() in r.lower() for src in expected_sources))
    return relevant / min(k, len(retrieved_top_k))


def compute_faithfulness(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> float:
    """
    Compute Faithfulness: keyword overlap between answer and retrieved chunks.
    
    Returns a score from 0.0 to 1.0.
    """
    if not answer or not retrieved_chunks:
        return 0.0
    
    # Extract keywords from answer (excluding citations)
    answer_clean = re.sub(r'\[.*?\]', '', answer)
    answer_keywords = extract_keywords(answer_clean)
    
    if not answer_keywords:
        return 0.0
    
    # Extract keywords from all retrieved chunks
    chunk_keywords = set()
    for chunk in retrieved_chunks:
        chunk_keywords.update(extract_keywords(chunk.get("text", "")))
    
    # Compute overlap
    overlap = answer_keywords & chunk_keywords
    return len(overlap) / len(answer_keywords)


def run_evaluation(
    queries: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """
    Run evaluation on a list of test queries.
    
    Each query dict should have:
        - question: str
        - expected_sources: list[str] (optional)
    
    Returns:
        {
            "results": [...],  # per-query results
            "summary": {
                "total_queries": int,
                "recall_at_k": float,
                "precision_at_k": float,
                "faithfulness": float,
            }
        }
    """
    results = []
    total_recall = 0.0
    total_precision = 0.0
    total_faithfulness = 0.0
    
    retriever = get_retriever()
    
    for query in queries:
        question = query["question"]
        expected_sources = query.get("expected_sources", [])
        
        # Get retrieval results
        retrieved_chunks = retriever.retrieve(question, top_k=k)
        retrieved_sources = [c["doc_title"] for c in retrieved_chunks]
        
        # Get answer
        answer_result = ask(question)
        answer = answer_result.get("answer", "")
        answer_label = answer_result.get("answer_label", "insufficient_context")
        
        # Compute metrics
        recall = compute_recall_at_k(retrieved_sources, expected_sources, k)
        precision = compute_precision_at_k(retrieved_sources, expected_sources, k)
        faithfulness = compute_faithfulness(answer, retrieved_chunks)
        
        # Determine status
        if answer_label == "grounded_answer" and recall > 0:
            status = "hit"
        elif answer_label == "grounded_answer" or recall > 0:
            status = "partial_hit"
        else:
            status = "miss"
        
        results.append({
            "question": question,
            "status": status,
            "answer_label": answer_label,
            "answer": answer,
            "retrieved_sources": retrieved_sources,
            "expected_sources": expected_sources,
            "recall": recall,
            "precision": precision,
            "faithfulness": faithfulness,
        })
        
        total_recall += recall
        total_precision += precision
        total_faithfulness += faithfulness
    
    total = len(queries)
    summary = {
        "total_queries": total,
        "recall_at_k": total_recall / total if total > 0 else 0.0,
        "precision_at_k": total_precision / total if total > 0 else 0.0,
        "faithfulness": total_faithfulness / total if total > 0 else 0.0,
    }
    
    return {
        "results": results,
        "summary": summary,
    }
