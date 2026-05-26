"""
grounding.py — Citation grounding validation.

Verifies that each citation in an answer:
1. Exists in the retrieved chunks
2. The cited chunk text contains supporting wording for the answer

Uses heuristic phrase overlap and keyword matching.
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ALLOWED_ANSWER_LABELS = {"grounded_answer", "insufficient_context", "conflicting_context"}
CITATION_FORMAT = "[{doc_title} §{chunk_id}]"

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "need", "must", "to", "of", "in", "for", "on", "with", "at",
    "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "and", "but", "or", "nor", "not",
    "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "such",
    "no", "only", "own", "same", "than", "too", "very", "just",
    "it", "its", "this", "that", "these", "those", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "they", "them", "their", "what", "which", "who",
    "whom", "how", "when", "where", "why",
}


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text."""
    words = re.findall(r'[a-z]+', text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


# ---------------------------------------------------------------------------
# Grounding validation
# ---------------------------------------------------------------------------

def check_grounding(
    answer_text: str,
    citations: list[str],
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Check grounding of an answer against retrieved chunks.

    Returns a dict with:
        - all_citations_valid: bool
        - citation_checks: list of per-citation results
        - grounding_score: float (0.0 to 1.0)
    """
    # Build lookup: chunk_id -> chunk
    chunk_map = {chunk.get("chunk_id", ""): chunk for chunk in retrieved_chunks}

    citation_checks = []
    all_valid = True
    total_score = 0.0

    for citation in citations:
        # Parse chunk_id from citation [doc_title §chunk_id]
        match = re.match(r'\[.*§(.+?)\]', citation)
        if not match:
            citation_checks.append({
                "citation": citation,
                "chunk_exists": False,
                "text_supports": False,
                "valid": False,
                "explanation": "Could not parse chunk_id from citation",
            })
            all_valid = False
            continue

        chunk_id = match.group(1).strip()

        # Check 1: Does the chunk exist in retrieval?
        chunk_exists = chunk_id in chunk_map

        if not chunk_exists:
            citation_checks.append({
                "citation": citation,
                "chunk_exists": False,
                "text_supports": False,
                "valid": False,
                "explanation": f"Chunk '{chunk_id}' not found in retrieval results",
            })
            all_valid = False
            continue

        # Check 2: Does the chunk text support the answer?
        chunk = chunk_map[chunk_id]
        chunk_text = chunk.get("text", chunk.get("chunk_text", ""))

        # Extract keywords from answer (excluding citation parts)
        answer_clean = re.sub(r'\[.*?\]', '', answer_text)
        answer_keywords = extract_keywords(answer_clean)
        chunk_keywords = extract_keywords(chunk_text)

        # Check keyword overlap
        if answer_keywords:
            overlap = answer_keywords & chunk_keywords
            overlap_ratio = len(overlap) / len(answer_keywords)
        else:
            overlap_ratio = 1.0

        # Check if key phrases from answer appear in chunk
        answer_phrases = re.findall(r'\b\w+(?:\s+\w+){2,}\b', answer_clean)
        phrase_support = 0
        phrase_total = 0

        for phrase in answer_phrases:
            phrase_total += 1
            if phrase.lower() in chunk_text.lower():
                phrase_support += 1

        phrase_ratio = phrase_support / phrase_total if phrase_total > 0 else 1.0

        # Determine if text supports answer
        text_supports = (
            overlap_ratio >= 0.3 or
            phrase_ratio >= 0.5 or
            len(answer_keywords & chunk_keywords) >= 2
        )

        if not text_supports:
            all_valid = False

        citation_score = max(overlap_ratio, phrase_ratio)
        total_score += citation_score

        citation_checks.append({
            "citation": citation,
            "chunk_exists": True,
            "text_supports": text_supports,
            "keyword_overlap_ratio": round(overlap_ratio, 3),
            "phrase_support_ratio": round(phrase_ratio, 3),
            "valid": text_supports,
            "explanation": (
                f"Keyword overlap: {overlap_ratio:.1%}, "
                f"Phrase support: {phrase_ratio:.1%}"
            ),
        })

    grounding_score = total_score / len(citations) if citations else 0.0

    return {
        "all_citations_valid": all_valid,
        "citation_checks": citation_checks,
        "grounding_score": round(grounding_score, 3),
    }
