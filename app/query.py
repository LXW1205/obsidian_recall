"""
query.py — Retrieval + Generation

Hybrid retrieval combining BM25 keyword scoring with Gemini embeddings.
Cross-encoder reranking for improved accuracy.
Citation-strict answers with inline citations and grounding validation.
Controlled answer labels for consistent UI status.
Confidence thresholds to prevent hallucination on bad retrieval.
Metadata filtering by folder and tags.
"""

import math
import os
import re

from dotenv import load_dotenv

from bm25 import BM25Index
from grounding import ALLOWED_ANSWER_LABELS, check_grounding, extract_keywords
from ingest import (
    CHROMA_PATH,
    collection,
    get_embedding,
    get_google_client,
)
from opencode_client import get_llm_client
from rerank import get_reranker, is_available as rerank_available, rerank_results

load_dotenv()

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

CITATION_FORMAT = "[{doc_title} §{chunk_id}]"

SYSTEM_PROMPT = """\
You are a personal knowledge assistant. Answer the user's question \
using ONLY the notes provided below as context. \
If the answer is not in the notes, say "I couldn't find this in your notes." \
Always cite which note(s) your answer came from using inline citations. \
Never make up information not present in the context. \
You may also use the "Additional context" section which contains user preferences and recently accessed notes."""

MODEL_NAME = "gemini-1.5-flash"
TOP_K = 5
RERANK_CANDIDATES = 20  # Get more candidates, rerank, then return top_k
CONFIDENCE_THRESHOLD = 0.3  # Minimum score to consider retrieval confident

# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid retriever combining BM25 and embedding similarity with reranking."""

    def __init__(self, bm25_weight: float = 0.5, embedding_weight: float = 0.5):
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.bm25_index = BM25Index()
        self.chunks: list[dict] = []
        self._built = False

    def build_index(self) -> None:
        """Build the hybrid index from ChromaDB collection."""
        # Get all documents from ChromaDB
        all_docs = collection.get(
            include=["documents", "metadatas", "embeddings"]
        )

        self.chunks = []
        for i, (doc, meta) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
            chunk = {
                "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                "doc_title": meta.get("filename", meta.get("source", "Unknown")),
                "source": meta.get("source", "Unknown"),
                "section": meta.get("section", ""),
                "text": doc,
                "embedding": all_docs["embeddings"][i] if all_docs["embeddings"] else None,
            }
            self.chunks.append(chunk)
            self.bm25_index.add_document(chunk["chunk_id"], doc)

        self.bm25_index.build()
        self._built = True

    def _apply_filters(self, folder: str = None, tags: list[str] = None) -> list[dict]:
        """Filter chunks by folder path and/or tags."""
        filtered = self.chunks

        if folder:
            filtered = [
                c for c in filtered
                if c.get("source", "").startswith(folder)
            ]

        if tags:
            # Filter chunks whose source file contains any of the tags in path
            # (tags are typically in the note content, but we can filter by source path patterns)
            # For now, filter by source path containing tag-like patterns
            tag_filters = [f"#{tag.lower()}" for tag in tags]
            filtered = [
                c for c in filtered
                if any(tf in c.get("text", "").lower() for tf in tag_filters)
            ]

        return filtered

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        folder: str = None,
        tags: list[str] = None,
        use_rerank: bool = True,
    ) -> list[dict]:
        """Retrieve top-k chunks using hybrid scoring with optional reranking."""
        if not self._built:
            self.build_index()

        # Apply metadata filters
        active_chunks = self._apply_filters(folder=folder, tags=tags)

        if not active_chunks:
            return []

        # Build temporary BM25 index for filtered chunks
        temp_bm25 = BM25Index()
        chunk_indices = []
        for idx, chunk in enumerate(active_chunks):
            temp_bm25.add_document(chunk["chunk_id"], chunk["text"])
            chunk_indices.append(idx)
        temp_bm25.build()

        # BM25 scores
        bm25_scores = temp_bm25.score(query)
        bm25_map = {cid: score for cid, score in bm25_scores}

        # Embedding scores
        query_embedding = get_embedding(query)
        embedding_scores = []
        for idx, chunk in enumerate(active_chunks):
            if chunk.get("embedding"):
                sim = self._cosine_similarity(query_embedding, chunk["embedding"])
                embedding_scores.append((idx, sim))
            else:
                embedding_scores.append((idx, 0.0))
        embedding_scores.sort(key=lambda x: x[1], reverse=True)
        embedding_map = {idx: score for idx, score in embedding_scores}

        # Normalize scores to [0, 1] range
        max_bm25 = max((s for _, s in bm25_scores), default=1.0)
        max_embed = max((s for _, s in embedding_scores), default=1.0)

        if max_bm25 == 0:
            max_bm25 = 1.0
        if max_embed == 0:
            max_embed = 1.0

        # Combine scores
        combined_scores = []
        for idx in range(len(active_chunks)):
            bm25_norm = bm25_map.get(idx, 0.0) / max_bm25
            embed_norm = embedding_map.get(idx, 0.0) / max_embed
            combined = (
                self.bm25_weight * bm25_norm +
                self.embedding_weight * embed_norm
            )
            combined_scores.append((idx, combined))

        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top candidates for reranking (or top_k if reranking disabled)
        n_candidates = RERANK_CANDIDATES if use_rerank and rerank_available() else top_k
        candidates = []
        for rank, (idx, score) in enumerate(combined_scores[:n_candidates], start=1):
            chunk = active_chunks[idx]
            candidates.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "doc_title": chunk["doc_title"],
                "source": chunk["source"],
                "section": chunk["section"],
                "score": round(score, 4),
                "text": chunk["text"],
            })

        # Rerank if available and enabled
        if use_rerank and rerank_available() and len(candidates) > top_k:
            candidates = rerank_results(query, candidates, top_k=top_k)
        else:
            # Just return top_k without reranking
            candidates = candidates[:top_k]
            for rank, candidate in enumerate(candidates, start=1):
                candidate["rank"] = rank

        return candidates

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# Global hybrid retriever (lazy-loaded)
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Get or build the global hybrid retriever."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Build a prompt for grounded answer generation with inline citations."""
    context_parts = []
    for chunk in retrieved_chunks:
        citation = CITATION_FORMAT.format(
            doc_title=chunk["doc_title"],
            chunk_id=chunk["chunk_id"],
        )
        section_info = f" (Section: {chunk['section']})" if chunk.get("section") else ""
        context_parts.append(f"{citation}{section_info}\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a personal knowledge assistant answering questions based ONLY on the provided notes context.

QUESTION: {question}

CONTEXT:
{context_block}

INSTRUCTIONS:
1. Answer the question using ONLY the information in the CONTEXT above.
2. If the CONTEXT does not contain enough information to answer, respond with exactly: "I couldn't find this in your notes."
3. If you can answer, provide a concise answer grounded in the context.
4. Every factual statement must include an inline citation in the format: [doc_title §chunk_id]
5. Use the exact doc_title and chunk_id from the CONTEXT.
6. Do not invent facts, speculate, or use outside knowledge.
7. Do not cite chunks that were not provided in the CONTEXT.

ANSWER:"""

    return prompt


def parse_answer_response(
    question: str,
    answer_text: str,
    retrieved_chunks: list[dict],
) -> dict:
    """Parse the LLM response into a structured answer record.

    Returns dict with:
        - answer_label: grounded_answer | insufficient_context | conflicting_context
        - answer: the answer text
        - citations: list of citation strings
        - used_chunk_ids: list of chunk IDs used
        - grounding: grounding validation result
    """
    # Check for insufficient context
    if "couldn't find this in your notes" in answer_text.lower() or "insufficient_context" in answer_text.lower():
        return {
            "answer_label": "insufficient_context",
            "answer": answer_text,
            "citations": [],
            "used_chunk_ids": [],
            "grounding": {"all_citations_valid": True, "grounding_score": 1.0},
        }

    # Extract inline citations from the answer text
    citation_pattern = r'\[([^\]]+)\s§([^\]]+)\]'
    citations = re.findall(citation_pattern, answer_text)

    # Build citation strings and used chunk IDs
    citation_strings = []
    used_chunk_ids = []
    valid_chunk_ids = {chunk["chunk_id"] for chunk in retrieved_chunks}

    for doc_title, chunk_id in citations:
        chunk_id = chunk_id.strip()
        if chunk_id in valid_chunk_ids:
            citation_str = f"[{doc_title.strip()} §{chunk_id}]"
            citation_strings.append(citation_str)
            if chunk_id not in used_chunk_ids:
                used_chunk_ids.append(chunk_id)

    # If no valid citations found, label as insufficient_context
    if not citation_strings:
        return {
            "answer_label": "insufficient_context",
            "answer": answer_text,
            "citations": [],
            "used_chunk_ids": [],
            "grounding": {"all_citations_valid": False, "grounding_score": 0.0},
        }

    # Run grounding validation
    grounding = check_grounding(answer_text, citation_strings, retrieved_chunks)

    # Determine answer label
    if grounding["all_citations_valid"]:
        answer_label = "grounded_answer"
    else:
        # Check if some citations are valid
        valid_count = sum(1 for c in grounding["citation_checks"] if c.get("valid"))
        if valid_count > 0:
            answer_label = "grounded_answer"  # Partially grounded is still grounded
        else:
            answer_label = "insufficient_context"

    return {
        "answer_label": answer_label,
        "answer": answer_text,
        "citations": citation_strings,
        "used_chunk_ids": used_chunk_ids,
        "grounding": grounding,
    }


def ask(
    question: str,
    conversation_history: list[dict] = None,
    hot_memory: dict = None,
    folder: str = None,
    tags: list[str] = None,
) -> dict:
    """
    Ask a question against the vault using hybrid retrieval.

    Args:
        question: The user's question
        conversation_history: Optional list of {role, content} dicts for multi-turn
        hot_memory: Optional Tier 1 memory dict with preferences, active_notes, etc.
        folder: Optional folder path filter (e.g., "Projects/")
        tags: Optional list of tag filters (e.g., ["active", "important"])

    Returns:
        {
            "answer_label": "grounded_answer" | "insufficient_context" | "conflicting_context",
            "answer": "string — the generated answer",
            "citations": ["[doc_title §chunk_id]", ...],
            "sources": ["Note Title 1.md", "Note Title 2.md"],
            "grounding": {...},
            "retrieval_score": float,
        }
    """
    retriever = get_retriever()

    # Step 1: Hybrid retrieval with optional filters
    retrieved_chunks = retriever.retrieve(
        question,
        top_k=TOP_K,
        folder=folder,
        tags=tags,
        use_rerank=True,
    )

    if not retrieved_chunks:
        return {
            "answer_label": "insufficient_context",
            "answer": "I couldn't find any relevant notes in your vault. Try re-indexing or ask a different question.",
            "citations": [],
            "sources": [],
            "grounding": {"all_citations_valid": True, "grounding_score": 1.0},
            "retrieval_score": 0.0,
        }

    # Step 2: Confidence threshold check
    top_score = retrieved_chunks[0].get("score", 0.0)
    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "answer_label": "insufficient_context",
            "answer": f"I found some notes but they don't seem relevant enough (confidence: {top_score:.0%}). Try rephrasing your question or re-indexing.",
            "citations": [],
            "sources": [c["doc_title"] for c in retrieved_chunks],
            "grounding": {"all_citations_valid": True, "grounding_score": 1.0},
            "retrieval_score": top_score,
        }

    # Step 3: Build prompt with hot memory injection (Tier 1)
    hot_memory_prefix = ""
    if hot_memory:
        hot_context_parts = []
        prefs = hot_memory.get("preferences", "")
        if prefs:
            hot_context_parts.append(f"User preferences:\n{prefs}")
        active = hot_memory.get("active_notes", [])
        if active:
            hot_context_parts.append(f"Recently accessed notes: {', '.join(active)}")
        if hot_context_parts:
            hot_memory_prefix = "Additional context:\n" + "\n\n".join(hot_context_parts) + "\n\n"

    # Build conversation history
    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append(msg)

    # Add the current question with context and hot memory
    context_block = build_prompt(question, retrieved_chunks)
    user_content = f"{hot_memory_prefix}{context_block}"
    messages.append({"role": "user", "content": user_content})

    # Step 4: Generate answer
    llm = get_llm_client()
    if llm is not None:
        # Opencode provider
        content_parts = []
        if hot_memory_prefix:
            content_parts.append(hot_memory_prefix)
        content_parts.append(context_block)
        combined_prompt = "\n\n".join(content_parts)

        response_text = llm.generate_content(
            prompt=combined_prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
        )
    else:
        # Gemini provider
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        response = get_google_client().models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.1},
        )
        response_text = response.text

    # Step 5: Parse and validate answer
    result = parse_answer_response(question, response_text, retrieved_chunks)

    # Extract unique sources
    sources = list(dict.fromkeys(chunk["doc_title"] for chunk in retrieved_chunks))

    return {
        "answer_label": result["answer_label"],
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": sources,
        "grounding": result["grounding"],
        "retrieval_score": top_score,
    }
