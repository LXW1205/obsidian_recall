"""
bm25.py — Pure Python BM25 keyword scoring index.

Provides keyword-based retrieval that complements semantic embeddings.
Combined with cosine similarity for hybrid retrieval.
"""

import math
import re
from collections import Counter


class BM25Index:
    """BM25 keyword-based retrieval index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.doc_freqs: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self._built = False

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        tokens = self._tokenize(text)
        self.documents.append(text)
        self.doc_ids.append(doc_id)
        self.doc_lengths.append(len(tokens))

        for token in set(tokens):
            self.doc_freqs[token] += 1

    def build(self) -> None:
        """Finalise the index, computing IDF values."""
        n_docs = len(self.documents)
        if n_docs == 0:
            return

        self.avg_doc_length = sum(self.doc_lengths) / n_docs

        for term, freq in self.doc_freqs.items():
            # IDF with smoothing
            self.idf[term] = math.log(
                (n_docs - freq + 0.5) / (freq + 0.5) + 1.0
            )

        self._built = True

    def score(self, query: str) -> list[tuple[int, float]]:
        """Score all documents against a query.

        Returns list of (doc_index, score) sorted by score descending.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        query_tokens = self._tokenize(query)
        scores = []

        for doc_idx, doc_text in enumerate(self.documents):
            doc_tokens = self._tokenize(doc_text)
            doc_len = self.doc_lengths[doc_idx]
            doc_counter = Counter(doc_tokens)

            score = 0.0
            for token in query_tokens:
                if token not in self.idf:
                    continue

                tf = doc_counter.get(token, 0)
                idf = self.idf[token]

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avg_doc_length)
                )
                score += idf * (numerator / denominator)

            scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_doc_id(self, idx: int) -> str:
        """Get the document ID for a given index."""
        return self.doc_ids[idx]

    @property
    def size(self) -> int:
        return len(self.documents)
