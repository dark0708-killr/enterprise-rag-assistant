"""Sparse Lexical Search Engine - BM25Okapi Implementation.

Implements the Robertson-Spärck Jones BM25 probabilistic ranking algorithm from first principles,
providing exact keyword, entity, acronym, and identifier retrieval for Hybrid RAG.
"""
import math
import re
from typing import List, Dict, Tuple
from collections import Counter

class BM25Okapi:
    """Okapi BM25 ranking algorithm with Robertson IDF smoothing."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lens) / max(self.corpus_size, 1)

        # Document frequencies for all terms
        self.doc_freqs: Dict[str, int] = {}
        # Term frequencies for each document
        self.term_freqs: List[Counter] = []

        for doc in corpus:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Calculate IDF for all vocabulary
        self.idf: Dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            # Robertson BM25 IDF with smoothing
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Normalize and tokenize text into lowercase alphanumeric words."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]+\b", text.lower())

    def get_scores(self, query: List[str]) -> List[float]:
        """Compute BM25 score for query against all documents in corpus."""
        scores = [0.0] * self.corpus_size

        for term in query:
            if term not in self.idf:
                continue

            idf = self.idf[term]
            for doc_idx, tf_dict in enumerate(self.term_freqs):
                freq = tf_dict.get(term, 0)
                if freq == 0:
                    continue

                doc_len = self.doc_lens[doc_idx]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf * (numerator / denominator)

        return scores

    def search(self, query_text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search top-K document indices with their BM25 scores."""
        query_tokens = self.tokenize(query_text)
        if not query_tokens:
            return []

        scores = self.get_scores(query_tokens)
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in scored_indices[:top_k] if score > 0]
