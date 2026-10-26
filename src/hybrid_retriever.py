"""Hybrid Retriever Engine - Dense Semantic & Sparse BM25 Fusion.

Implements Reciprocal Rank Fusion (RRF) and Convex Linear Combination
to combine high-dimensional embeddings with lexical BM25 matching.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from .chunker import Chunk
from .bm25 import BM25Okapi
from .embeddings import VectorEmbedder

@dataclass
class SearchResult:
    chunk: Chunk
    hybrid_score: float
    dense_score: float
    sparse_score: float
    rank: int = 1


class HybridRetriever:
    """State-of-the-art Hybrid Search Engine with Reciprocal Rank Fusion."""

    def __init__(self, embedder: Optional[VectorEmbedder] = None):
        self.embedder = embedder or VectorEmbedder(provider="local")
        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        self.dense_matrix: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

    def index(self, chunks: List[Chunk]):
        """Index a collection of chunks into both dense vector and sparse BM25 indices."""
        self.chunks = chunks
        self.chunk_texts = [c.text for c in chunks]

        if not chunks:
            self.dense_matrix = None
            self.bm25 = None
            return

        # 1. Build BM25 index
        tokenized_corpus = [BM25Okapi.tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build Dense Vector Index
        if self.embedder.provider == "local":
            self.embedder.fit_local_vocab(self.chunk_texts)
        self.dense_matrix = self.embedder.embed_batch(self.chunk_texts)

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Retrieve top-K chunks using cosine similarity."""
        if self.dense_matrix is None or len(self.chunks) == 0:
            return []

        q_vec = self.embedder.embed_text(query)
        sims = VectorEmbedder.cosine_similarity(q_vec, self.dense_matrix)
        indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in indices]

    def search_sparse(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Retrieve top-K chunks using BM25Okapi."""
        if self.bm25 is None or len(self.chunks) == 0:
            return []
        return self.bm25.search(query, top_k=top_k)

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        method: str = "rrf",
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """Hybrid search combining Dense and Sparse scores.

        Args:
            query: The search query string.
            top_k: Number of final fused results to return.
            alpha: Weight for dense search (0.0 = pure BM25, 1.0 = pure Dense).
            method: 'rrf' (Reciprocal Rank Fusion) or 'linear' (Convex Combination).
            rrf_k: Smoothing constant for RRF (standard is 60).
        """
        if len(self.chunks) == 0:
            return []

        # Retrieve candidate pools
        dense_candidates = self.search_dense(query, top_k=min(len(self.chunks), 25))
        sparse_candidates = self.search_sparse(query, top_k=min(len(self.chunks), 25))

        dense_scores_map = {idx: score for idx, score in dense_candidates}
        sparse_scores_map = {idx: score for idx, score in sparse_candidates}

        fused_scores: Dict[int, float] = {}

        if method == "rrf":
            # Reciprocal Rank Fusion: 1 / (k + rank)
            for rank, (idx, _) in enumerate(dense_candidates, start=1):
                fused_scores[idx] = fused_scores.get(idx, 0.0) + (alpha * (1.0 / (rrf_k + rank)))

            for rank, (idx, _) in enumerate(sparse_candidates, start=1):
                fused_scores[idx] = fused_scores.get(idx, 0.0) + ((1.0 - alpha) * (1.0 / (rrf_k + rank)))

        else:
            # Linear Convex Combination with Min-Max Score Normalization
            all_indices = set(dense_scores_map.keys()) | set(sparse_scores_map.keys())

            d_vals = list(dense_scores_map.values()) or [0.0]
            s_vals = list(sparse_scores_map.values()) or [0.0]

            min_d, max_d = min(d_vals), max(d_vals)
            min_s, max_s = min(s_vals), max(s_vals)

            for idx in all_indices:
                raw_d = dense_scores_map.get(idx, 0.0)
                raw_s = sparse_scores_map.get(idx, 0.0)

                norm_d = (raw_d - min_d) / (max_d - min_d + 1e-8) if max_d > min_d else raw_d
                norm_s = (raw_s - min_s) / (max_s - min_s + 1e-8) if max_s > min_s else raw_s

                fused_scores[idx] = alpha * norm_d + (1.0 - alpha) * norm_s

        # Rank and construct SearchResult objects
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (idx, f_score) in enumerate(sorted_results, start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    hybrid_score=round(float(f_score), 4),
                    dense_score=round(float(dense_scores_map.get(idx, 0.0)), 4),
                    sparse_score=round(float(sparse_scores_map.get(idx, 0.0)), 4),
                    rank=rank
                )
            )

        return results
