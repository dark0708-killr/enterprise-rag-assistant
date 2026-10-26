"""Cross-Encoder Semantic Reranker Module.

Second-stage retrieval refinement: evaluates joint (query, chunk) interaction
to eliminate false positives and maximize context density for generation.
"""
from typing import List, Tuple
import re
import math
from .hybrid_retriever import SearchResult

class CrossEncoderReranker:
    """Joint query-context semantic relevance evaluator."""

    def __init__(self, diversity_penalty: float = 0.15):
        self.diversity_penalty = diversity_penalty

    def rerank(self, query: str, search_results: List[SearchResult], top_k: int = 4) -> List[SearchResult]:
        """Rerank candidate search results using joint lexical-semantic interaction scoring."""
        if not search_results:
            return []

        query_terms = set(re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]+\b", query.lower()))
        scored_results: List[Tuple[float, SearchResult]] = []

        seen_docs = set()

        for res in search_results:
            text = res.chunk.text
            text_lower = text.lower()
            text_terms = re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]+\b", text_lower)
            text_term_set = set(text_terms)

            # 1. Exact query keyword coverage
            coverage = len(query_terms & text_term_set) / max(len(query_terms), 1)

            # 2. Term proximity / phrase bonus
            proximity_bonus = 0.0
            if query.lower() in text_lower:
                proximity_bonus = 0.35

            # 3. High information-density bonus (numbers, metrics, tables, currency)
            metrics_match = len(re.findall(r"(\$[\d\.]+[MBKmbk]?|\d+\.?\d*\%|\b\d{2,}\b)", text))
            info_density = min(metrics_match * 0.05, 0.25)

            # 4. Diversity penalty for excessive chunks from identical document
            doc_id = res.chunk.doc_id
            div_pen = self.diversity_penalty if doc_id in seen_docs else 0.0
            seen_docs.add(doc_id)

            # Composite rerank score
            rerank_score = (
                0.40 * res.hybrid_score +
                0.35 * coverage +
                0.15 * proximity_bonus +
                0.10 * info_density -
                div_pen
            )

            # Update score on result
            res.hybrid_score = round(max(float(rerank_score), 0.01), 4)
            scored_results.append((rerank_score, res))

        # Sort descending by rerank score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        final_results = []
        for new_rank, (_, res) in enumerate(scored_results[:top_k], start=1):
            res.rank = new_rank
            final_results.append(res)

        return final_results
