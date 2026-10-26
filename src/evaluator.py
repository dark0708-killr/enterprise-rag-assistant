"""RAG Triad Evaluation & Hallucination Guardrails Module.

Implements automated evaluation metrics inspired by Ragas and TruLens:
1. Faithfulness (Groundedness / Hallucination Detection).
2. Answer Relevance (Direct question-answering alignment).
3. Context Precision (Information density of retrieved chunks).
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import re
from .hybrid_retriever import SearchResult

@dataclass
class RAGMetrics:
    faithfulness: float        # 0.0 - 1.0 (Groundedness)
    answer_relevance: float    # 0.0 - 1.0 (Direct question answering)
    context_precision: float   # 0.0 - 1.0 (Signal to noise of chunks)
    overall_score: float       # Harmonic mean (F1 style)
    hallucination_detected: bool
    verdict: str


class RAGEvaluator:
    """Evaluates the generation and retrieval quality of a RAG pipeline turn."""

    @classmethod
    def evaluate(
        cls,
        query: str,
        response: str,
        retrieved_contexts: List[SearchResult]
    ) -> RAGMetrics:
        if not retrieved_contexts:
            return RAGMetrics(
                faithfulness=0.0,
                answer_relevance=0.0,
                context_precision=0.0,
                overall_score=0.0,
                hallucination_detected=True,
                verdict="No context retrieved (High Hallucination Risk)"
            )

        context_text = " ".join([r.chunk.text for r in retrieved_contexts]).lower()
        response_lower = response.lower()
        query_lower = query.lower()

        # 1. Faithfulness (Groundedness): Check if facts / numbers / entities in response appear in context
        response_claims = re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]{3,}\b", response_lower)
        context_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]{3,}\b", context_text))

        # Check critical numerical & currency claims specifically
        numbers_in_response = set(re.findall(r"(\$[\d\.]+[MBKmbk]?|\d+\.?\d*\%|\b\d{2,}\b)", response))
        numbers_in_context = set(re.findall(r"(\$[\d\.]+[MBKmbk]?|\d+\.?\d*\%|\b\d{2,}\b)", " ".join([r.chunk.text for r in retrieved_contexts])))

        grounded_claims = sum(1 for c in response_claims if c in context_words)
        total_claims = max(len(response_claims), 1)
        lexical_grounding = grounded_claims / total_claims

        # If numerical claims exist, they must be strictly present in context
        num_grounding = 1.0
        if numbers_in_response:
            matched_numbers = sum(1 for n in numbers_in_response if n in numbers_in_context)
            num_grounding = matched_numbers / len(numbers_in_response)

        faithfulness = round(0.4 * lexical_grounding + 0.6 * num_grounding, 3)
        faithfulness = min(max(faithfulness, 0.0), 1.0)

        # 2. Answer Relevance: Check semantic alignment between response and query
        query_terms = set(re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]{3,}\b", query_lower))
        relevance_matches = sum(1 for q in query_terms if q in response_lower)
        answer_relevance = round(min(relevance_matches / max(len(query_terms), 1), 1.0), 3)

        # 3. Context Precision: How many of the retrieved chunks actually helped answer the query?
        relevant_chunks = 0
        for res in retrieved_contexts:
            chunk_terms = set(re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]{3,}\b", res.chunk.text.lower()))
            overlap = len(query_terms & chunk_terms)
            if overlap >= 1 or res.hybrid_score > 0.15:
                relevant_chunks += 1
        context_precision = round(relevant_chunks / len(retrieved_contexts), 3)

        # Harmonic Mean Overall Score
        denom = (faithfulness + answer_relevance + context_precision)
        overall_score = round(
            (3 * faithfulness * answer_relevance * context_precision) / (denom + 1e-6),
            3
        )

        hallucination_detected = faithfulness < 0.65

        if overall_score >= 0.80:
            verdict = "Excellent (Highly Grounded & Accurate)"
        elif overall_score >= 0.60:
            verdict = "Good (Grounded with minor semantic divergence)"
        elif overall_score >= 0.40:
            verdict = "Moderate (Review citations carefully)"
        else:
            verdict = "Low Quality (Potential Hallucination)"

        return RAGMetrics(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            overall_score=overall_score,
            hallucination_detected=hallucination_detected,
            verdict=verdict
        )
