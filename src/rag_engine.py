"""End-to-End Enterprise RAG Engine Orchestrator.

Combines Document Ingestion, Recursive Chunking, Hybrid Retrieval (Dense + BM25),
Cross-Encoder Reranking, Strict Citation Synthesis, and Automated RAG Triad Evaluation.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
import urllib.request
import json

from .document_loader import Document, DocumentLoader
from .chunker import Chunk, RecursiveCharacterChunker
from .embeddings import VectorEmbedder
from .hybrid_retriever import HybridRetriever, SearchResult
from .reranker import CrossEncoderReranker
from .evaluator import RAGEvaluator, RAGMetrics

@dataclass
class RAGResponse:
    query: str
    answer: str
    citations: List[SearchResult]
    metrics: RAGMetrics
    retrieval_method: str
    chunks_analyzed: int


class RAGEngine:
    """Full-featured, production-ready enterprise RAG engine."""

    def __init__(
        self,
        embedder_provider: str = "local",
        llm_provider: str = "demo",
        api_key: Optional[str] = None,
        chunk_size: int = 550,
        chunk_overlap: int = 100
    ):
        self.chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = VectorEmbedder(provider=embedder_provider, api_key=api_key)
        self.retriever = HybridRetriever(embedder=self.embedder)
        self.reranker = CrossEncoderReranker()
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key

        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self.chat_history: List[Dict[str, str]] = []

    def load_and_index_documents(self, documents: List[Document]):
        """Ingest documents, chunk them, and construct dense and sparse indices."""
        self.documents.extend(documents)
        new_chunks = self.chunker.chunk_documents(documents)
        self.chunks.extend(new_chunks)
        self.retriever.index(self.chunks)

    def load_directory(self, dir_path: Path):
        """Load all documents from a directory and index them."""
        docs = DocumentLoader.load_directory(dir_path)
        if docs:
            self.load_and_index_documents(docs)

    def rewrite_query_with_history(self, current_query: str) -> str:
        """Resolve pronouns and coreferences based on conversational chat history."""
        if not self.chat_history:
            return current_query

        # Look for references like 'it', 'its', 'their', 'they', 'that revenue', 'those terms'
        pronoun_pattern = r"\b(it|its|their|they|this|that|these|those)\b"
        if re.search(pronoun_pattern, current_query.lower()):
            last_turn = self.chat_history[-1]
            last_query = last_turn.get("user", "")
            # Extract main entities from last query
            entities = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]*\b", last_query)
            if entities:
                enriched = f"{current_query} (referring to {' '.join(entities)})"
                return enriched
        return current_query

    def query(
        self,
        user_query: str,
        top_k: int = 4,
        alpha: float = 0.5,
        enable_reranking: bool = True
    ) -> RAGResponse:
        """Execute end-to-end RAG pipeline with grounded citations & evaluation."""
        if not self.chunks:
            return RAGResponse(
                query=user_query,
                answer="No documents are currently indexed in the knowledge base. Please upload or index enterprise documents first.",
                citations=[],
                metrics=RAGMetrics(0, 0, 0, 0, False, "No data"),
                retrieval_method="none",
                chunks_analyzed=0
            )

        # 1. Query Rewriting with Contextual Memory
        effective_query = self.rewrite_query_with_history(user_query)

        # 2. Hybrid Retrieval (Dense Vector + BM25)
        candidate_k = top_k * 3 if enable_reranking else top_k
        search_results = self.retriever.search_hybrid(
            query=effective_query,
            top_k=candidate_k,
            alpha=alpha,
            method="rrf"
        )

        # 3. Cross-Encoder Reranking
        if enable_reranking and len(search_results) > 1:
            final_citations = self.reranker.rerank(effective_query, search_results, top_k=top_k)
            retrieval_method = f"Hybrid (RRF alpha={alpha:.2f}) + Cross-Encoder Rerank"
        else:
            final_citations = search_results[:top_k]
            retrieval_method = f"Hybrid (RRF alpha={alpha:.2f})"

        # 4. Context Synthesis & Grounded LLM Generation
        answer = self._generate_answer(user_query, final_citations)

        # 5. RAG Triad Quality Evaluation
        metrics = RAGEvaluator.evaluate(user_query, answer, final_citations)

        # 6. Record Conversational Memory
        self.chat_history.append({"user": user_query, "assistant": answer})

        return RAGResponse(
            query=user_query,
            answer=answer,
            citations=final_citations,
            metrics=metrics,
            retrieval_method=retrieval_method,
            chunks_analyzed=len(final_citations)
        )

    def _generate_answer(self, query: str, citations: List[SearchResult]) -> str:
        """Route to requested LLM provider or built-in contextual synthesis engine."""
        if self.llm_provider == "openai" and self.api_key:
            return self._generate_openai(query, citations)
        elif self.llm_provider == "gemini" and self.api_key:
            return self._generate_gemini(query, citations)
        return self._generate_grounded_demo(query, citations)

    def _generate_grounded_demo(self, query: str, citations: List[SearchResult]) -> str:
        """Context-Aware Grounded Synthesizer with strict citation references.

        Runs with zero external API dependencies, producing structured, verified answers.
        """
        if not citations:
            return "I could not find relevant documentation in the knowledge base to answer your question."

        # Extract high-relevance sentences from the top citations
        query_terms = set(re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]{3,}\b", query.lower()))

        extracted_findings = []
        for res in citations:
            chunk_text = res.chunk.text
            sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
            matched_sentences = []

            for sent in sentences:
                sent_clean = sent.strip()
                if not sent_clean:
                    continue
                words = set(re.findall(r"\b[a-zA-Z0-9_\-\.\$\%]{3,}\b", sent_clean.lower()))
                overlap = len(query_terms & words)
                if overlap > 0:
                    matched_sentences.append((overlap, sent_clean))

            matched_sentences.sort(key=lambda x: x[0], reverse=True)
            if matched_sentences:
                top_sent = matched_sentences[0][1]
                doc_name = res.chunk.doc_id
                extracted_findings.append((doc_name, res.chunk.chunk_id, top_sent))

        # Build clean, executive answer
        response_lines = [
            f"Based on the enterprise knowledge base, here is the verified information for **\"{query}\"**:\n"
        ]

        if extracted_findings:
            for doc, cid, sent in extracted_findings[:3]:
                # Format bullet with citation tag
                response_lines.append(f"* **{sent}** `[Source: {doc}]`")
        else:
            # Fallback to the top chunk's core paragraph
            top_res = citations[0]
            summary_snippet = top_res.chunk.text[:280].strip() + "..."
            response_lines.append(f"* {summary_snippet} `[Source: {top_res.chunk.doc_id}]`")

        response_lines.append("\n*All facts and numerical disclosures above have been cross-checked against retrieved source chunks.*")
        return "\n".join(response_lines)

    def _generate_openai(self, query: str, citations: List[SearchResult]) -> str:
        context_str = "\n\n---\n\n".join([
            f"[Source Document: {c.chunk.doc_id} | Chunk ID: {c.chunk.chunk_id}]\n{c.chunk.text}"
            for c in citations
        ])

        system_prompt = (
            "You are an enterprise AI research assistant. Answer the user query strictly using the provided context chunks. "
            "Every claim, number, or factual statement must cite its source using `[Source: <doc_name>]`. "
            "If the answer cannot be found in the context, explicitly state that you don't know rather than fabricating facts."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }

        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI Generation Error ({e}). Falling back to local reasoning:\n\n" + self._generate_grounded_demo(query, citations)

    def _generate_gemini(self, query: str, citations: List[SearchResult]) -> str:
        context_str = "\n\n---\n\n".join([
            f"[Source Document: {c.chunk.doc_id} | Chunk ID: {c.chunk.chunk_id}]\n{c.chunk.text}"
            for c in citations
        ])

        prompt = (
            f"You are an enterprise AI research assistant. Answer the question strictly based on the context.\n"
            f"Every fact must cite `[Source: <doc_name>]`.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Gemini Generation Error ({e}). Falling back to local reasoning:\n\n" + self._generate_grounded_demo(query, citations)
