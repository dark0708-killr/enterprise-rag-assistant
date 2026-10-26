"""Dense Vector Embedding Engine.

Provides multi-backend semantic embeddings:
1. Built-in Local Semantic Projection (Zero API key required, fast & portable).
2. OpenAI `text-embedding-3-small` (When API key provided).
3. Google Gemini `text-embedding-004` (When API key provided).
"""
from typing import List, Dict, Any, Optional
import numpy as np
import re
import math

class VectorEmbedder:
    """Multi-backend embedding provider with cosine similarity search."""

    def __init__(
        self,
        provider: str = "local",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.dim = 384

    def fit_local_vocab(self, texts: List[str]):
        """Build vocabulary and inverse document frequencies for local dense embeddings."""
        doc_count = len(texts)
        df: Dict[str, int] = {}

        for text in texts:
            words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower()))
            for w in words:
                df[w] = df.get(w, 0) + 1

        # Keep most informative vocabulary terms
        sorted_vocab = sorted(df.items(), key=lambda x: x[1], reverse=True)[:4000]
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}

        # Sublinear smoothed IDF vector
        idf_list = [
            math.log((doc_count + 1.0) / (df.get(w, 0) + 1.0)) + 1.0
            for w, _ in sorted_vocab
        ]
        self.idf = np.array(idf_list, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalized 1D vector."""
        if self.provider == "openai" and self.api_key:
            return self._embed_openai([text])[0]
        elif self.provider == "gemini" and self.api_key:
            return self._embed_gemini([text])[0]
        return self._embed_local(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts into a 2D matrix of shape (n_texts, dim)."""
        if self.provider == "openai" and self.api_key:
            return self._embed_openai(texts)
        elif self.provider == "gemini" and self.api_key:
            return self._embed_gemini(texts)

        return np.vstack([self._embed_local(t) for t in texts])

    def _embed_local(self, text: str) -> np.ndarray:
        """Deterministic, sublinear term-frequency semantic projection with L2 normalization."""
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())

        if not words:
            return np.ones(max(len(self.vocab), 64), dtype=np.float32) / math.sqrt(max(len(self.vocab), 64))

        for w in words:
            idx = self.vocab.get(w)
            if idx is not None:
                vec[idx] += 1.0

        # Apply sublinear log scaling 1 + log(tf)
        nz = vec > 0
        vec[nz] = (1.0 + np.log(vec[nz])) * self.idf[nz]

        # Fast pseudo-random projection to fixed dense dimensionality (dim=384)
        if len(self.vocab) > self.dim:
            # Deterministic folding
            dense_vec = np.zeros(self.dim, dtype=np.float32)
            for i, val in enumerate(vec):
                dense_vec[i % self.dim] += val
        else:
            dense_vec = vec

        # L2 unit normalization
        norm = np.linalg.norm(dense_vec)
        if norm > 1e-6:
            dense_vec = dense_vec / norm
        return dense_vec

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        import urllib.request
        import json

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name or "text-embedding-3-small",
            "input": texts,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        embeddings = [item["embedding"] for item in data["data"]]
        arr = np.array(embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        return arr / norms

    def _embed_gemini(self, texts: List[str]) -> np.ndarray:
        import urllib.request
        import json

        model = self.model_name or "text-embedding-004"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={self.api_key}"
        requests = [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in texts]
        payload = {"requests": requests}

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        embeddings = [item["values"] for item in data["embeddings"]]
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        return arr / norms

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between 1D query vector and 2D document matrix."""
        # Since vectors are unit-normalized, dot product equals cosine similarity
        return np.dot(doc_matrix, query_vec)
