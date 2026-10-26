"""Advanced Chunking Strategy Module.

Demonstrates production Recursive Character Chunking with overlap
and semantic sentence boundary preservation.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import re
from .document_loader import Document

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    char_start: int = 0
    char_end: int = 0
    word_count: int = 0

    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split())


class RecursiveCharacterChunker:
    """Recursively splits text on natural boundaries (paragraphs, lines, sentences, words)."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

    def split_text(self, text: str) -> List[str]:
        """Split text recursively into chunks under chunk_size."""
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits = []
        for s in splits:
            if s.strip():
                if len(s) < self.chunk_size:
                    good_splits.append(s)
                else:
                    if new_separators:
                        sub_chunks = self._split(s, new_separators)
                        good_splits.extend(sub_chunks)
                    else:
                        good_splits.append(s)

        # Merge splits with overlap
        return self._merge_splits(good_splits, separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total_len = 0

        for d in splits:
            d_len = len(d)
            sep_len = len(separator) if current_doc else 0

            if total_len + d_len + sep_len > self.chunk_size:
                if total_len > 0:
                    joined = separator.join(current_doc).strip()
                    if joined:
                        docs.append(joined)

                    # Keep overlap from the end of current_doc
                    while total_len > self.chunk_overlap and current_doc:
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + len(separator)

                current_doc.append(d)
                total_len = sum(len(x) + len(separator) for x in current_doc)
            else:
                current_doc.append(d)
                total_len += d_len + sep_len

        if current_doc:
            joined = separator.join(current_doc).strip()
            if joined:
                docs.append(joined)

        return docs

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Convert a Document into an indexed sequence of Chunks with metadata."""
        raw_chunks = self.split_text(document.content)
        chunks = []
        cursor = 0

        for idx, text in enumerate(raw_chunks):
            start = document.content.find(text[:30], cursor) if len(text) >= 30 else cursor
            if start == -1:
                start = cursor
            end = start + len(text)
            cursor = max(cursor, start + 1)

            chunk_meta = dict(document.metadata)
            chunk_meta["chunk_index"] = idx
            chunk_meta["total_chunks_in_doc"] = len(raw_chunks)

            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#chunk_{idx}",
                    doc_id=document.doc_id,
                    text=text,
                    metadata=chunk_meta,
                    char_start=start,
                    char_end=end,
                    word_count=len(text.split()),
                )
            )

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
