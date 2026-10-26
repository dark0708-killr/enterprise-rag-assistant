"""Document Ingestion & Multi-Format Loader Pipeline.

Handles extraction of clean text and metadata across Markdown, TXT, PDF, CSV, and JSON.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import csv
import io

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id and "filename" in self.metadata:
            self.doc_id = self.metadata["filename"]


class DocumentLoader:
    """Robust document parser supporting enterprise file types."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".csv", ".json"}

    @classmethod
    def load_file(cls, file_path: Path) -> Optional[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        ext = file_path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            return None

        metadata = {
            "filename": file_path.name,
            "filepath": str(file_path),
            "file_type": ext[1:],
            "size_bytes": file_path.stat().st_size,
        }

        try:
            if ext in {".md", ".txt"}:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                return Document(content=text, metadata=metadata, doc_id=file_path.name)

            elif ext == ".pdf":
                text = cls._extract_pdf_text(file_path.read_bytes())
                return Document(content=text, metadata=metadata, doc_id=file_path.name)

            elif ext == ".json":
                data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
                text = json.dumps(data, indent=2)
                return Document(content=text, metadata=metadata, doc_id=file_path.name)

            elif ext == ".csv":
                text_lines = []
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        text_lines.append(" | ".join(row))
                text = "\n".join(text_lines)
                return Document(content=text, metadata=metadata, doc_id=file_path.name)

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None

        return None

    @classmethod
    def load_bytes(cls, filename: str, file_bytes: bytes) -> Optional[Document]:
        ext = Path(filename).suffix.lower()
        metadata = {
            "filename": filename,
            "file_type": ext[1:] if ext else "unknown",
            "size_bytes": len(file_bytes),
        }

        try:
            if ext in {".md", ".txt", ".json", ".csv"}:
                text = file_bytes.decode("utf-8", errors="replace")
                return Document(content=text, metadata=metadata, doc_id=filename)

            elif ext == ".pdf":
                text = cls._extract_pdf_text(file_bytes)
                return Document(content=text, metadata=metadata, doc_id=filename)

        except Exception as e:
            print(f"Error reading byte stream for {filename}: {e}")
            return None

        return None

    @staticmethod
    def _extract_pdf_text(file_bytes: bytes) -> str:
        """Extract text from PDF using PyPDF/PyPDF2 or fallback to basic string stream."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)
            except ImportError:
                # Basic stream decode fallback
                text = file_bytes.decode("latin1", errors="ignore")
                return f"[PDF raw content - install pypdf for layout parsing]\n\n{text[:5000]}"

    @classmethod
    def load_directory(cls, dir_path: Path) -> List[Document]:
        docs = []
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            return docs

        for f in sorted(dir_path.glob("*.*")):
            if f.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                doc = cls.load_file(f)
                if doc:
                    docs.append(doc)
        return docs
