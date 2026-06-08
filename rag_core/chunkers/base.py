"""
rag_core.chunkers.base
~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all document chunkers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..types import Chunk


class BaseChunker(ABC):
    """Interface that every chunker must implement.

    A chunker takes raw text (or a file) and splits it into smaller,
    semantically meaningful pieces suitable for embedding.
    """

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: The raw text to chunk.
            metadata: Optional metadata to attach to every chunk (e.g., source file name).

        Returns:
            A list of ``Chunk`` objects.
        """

    def chunk_file(self, file_path: str) -> List[Chunk]:
        """Read a file and chunk its contents.

        Supports ``.pdf``, ``.txt``, and ``.md`` by default.
        Subclasses may override to support additional formats.

        Args:
            file_path: Path to the file.

        Returns:
            A list of ``Chunk`` objects with source metadata.
        """
        import os

        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        if ext == ".pdf":
            text = self._read_pdf(file_path)
        elif ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read()
        else:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported: .pdf, .txt, .md"
            )

        return self.chunk(text, metadata={"source": file_name})

    @staticmethod
    def _read_pdf(file_path: str) -> str:
        """Extract text from a PDF using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required to read PDF files. Install with: pip install pypdf")

        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
