"""Page-aware PDF text extraction using PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangChainDocument


logger = get_logger(__name__)

PDF_READ_ERRORS = (
    pymupdf.FileDataError,
    pymupdf.EmptyFileError,
    OSError,
    RuntimeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text and source metadata extracted from one PDF page."""

    page_number: int
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of this page."""

        return {
            "page_number": self.page_number,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Structured text extraction result for an entire PDF."""

    source: str
    pages: tuple[ExtractedPage, ...]

    @property
    def total_pages(self) -> int:
        """Return the number of pages in the source PDF."""

        return len(self.pages)

    @property
    def full_text(self) -> str:
        """Return page text joined in source order."""

        return "\n\n".join(page.text for page in self.pages)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of the result."""

        return {
            "source": self.source,
            "total_pages": self.total_pages,
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_langchain_documents(self) -> list[LangChainDocument]:
        """Convert extracted pages to LangChain documents without losing metadata."""

        from langchain_core.documents import Document as LangChainDocument

        return [
            LangChainDocument(
                page_content=page.text,
                metadata=dict(page.metadata),
            )
            for page in self.pages
        ]


class ExtractionService:
    """Extract text and page metadata from PDF documents."""

    def extract_pdf(self, file_path: str | Path) -> ExtractionResult:
        """Extract text page by page from a readable, unencrypted PDF."""

        pdf_path = Path(file_path)
        source = pdf_path.as_posix()

        try:
            with pymupdf.open(pdf_path) as pdf:
                if pdf.needs_pass:
                    raise DocumentProcessingError(
                        "Password-protected PDFs cannot be extracted."
                    )

                total_pages = pdf.page_count
                pages = tuple(
                    self._extract_page(
                        page=page,
                        page_index=page_index,
                        total_pages=total_pages,
                        source=source,
                    )
                    for page_index, page in enumerate(pdf)
                )
        except DocumentProcessingError:
            logger.exception("PDF extraction failed for %s", pdf_path)
            raise
        except PDF_READ_ERRORS as exc:
            logger.exception("PDF extraction failed for %s", pdf_path)
            raise DocumentProcessingError(
                "The PDF is corrupted, unreadable, or unavailable."
            ) from exc

        logger.info("Extracted %s pages from %s", len(pages), pdf_path)
        return ExtractionResult(source=source, pages=pages)

    @staticmethod
    def _extract_page(
        *,
        page: pymupdf.Page,
        page_index: int,
        total_pages: int,
        source: str,
    ) -> ExtractedPage:
        page_number = page_index + 1
        try:
            text = page.get_text("text", sort=True)
        except PDF_READ_ERRORS as exc:
            raise DocumentProcessingError(
                f"Text extraction failed on PDF page {page_number}."
            ) from exc

        return ExtractedPage(
            page_number=page_number,
            text=text,
            metadata={
                "source": source,
                "page": page_index,
                "page_number": page_number,
                "total_pages": total_pages,
            },
        )
