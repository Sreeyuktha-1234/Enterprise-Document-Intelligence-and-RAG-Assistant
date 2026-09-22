"""Tests for LangChain document chunking."""

import pytest
from langchain_core.documents import Document as LangChainDocument

from app.services.chunking_service import ChunkingService, settings
from app.services.extraction_service import ExtractedPage, ExtractionResult


def _page(
    page_number: int,
    text: str,
    **metadata,
) -> ExtractedPage:
    return ExtractedPage(
        page_number=page_number,
        text=text,
        metadata={
            "source": "data/uploads/report.pdf",
            "page": page_number - 1,
            "page_number": page_number,
            "total_pages": 2,
            **metadata,
        },
    )


def test_chunk_document_returns_langchain_documents_with_metadata() -> None:
    content = ExtractionResult(
        source="data/uploads/report.pdf",
        pages=(
            _page(
                1,
                "Risk controls are reviewed quarterly. " * 8,
                department="Risk",
            ),
            _page(
                2,
                "Financial evidence is retained for seven years. " * 8,
                department="Finance",
            ),
        ),
    )

    chunks = ChunkingService(chunk_size=120, chunk_overlap=20).chunk_document(
        content,
        document_id=42,
        filename="report.pdf",
    )

    assert len(chunks) > 2
    assert all(isinstance(chunk, LangChainDocument) for chunk in chunks)
    assert all(len(chunk.page_content) <= 120 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert {chunk.metadata["document_id"] for chunk in chunks} == {42}
    assert {chunk.metadata["filename"] for chunk in chunks} == {"report.pdf"}
    assert {chunk.metadata["page_number"] for chunk in chunks} == {1, 2}
    assert {chunk.metadata["source"] for chunk in chunks} == {
        "data/uploads/report.pdf"
    }
    assert {chunk.metadata["department"] for chunk in chunks} == {
        "Risk",
        "Finance",
    }


def test_chunk_document_preserves_page_boundaries() -> None:
    content = ExtractionResult(
        source="report.pdf",
        pages=(
            _page(1, "Page one content."),
            _page(2, "Page two content."),
        ),
    )

    chunks = ChunkingService(chunk_size=100, chunk_overlap=10).chunk_document(
        content,
        document_id=7,
        filename="report.pdf",
    )

    assert len(chunks) == 2
    assert chunks[0].page_content == "Page one content."
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[1].page_content == "Page two content."
    assert chunks[1].metadata["page_number"] == 2


def test_chunk_document_skips_empty_pages_safely() -> None:
    content = ExtractionResult(
        source="empty.pdf",
        pages=(_page(1, ""), _page(2, "  \n")),
    )

    chunks = ChunkingService(chunk_size=100, chunk_overlap=10).chunk_document(
        content,
        document_id=1,
        filename="empty.pdf",
    )

    assert chunks == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "greater than zero"),
        (100, -1, "cannot be negative"),
        (100, 100, "smaller than chunk_size"),
        (100, 101, "smaller than chunk_size"),
    ],
)
def test_chunking_configuration_is_validated(
    chunk_size: int,
    chunk_overlap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_chunking_uses_application_defaults() -> None:
    service = ChunkingService()

    assert service.chunk_size == settings.CHUNK_SIZE
    assert service.chunk_overlap == settings.CHUNK_OVERLAP
