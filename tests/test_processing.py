"""Tests for end-to-end document processing orchestration and API."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document as LangChainDocument
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import main
from app.api.processing import get_processing_service
from app.core.database import Base
from app.core.exceptions import DocumentProcessingError
from app.models.document import Document
from app.schemas.processing import ProcessingResponse, ProcessingStatistics
from app.services.document_service import DocumentService
from app.services.extraction_service import ExtractedPage, ExtractionResult
from app.services.processing_service import DocumentProcessingService


def _stored_document(path: Path) -> Document:
    return Document(
        filename="stored.pdf",
        original_filename="report.pdf",
        file_path=path.as_posix(),
        file_size=100,
        content_type="application/pdf",
    )


def _content(text: str = "Enterprise policy content.") -> ExtractionResult:
    return ExtractionResult(
        source="stored.pdf",
        pages=(
            ExtractedPage(
                page_number=1,
                text=text,
                metadata={"source": "stored.pdf", "page_number": 1},
            ),
        ),
    )


def test_processing_pipeline_updates_status_and_returns_statistics(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    source = tmp_path / "stored.pdf"
    source.write_bytes(b"%PDF-test")

    with Session(engine) as session:
        document = _stored_document(source)
        session.add(document)
        session.commit()
        session.refresh(document)
        document_id = document.id

        extraction = Mock()
        extraction.extract_pdf.return_value = _content()
        preprocessing = Mock()
        preprocessing.preprocess.return_value = _content("Clean policy content.")
        chunks = [
            LangChainDocument(
                page_content="Clean policy content.",
                metadata={"document_id": document_id, "chunk_index": 0},
            )
        ]
        chunking = Mock()
        chunking.chunk_document.return_value = chunks
        vector_store = Mock()

        result = DocumentProcessingService(
            DocumentService(session, tmp_path),
            extraction,
            preprocessing,
            chunking,
            vector_store,
        ).process_document(document_id)

        assert result.document_id == document_id
        assert result.status == "processed"
        assert result.statistics.pages_extracted == 1
        assert result.statistics.pages_processed == 1
        assert result.statistics.chunks_created == 1
        assert session.get(Document, document_id).status == "processed"
        vector_store.replace_document.assert_called_once_with(document_id, chunks)

    engine.dispose()


def test_processing_failure_marks_document_failed(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        document = _stored_document(tmp_path / "stored.pdf")
        session.add(document)
        session.commit()
        session.refresh(document)
        document_id = document.id
        extraction = Mock()
        extraction.extract_pdf.side_effect = RuntimeError("internal detail")

        with pytest.raises(DocumentProcessingError, match="pipeline failed"):
            DocumentProcessingService(
                DocumentService(session, tmp_path),
                extraction_service=extraction,
                preprocessing_service=Mock(),
                chunking_service=Mock(),
                vector_store_service=Mock(),
            ).process_document(document_id)

        assert session.get(Document, document_id).status == "failed"

    engine.dispose()


def test_processing_endpoint_returns_pipeline_response() -> None:
    response = ProcessingResponse(
        document_id=42,
        status="processed",
        statistics=ProcessingStatistics(
            pages_extracted=2,
            pages_processed=2,
            chunks_created=5,
            characters_extracted=100,
            characters_processed=90,
        ),
    )
    service = Mock(spec=DocumentProcessingService)
    service.process_document.return_value = response
    main.app.dependency_overrides[get_processing_service] = lambda: service

    try:
        with TestClient(main.app) as client:
            api_response = client.post("/documents/42/process")
    finally:
        main.app.dependency_overrides.clear()

    assert api_response.status_code == 200
    assert api_response.json() == response.model_dump()
