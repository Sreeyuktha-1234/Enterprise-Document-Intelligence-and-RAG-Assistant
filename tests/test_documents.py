"""Tests for the document model and API schemas."""

from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.datastructures import Headers

from app import main
from app.core.database import Base, get_db
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    UnsupportedFileTypeError,
)
from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentResponse, DocumentUpdate
from app.services import document_service
from app.services.document_service import DocumentService


DOCUMENT_DATA = {
    "filename": "stored-document.pdf",
    "original_filename": "Quarterly Report.pdf",
    "file_path": "data/uploads/stored-document.pdf",
    "file_size": 4096,
    "content_type": "application/pdf",
}

PDF_BYTES = b"%PDF-1.7\n% test document\n%%EOF"


def _upload_file(
    *,
    filename: str = "report.pdf",
    content_type: str = "application/pdf",
    content: bytes = PDF_BYTES,
) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def test_document_model_contains_required_columns() -> None:
    columns = inspect(Document).columns

    assert set(columns.keys()) == {
        "id",
        "filename",
        "original_filename",
        "file_path",
        "file_size",
        "content_type",
        "status",
        "created_at",
        "updated_at",
    }
    assert columns.id.primary_key is True
    assert columns.file_size.nullable is False


def test_document_defaults_and_timestamps_are_persisted() -> None:
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        document = Document(**DOCUMENT_DATA)
        session.add(document)
        session.flush()

        assert document.id is not None
        assert document.status == "uploaded"
        assert document.created_at is not None
        assert document.updated_at is not None

    test_engine.dispose()


def test_document_create_validates_request_data() -> None:
    document = DocumentCreate(**DOCUMENT_DATA)

    assert document.status == "uploaded"
    assert document.file_size == 4096


def test_document_create_rejects_negative_file_size() -> None:
    with pytest.raises(ValidationError):
        DocumentCreate(**{**DOCUMENT_DATA, "file_size": -1})


def test_document_update_allows_partial_changes() -> None:
    update = DocumentUpdate(status="processed")

    assert update.model_dump(exclude_unset=True) == {"status": "processed"}


def test_document_response_serializes_orm_model() -> None:
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        document = Document(**DOCUMENT_DATA)
        session.add(document)
        session.flush()

        response = DocumentResponse.model_validate(document)

        assert response.id == document.id
        assert response.filename == document.filename
        assert response.created_at == document.created_at

    test_engine.dispose()


def test_document_service_saves_pdf_with_unique_safe_filename(tmp_path: Path) -> None:
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        service = DocumentService(session, tmp_path)
        first = service.upload_pdf(_upload_file(filename="../Quarterly Report.PDF"))
        second = service.upload_pdf(_upload_file(filename="Quarterly Report.PDF"))

        assert first.filename != second.filename
        assert first.filename.endswith(".pdf")
        assert first.original_filename == "Quarterly Report.PDF"
        assert first.file_size == len(PDF_BYTES)
        assert Path(first.file_path).read_bytes() == PDF_BYTES
        assert session.query(Document).count() == 2

    test_engine.dispose()


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        ("report.txt", "application/pdf", PDF_BYTES),
        ("report.pdf", "text/plain", PDF_BYTES),
        ("report.pdf", "application/pdf", b"not a pdf"),
    ],
)
def test_document_service_rejects_non_pdf_uploads(
    tmp_path: Path,
    filename: str,
    content_type: str,
    content: bytes,
) -> None:
    session = Mock(spec=Session)
    service = DocumentService(session, tmp_path)

    with pytest.raises(UnsupportedFileTypeError):
        service.upload_pdf(
            _upload_file(
                filename=filename,
                content_type=content_type,
                content=content,
            )
        )

    session.add.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_document_service_removes_file_when_database_write_fails(
    tmp_path: Path,
) -> None:
    session = Mock(spec=Session)
    session.commit.side_effect = SQLAlchemyError("database unavailable")
    service = DocumentService(session, tmp_path)

    with pytest.raises(DocumentProcessingError):
        service.upload_pdf(_upload_file())

    session.rollback.assert_called_once_with()
    assert list(tmp_path.iterdir()) == []


def test_document_service_lists_documents_with_pagination(tmp_path: Path) -> None:
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        service = DocumentService(session, tmp_path)
        documents = [
            service.upload_pdf(_upload_file(filename=f"report-{index}.pdf"))
            for index in range(3)
        ]

        page, total = service.list_documents(skip=1, limit=1)

        assert total == 3
        assert [document.id for document in page] == [documents[1].id]

    test_engine.dispose()


def test_document_service_gets_and_deletes_document_and_file(tmp_path: Path) -> None:
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        service = DocumentService(session, tmp_path)
        document = service.upload_pdf(_upload_file())
        document_id = document.id
        uploaded_path = Path(document.file_path)

        assert service.get_document(document_id).id == document_id
        assert uploaded_path.is_file()

        service.delete_document(document_id)

        assert session.get(Document, document_id) is None
        assert not uploaded_path.exists()
        with pytest.raises(DocumentNotFoundError):
            service.get_document(document_id)
        with pytest.raises(DocumentNotFoundError):
            service.delete_document(document_id)

    test_engine.dispose()


def test_document_management_api_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    testing_session = sessionmaker(bind=test_engine, expire_on_commit=False)

    def override_get_db():
        with testing_session() as session:
            yield session

    main.app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(document_service.settings, "UPLOAD_DIRECTORY", tmp_path)

    try:
        with TestClient(main.app) as client:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("report.pdf", PDF_BYTES, "application/pdf")},
            )
            assert upload_response.status_code == 201
            payload = upload_response.json()
            document_id = payload["id"]
            uploaded_path = tmp_path / payload["filename"]

            get_response = client.get(f"/documents/{document_id}")
            list_response = client.get("/documents", params={"skip": 0, "limit": 1})
            delete_response = client.delete(f"/documents/{document_id}")
            missing_get = client.get(f"/documents/{document_id}")
            missing_delete = client.delete(f"/documents/{document_id}")

        assert payload["original_filename"] == "report.pdf"
        assert payload["status"] == "uploaded"
        assert payload["file_size"] == len(PDF_BYTES)
        assert get_response.status_code == 200
        assert get_response.json()["id"] == document_id
        assert list_response.status_code == 200
        assert list_response.json()["total"] == 1
        assert list_response.json()["items"] == [payload]
        assert list_response.json()["skip"] == 0
        assert list_response.json()["limit"] == 1
        assert delete_response.status_code == 204
        assert delete_response.content == b""
        assert not uploaded_path.exists()
        assert missing_get.status_code == 404
        assert missing_get.json()["error"]["code"] == "document_not_found"
        assert missing_delete.status_code == 404
        assert missing_delete.json()["error"]["code"] == "document_not_found"

        with testing_session() as session:
            assert session.query(Document).count() == 0
    finally:
        main.app.dependency_overrides.clear()
        test_engine.dispose()


def test_upload_endpoint_returns_consistent_error_for_non_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(document_service.settings, "UPLOAD_DIRECTORY", tmp_path)

    with TestClient(main.app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"plain text", "text/plain")},
        )

    assert response.status_code == 415
    assert response.json() == {
        "error": {
            "code": "unsupported_file_type",
            "message": "Only PDF files are supported.",
            "details": None,
        }
    }
