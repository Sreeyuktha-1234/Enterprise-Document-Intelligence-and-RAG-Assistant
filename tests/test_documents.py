"""Tests for the document model and API schemas."""

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from app.core.database import Base
from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentResponse, DocumentUpdate


DOCUMENT_DATA = {
    "filename": "stored-document.pdf",
    "original_filename": "Quarterly Report.pdf",
    "file_path": "data/uploads/stored-document.pdf",
    "file_size": 4096,
    "content_type": "application/pdf",
}


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
