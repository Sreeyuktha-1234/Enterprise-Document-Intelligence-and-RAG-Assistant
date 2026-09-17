"""Document management API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.document_service import DocumentService


router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents",
)
def list_documents(
    db: Annotated[Session, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DocumentListResponse:
    """Return a paginated collection of stored documents."""

    documents, total = DocumentService(db).list_documents(skip, limit)
    return DocumentListResponse(
        items=documents,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
)
def upload_document(
    file: Annotated[UploadFile, File(description="PDF document to upload")],
    db: Annotated[Session, Depends(get_db)],
) -> Document:
    """Store one PDF and return its persisted metadata."""

    return DocumentService(db).upload_pdf(file)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Retrieve a document",
)
def get_document(
    document_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> Document:
    """Return metadata for one document."""

    return DocumentService(db).get_document(document_id)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
def delete_document(
    document_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    """Delete one document and its uploaded file."""

    DocumentService(db).delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
