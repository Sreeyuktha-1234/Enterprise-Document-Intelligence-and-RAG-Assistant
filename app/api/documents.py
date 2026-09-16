"""Document management API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentService


router = APIRouter(prefix="/documents", tags=["Documents"])


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
