"""Document processing API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.processing import ProcessingResponse
from app.services.document_service import DocumentService
from app.services.processing_service import DocumentProcessingService


router = APIRouter(prefix="/documents", tags=["Processing"])


def get_processing_service(
    db: Annotated[Session, Depends(get_db)],
) -> DocumentProcessingService:
    """Build the request-scoped document processing orchestrator."""

    return DocumentProcessingService(DocumentService(db))


@router.post(
    "/{document_id}/process",
    response_model=ProcessingResponse,
    summary="Process a document",
)
def process_document(
    document_id: int,
    service: Annotated[DocumentProcessingService, Depends(get_processing_service)],
) -> ProcessingResponse:
    """Run extraction, cleanup, chunking, embedding, and indexing."""

    return service.process_document(document_id)
