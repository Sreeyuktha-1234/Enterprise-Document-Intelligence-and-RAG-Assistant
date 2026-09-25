"""Orchestration for the end-to-end document ingestion pipeline."""

from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger
from app.schemas.processing import ProcessingResponse, ProcessingStatistics
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.extraction_service import ExtractionService
from app.services.preprocessing_service import PreprocessingService
from app.services.vector_store_service import VectorStoreService


logger = get_logger(__name__)


class DocumentProcessingService:
    """Coordinate ingestion while keeping each pipeline stage independent."""

    def __init__(
        self,
        document_service: DocumentService,
        extraction_service: ExtractionService | None = None,
        preprocessing_service: PreprocessingService | None = None,
        chunking_service: ChunkingService | None = None,
        vector_store_service: VectorStoreService | None = None,
    ) -> None:
        self.document_service = document_service
        self.extraction_service = extraction_service or ExtractionService()
        self.preprocessing_service = preprocessing_service or PreprocessingService()
        self.chunking_service = chunking_service or ChunkingService()
        self.vector_store_service = vector_store_service or VectorStoreService()

    def process_document(self, document_id: int) -> ProcessingResponse:
        """Process one PDF and atomically replace its indexed chunks."""

        document = self.document_service.get_document(document_id)
        self.document_service.update_status(document, "processing")

        try:
            extracted = self.extraction_service.extract_pdf(document.file_path)
            cleaned = self.preprocessing_service.preprocess(extracted)
            chunks = self.chunking_service.chunk_document(
                cleaned,
                document_id=document.id,
                filename=document.original_filename,
            )
            if not chunks:
                raise DocumentProcessingError(
                    "The document did not contain any processable text."
                )

            self.vector_store_service.replace_document(
                document.id,
                chunks,
            )
            self.document_service.update_status(document, "processed")
        except Exception as exc:
            logger.exception("Processing failed for document ID %s", document_id)
            try:
                self.document_service.update_status(document, "failed")
            except DocumentProcessingError:
                logger.exception(
                    "Could not mark document ID %s as failed", document_id
                )

            if isinstance(exc, DocumentProcessingError):
                raise
            raise DocumentProcessingError(
                "The document processing pipeline failed."
            ) from exc

        statistics = ProcessingStatistics(
            pages_extracted=extracted.total_pages,
            pages_processed=sum(bool(page.text.strip()) for page in cleaned.pages),
            chunks_created=len(chunks),
            characters_extracted=len(extracted.full_text),
            characters_processed=len(cleaned.full_text),
        )
        logger.info(
            "Processed document ID %s into %s chunks",
            document_id,
            len(chunks),
        )
        return ProcessingResponse(
            document_id=document.id,
            status=document.status,
            statistics=statistics,
        )
