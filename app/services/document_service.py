"""Document storage and metadata persistence services."""

from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.models.document import Document


logger = get_logger(__name__)
settings = get_settings()

PDF_CONTENT_TYPE = "application/pdf"
PDF_EXTENSION = ".pdf"
PDF_SIGNATURE = b"%PDF-"
COPY_CHUNK_SIZE = 1024 * 1024


class DocumentService:
    """Save uploaded documents and persist their metadata."""

    def __init__(
        self,
        db: Session,
        upload_directory: Path | None = None,
    ) -> None:
        self.db = db
        self.upload_directory = Path(upload_directory or settings.UPLOAD_DIRECTORY)

    def upload_pdf(self, upload: UploadFile) -> Document:
        """Validate, store, and persist one PDF upload."""

        original_filename = self._validate_pdf(upload)
        stored_filename = f"{uuid4().hex}{PDF_EXTENSION}"
        destination = self.upload_directory / stored_filename

        try:
            self.upload_directory.mkdir(parents=True, exist_ok=True)
            file_size = self._save_upload(upload, destination)
        except (OSError, ValueError) as exc:
            self._remove_file(destination)
            logger.exception("Failed to save uploaded PDF %s", original_filename)
            raise DocumentProcessingError(
                "The uploaded PDF could not be saved."
            ) from exc

        document = Document(
            filename=stored_filename,
            original_filename=original_filename,
            file_path=destination.as_posix(),
            file_size=file_size,
            content_type=PDF_CONTENT_TYPE,
            status="uploaded",
        )

        try:
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
        except SQLAlchemyError as exc:
            self.db.rollback()
            self._remove_file(destination)
            logger.exception(
                "Failed to persist metadata for uploaded PDF %s",
                original_filename,
            )
            raise DocumentProcessingError(
                "The uploaded PDF metadata could not be stored."
            ) from exc

        logger.info("Stored PDF document %s as %s", original_filename, stored_filename)
        return document

    def list_documents(self, skip: int, limit: int) -> tuple[list[Document], int]:
        """Return one page of documents and the total record count."""

        total = self.db.scalar(select(func.count()).select_from(Document)) or 0
        statement = (
            select(Document)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .offset(skip)
            .limit(limit)
        )
        documents = list(self.db.scalars(statement).all())
        return documents, total

    def get_document(self, document_id: int) -> Document:
        """Return a document or raise the shared not-found error."""

        document = self.db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(
                f"Document with ID {document_id} was not found."
            )
        return document

    def delete_document(self, document_id: int) -> None:
        """Delete document metadata and its managed uploaded file."""

        document = self.get_document(document_id)
        file_path = self._resolve_managed_path(document.file_path)
        staged_path: Path | None = None

        if file_path.exists():
            staged_path = file_path.with_name(
                f".{file_path.name}.{uuid4().hex}.deleting"
            )
            try:
                file_path.replace(staged_path)
            except OSError as exc:
                raise DocumentProcessingError(
                    "The document file could not be prepared for deletion."
                ) from exc

        try:
            self.db.delete(document)
            self.db.commit()
        except SQLAlchemyError as exc:
            self.db.rollback()
            if staged_path is not None:
                self._restore_staged_file(staged_path, file_path)
            logger.exception("Failed to delete document metadata for ID %s", document_id)
            raise DocumentProcessingError(
                "The document metadata could not be deleted."
            ) from exc

        if staged_path is not None:
            self._remove_file(staged_path)
        logger.info("Deleted document ID %s", document_id)

    @staticmethod
    def _validate_pdf(upload: UploadFile) -> str:
        raw_filename = upload.filename or ""
        original_filename = PurePosixPath(raw_filename.replace("\\", "/")).name
        content_type = (upload.content_type or "").split(";", maxsplit=1)[0].lower()

        if (
            not original_filename
            or Path(original_filename).suffix.lower() != PDF_EXTENSION
            or content_type != PDF_CONTENT_TYPE
        ):
            raise UnsupportedFileTypeError("Only PDF files are supported.")

        try:
            header = upload.file.read(1024)
            upload.file.seek(0)
        except (OSError, ValueError) as exc:
            raise DocumentProcessingError(
                "The uploaded file could not be read."
            ) from exc

        if PDF_SIGNATURE not in header:
            raise UnsupportedFileTypeError(
                "The uploaded file does not contain valid PDF data."
            )

        return original_filename

    @staticmethod
    def _save_upload(upload: UploadFile, destination: Path) -> int:
        file_size = 0
        with destination.open("xb") as output:
            while chunk := upload.file.read(COPY_CHUNK_SIZE):
                output.write(chunk)
                file_size += len(chunk)
        return file_size

    def _resolve_managed_path(self, stored_path: str) -> Path:
        upload_root = self.upload_directory.resolve()
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved_path = candidate.resolve()

        if not resolved_path.is_relative_to(upload_root):
            logger.error(
                "Refusing to delete document path outside upload directory: %s",
                resolved_path,
            )
            raise DocumentProcessingError(
                "The stored document path is outside the upload directory."
            )
        return resolved_path

    @staticmethod
    def _restore_staged_file(staged_path: Path, original_path: Path) -> None:
        try:
            staged_path.replace(original_path)
        except OSError:
            logger.error(
                "Could not restore document file %s after database rollback",
                original_path,
                exc_info=True,
            )

    @staticmethod
    def _remove_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove orphaned upload %s", path, exc_info=True)
