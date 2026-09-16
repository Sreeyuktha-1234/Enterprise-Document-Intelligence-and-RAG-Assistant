"""Document storage and metadata persistence services."""

from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import DocumentProcessingError, UnsupportedFileTypeError
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

    @staticmethod
    def _remove_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove orphaned upload %s", path, exc_info=True)
