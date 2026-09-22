"""LangChain-based chunking for page-aware document content."""

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.extraction_service import ExtractionResult


logger = get_logger(__name__)
settings = get_settings()


class ChunkingService:
    """Split preprocessed pages into metadata-rich LangChain documents."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = (
            settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        )
        self._validate_configuration()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            keep_separator=True,
        )

    def chunk_document(
        self,
        content: ExtractionResult,
        *,
        document_id: int,
        filename: str,
    ) -> list[LangChainDocument]:
        """Convert page content to LangChain documents and split it into chunks."""

        page_documents: list[LangChainDocument] = []
        for page_document, page in zip(
            content.to_langchain_documents(),
            content.pages,
            strict=True,
        ):
            if not page_document.page_content.strip():
                continue

            page_documents.append(
                LangChainDocument(
                    page_content=page_document.page_content,
                    metadata={
                        **page_document.metadata,
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page.page_number,
                    },
                )
            )

        split_documents = self.splitter.split_documents(page_documents)
        chunks = [
            LangChainDocument(
                page_content=document.page_content,
                metadata={**document.metadata, "chunk_index": index},
            )
            for index, document in enumerate(split_documents)
        ]

        logger.info(
            "Created %s chunks from %s pages for document ID %s",
            len(chunks),
            len(page_documents),
            document_id,
        )
        return chunks

    def _validate_configuration(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
