"""Reusable Hugging Face embeddings for the LangChain RAG pipeline."""

from collections.abc import Sequence
from functools import lru_cache

from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()

DEFAULT_BATCH_SIZE = 32


@lru_cache(maxsize=8)
def get_embedding_backend(
    model_name: str,
    device: str,
    normalize_embeddings: bool,
    batch_size: int,
) -> Embeddings:
    """Load and cache a Hugging Face embedding model by configuration."""

    logger.info("Loading Hugging Face embedding model %s on %s", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
            "batch_size": batch_size,
        },
    )


class EmbeddingService:
    """Create document and query vectors through a LangChain embedding backend."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        backend: Embeddings | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.backend = (
            backend
            if backend is not None
            else get_embedding_backend(
                self.model_name,
                self.device,
                self.normalize_embeddings,
                self.batch_size,
            )
        )

    def embed_documents(
        self,
        documents: Sequence[LangChainDocument],
    ) -> list[list[float]]:
        """Embed LangChain documents in bounded batches."""

        return self.embed_texts([document.page_content for document in documents])

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed raw text in bounded batches while preserving input order."""

        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            batch_vectors = self.backend.embed_documents(batch)
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    "The embedding backend returned an unexpected vector count."
                )
            vectors.extend(batch_vectors)

        logger.debug("Embedded %s document texts", len(vectors))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed one non-empty retrieval query."""

        if not query.strip():
            raise ValueError("query must contain non-whitespace text")

        vector = self.backend.embed_query(query)
        logger.debug("Embedded query using model %s", self.model_name)
        return vector
