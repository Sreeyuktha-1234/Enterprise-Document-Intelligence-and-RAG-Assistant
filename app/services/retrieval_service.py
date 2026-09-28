"""Reusable semantic retrieval backed by the persisted LangChain FAISS index."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LangChainDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.services.vector_store_service import VectorStoreService

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS


logger = get_logger(__name__)
settings = get_settings()


class EmptyRetriever(BaseRetriever):
    """LangChain-compatible retriever used before any documents are indexed."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[LangChainDocument]:
        return []


class RetrievalService:
    """Load a FAISS index and expose scored and LangChain retrieval APIs."""

    def __init__(
        self,
        vector_store_service: VectorStoreService | None = None,
        top_k: int | None = None,
    ) -> None:
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.top_k = settings.RETRIEVAL_TOP_K if top_k is None else top_k
        self._validate_top_k(self.top_k)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[LangChainDocument]:
        """Return relevant documents with FAISS distance and similarity metadata."""

        if not query.strip():
            raise ValueError("query must contain non-whitespace text")
        result_limit = self.top_k if top_k is None else top_k
        self._validate_top_k(result_limit)

        vector_store = self._load_vector_store()
        if vector_store is None or vector_store.index.ntotal == 0:
            logger.info("No indexed documents are available for retrieval")
            return []

        try:
            matches = vector_store.similarity_search_with_score(
                query,
                k=min(result_limit, vector_store.index.ntotal),
            )
        except Exception as exc:
            logger.exception("Semantic retrieval failed")
            raise RetrievalError("Semantic document retrieval failed.") from exc

        documents = [
            self._with_similarity_metadata(document, distance)
            for document, distance in matches
        ]
        logger.info("Retrieved %s documents for semantic query", len(documents))
        return documents

    def as_retriever(
        self,
        *,
        top_k: int | None = None,
        search_kwargs: dict[str, Any] | None = None,
    ) -> BaseRetriever:
        """Return a standard LangChain retriever for chains and RAG services."""

        result_limit = self.top_k if top_k is None else top_k
        self._validate_top_k(result_limit)
        vector_store = self._load_vector_store()
        if vector_store is None or vector_store.index.ntotal == 0:
            return EmptyRetriever()

        options = dict(search_kwargs or {})
        options["k"] = min(result_limit, vector_store.index.ntotal)
        return vector_store.as_retriever(search_kwargs=options)

    def _load_vector_store(self) -> FAISS | None:
        try:
            if self.vector_store_service.vector_store is not None:
                return self.vector_store_service.vector_store
            return self.vector_store_service.load_vector_store_if_exists()
        except Exception as exc:
            logger.exception("Could not load the FAISS index for retrieval")
            raise RetrievalError(
                "The document vector store could not be loaded."
            ) from exc

    @staticmethod
    def _with_similarity_metadata(
        document: LangChainDocument,
        distance: float,
    ) -> LangChainDocument:
        numeric_distance = float(distance)
        similarity = 1.0 / (1.0 + max(numeric_distance, 0.0))
        return LangChainDocument(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "distance": numeric_distance,
                "similarity_score": similarity,
            },
        )

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
