"""Semantic search API endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from langchain_core.documents import Document as LangChainDocument
from pydantic import ValidationError

from app.core.exceptions import RetrievalError
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.retrieval_service import RetrievalService


router = APIRouter(tags=["Search"])


def get_retrieval_service() -> RetrievalService:
    """Create the semantic retrieval dependency."""

    return RetrievalService()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search document content",
)
def semantic_search(
    request: SearchRequest,
    retrieval_service: Annotated[
        RetrievalService,
        Depends(get_retrieval_service),
    ],
) -> SearchResponse:
    """Return document chunks most relevant to the supplied query."""

    documents = retrieval_service.retrieve(
        request.query,
        top_k=request.top_k,
    )
    results = [_to_search_result(document) for document in documents]
    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        total=len(results),
        results=results,
    )


def _to_search_result(document: LangChainDocument) -> SearchResult:
    """Map one internal LangChain document to the public API contract."""

    metadata: dict[str, Any] = document.metadata
    try:
        return SearchResult(
            chunk_content=document.page_content,
            filename=metadata["filename"],
            page_number=metadata["page_number"],
            document_id=metadata["document_id"],
            relevance_score=metadata["similarity_score"],
            distance=metadata["distance"],
        )
    except (KeyError, TypeError, ValidationError) as exc:
        raise RetrievalError(
            "A retrieved document contained invalid source metadata."
        ) from exc
