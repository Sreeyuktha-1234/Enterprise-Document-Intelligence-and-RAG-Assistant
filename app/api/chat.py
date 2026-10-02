"""Stateless retrieval-augmented chat API."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from langchain_core.documents import Document as LangChainDocument
from pydantic import ValidationError

from app.core.exceptions import RetrievalError
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.rag_service import RAGService


router = APIRouter(tags=["Chat"])


def get_rag_service() -> RAGService:
    """Create the stateless RAG service dependency."""

    return RAGService()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a grounded document question",
)
def chat(
    request: ChatRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatResponse:
    """Answer one question and return the document chunks used as evidence."""

    result = rag_service.answer(request.question)
    sources = [
        _to_chat_source(document, source_number=index)
        for index, document in enumerate(result.source_documents, start=1)
    ]
    return ChatResponse(answer=result.answer, sources=sources)


def _to_chat_source(
    document: LangChainDocument,
    *,
    source_number: int,
) -> ChatSource:
    """Map one LangChain source document to the public citation schema."""

    metadata: dict[str, Any] = document.metadata
    try:
        return ChatSource(
            citation=f"[Source {source_number}]",
            document_id=metadata["document_id"],
            filename=metadata["filename"],
            page_number=metadata["page_number"],
            excerpt=document.page_content.strip(),
            relevance_score=metadata.get("similarity_score"),
        )
    except (KeyError, TypeError, ValidationError) as exc:
        raise RetrievalError(
            "A RAG source contained invalid document metadata."
        ) from exc
