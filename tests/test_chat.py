"""Tests for the stateless RAG chat API."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document as LangChainDocument
from pydantic import ValidationError

from app.api.chat import get_rag_service, router
from app.core.exceptions import LLMServiceError, RetrievalError
from app.core.exceptions import register_exception_handlers
from app.schemas.chat import ChatRequest
from app.services.rag_service import NO_RELEVANT_INFORMATION, RAGResult, RAGService


def _client(service: Mock) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_rag_service] = lambda: service
    return TestClient(app)


def _source(
    content: str,
    *,
    document_id: int,
    filename: str,
    page_number: int,
    score: float | None,
) -> LangChainDocument:
    metadata = {
        "document_id": document_id,
        "filename": filename,
        "page_number": page_number,
    }
    if score is not None:
        metadata["similarity_score"] = score
    return LangChainDocument(page_content=content, metadata=metadata)


def test_chat_request_strips_question_and_forbids_memory_fields() -> None:
    assert ChatRequest(question="  What is the policy?  ").question == (
        "What is the policy?"
    )

    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {"question": "What is the policy?", "conversation_id": 10}
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": 42},
        {"question": "x" * 4001},
    ],
)
def test_chat_request_rejects_invalid_questions(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(payload)


def test_chat_endpoint_returns_answer_sources_pages_and_excerpts() -> None:
    service = Mock(spec=RAGService)
    source_documents = (
        _source(
            "Records are retained for seven years.",
            document_id=10,
            filename="retention.pdf",
            page_number=4,
            score=0.94,
        ),
        _source(
            "The policy is reviewed annually.",
            document_id=11,
            filename="governance.pdf",
            page_number=2,
            score=None,
        ),
    )
    service.answer.return_value = RAGResult(
        answer=(
            "Records are retained for seven years [Source 1], and the policy "
            "is reviewed annually [Source 2]."
        ),
        source_documents=source_documents,
    )

    with _client(service) as client:
        response = client.post(
            "/chat",
            json={"question": "  What is the retention policy?  "},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "Records are retained for seven years [Source 1], and the policy "
            "is reviewed annually [Source 2]."
        ),
        "sources": [
            {
                "citation": "[Source 1]",
                "document_id": 10,
                "filename": "retention.pdf",
                "page_number": 4,
                "excerpt": "Records are retained for seven years.",
                "relevance_score": 0.94,
            },
            {
                "citation": "[Source 2]",
                "document_id": 11,
                "filename": "governance.pdf",
                "page_number": 2,
                "excerpt": "The policy is reviewed annually.",
                "relevance_score": None,
            },
        ],
    }
    service.answer.assert_called_once_with("What is the retention policy?")


def test_chat_endpoint_returns_fallback_without_sources() -> None:
    service = Mock(spec=RAGService)
    service.answer.return_value = RAGResult(
        answer=NO_RELEVANT_INFORMATION,
        source_documents=(),
    )

    with _client(service) as client:
        response = client.post(
            "/chat",
            json={"question": "What is not indexed?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": NO_RELEVANT_INFORMATION,
        "sources": [],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "   "},
        {"question": "Valid", "conversation_id": "not-supported"},
    ],
)
def test_chat_endpoint_returns_consistent_validation_errors(
    payload: dict[str, object],
) -> None:
    service = Mock(spec=RAGService)

    with _client(service) as client:
        response = client.post("/chat", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    service.answer.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (RetrievalError("Retrieval failed."), 500, "retrieval_error"),
        (LLMServiceError("Ollama unavailable."), 502, "llm_service_error"),
    ],
)
def test_chat_endpoint_returns_controlled_pipeline_errors(
    error: Exception,
    status_code: int,
    error_code: str,
) -> None:
    service = Mock(spec=RAGService)
    service.answer.side_effect = error

    with _client(service) as client:
        response = client.post("/chat", json={"question": "Question"})

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == error_code


def test_chat_endpoint_rejects_invalid_source_metadata() -> None:
    service = Mock(spec=RAGService)
    service.answer.return_value = RAGResult(
        answer="Answer [Source 1].",
        source_documents=(
            LangChainDocument(
                page_content="Evidence without required metadata.",
                metadata={"filename": "incomplete.pdf"},
            ),
        ),
    )

    with _client(service) as client:
        response = client.post("/chat", json={"question": "Question"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "retrieval_error"
