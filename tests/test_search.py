"""Tests for semantic search schemas and API behavior."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document as LangChainDocument
from pydantic import ValidationError

from app.api.search import get_retrieval_service, router
from app.core.exceptions import RetrievalError, register_exception_handlers
from app.schemas.search import SearchRequest
from app.services.retrieval_service import RetrievalService


def _client(service: Mock) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_retrieval_service] = lambda: service
    return TestClient(app)


def _result_document() -> LangChainDocument:
    return LangChainDocument(
        page_content="Quarterly risk controls require documented approval.",
        metadata={
            "filename": "risk-policy.pdf",
            "page_number": 7,
            "document_id": 42,
            "chunk_index": 3,
            "similarity_score": 0.91,
            "distance": 0.098,
        },
    )


def test_search_request_uses_configured_default_top_k() -> None:
    request = SearchRequest(query="  risk controls  ")

    assert request.query == "risk controls"
    assert request.top_k > 0


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "risk", "top_k": 0},
        {"query": "risk", "top_k": 101},
        {"query": "risk", "top_k": True},
        {"query": "risk", "unknown": "value"},
    ],
)
def test_search_request_rejects_invalid_input(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)


def test_search_endpoint_returns_source_and_relevance_information() -> None:
    service = Mock(spec=RetrievalService)
    service.retrieve.return_value = [_result_document()]

    with _client(service) as client:
        response = client.post(
            "/search",
            json={"query": "risk approvals", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json() == {
        "query": "risk approvals",
        "top_k": 3,
        "total": 1,
        "results": [
            {
                "chunk_content": (
                    "Quarterly risk controls require documented approval."
                ),
                "filename": "risk-policy.pdf",
                "page_number": 7,
                "document_id": 42,
                "relevance_score": 0.91,
                "distance": 0.098,
            }
        ],
    }
    service.retrieve.assert_called_once_with("risk approvals", top_k=3)


def test_search_endpoint_returns_empty_result_collection() -> None:
    service = Mock(spec=RetrievalService)
    service.retrieve.return_value = []

    with _client(service) as client:
        response = client.post("/search", json={"query": "missing topic"})

    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["results"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": "   "},
        {"query": "risk", "top_k": 0},
        {"query": "risk", "top_k": 101},
        {"query": "risk", "extra": "forbidden"},
    ],
)
def test_search_endpoint_returns_validation_error(
    payload: dict[str, object],
) -> None:
    service = Mock(spec=RetrievalService)

    with _client(service) as client:
        response = client.post("/search", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    service.retrieve.assert_not_called()


def test_search_endpoint_returns_controlled_retrieval_error() -> None:
    service = Mock(spec=RetrievalService)
    service.retrieve.side_effect = RetrievalError("Vector search unavailable.")

    with _client(service) as client:
        response = client.post("/search", json={"query": "risk"})

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "retrieval_error",
        "message": "Vector search unavailable.",
        "details": None,
    }


def test_search_endpoint_rejects_invalid_retrieved_metadata() -> None:
    service = Mock(spec=RetrievalService)
    service.retrieve.return_value = [
        LangChainDocument(
            page_content="Chunk without source metadata.",
            metadata={"similarity_score": 0.8, "distance": 0.2},
        )
    ]

    with _client(service) as client:
        response = client.post("/search", json={"query": "risk"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "retrieval_error"
