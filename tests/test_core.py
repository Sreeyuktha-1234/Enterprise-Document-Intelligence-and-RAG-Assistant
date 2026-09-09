"""Tests for centralized logging and exception handling."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import (
    AuthenticationError,
    DocumentNotFoundError,
    DocumentProcessingError,
    LLMServiceError,
    RetrievalError,
    UnsupportedFileTypeError,
    register_exception_handlers,
)
from app.core.logging import configure_logging, get_logger


@pytest.mark.parametrize(
    ("exception_type", "status_code", "error_code"),
    [
        (DocumentNotFoundError, 404, "document_not_found"),
        (DocumentProcessingError, 500, "document_processing_error"),
        (UnsupportedFileTypeError, 415, "unsupported_file_type"),
        (RetrievalError, 500, "retrieval_error"),
        (LLMServiceError, 502, "llm_service_error"),
        (AuthenticationError, 401, "authentication_error"),
    ],
)
def test_application_exception_metadata(
    exception_type: type[Exception], status_code: int, error_code: str
) -> None:
    exception = exception_type()

    assert exception.status_code == status_code
    assert exception.error_code == error_code
    assert str(exception)


def test_logging_includes_timestamp_level_and_name(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    get_logger("test.logger").info("configuration ready")

    output = capsys.readouterr().out
    assert "INFO" in output
    assert "test.logger" in output
    assert "configuration ready" in output
    assert output[:4].isdigit()


def _create_test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/document")
    async def document() -> None:
        raise DocumentNotFoundError(details={"document_id": "missing"})

    @app.get("/authenticated")
    async def authenticated() -> None:
        raise AuthenticationError()

    @app.get("/validated")
    async def validated(limit: int) -> dict[str, int]:
        return {"limit": limit}

    @app.get("/unexpected")
    async def unexpected() -> None:
        raise RuntimeError("sensitive internal message")

    return app


def test_custom_error_response_uses_standard_envelope() -> None:
    client = TestClient(_create_test_app())

    response = client.get("/document")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "document_not_found",
            "message": "The requested document was not found.",
            "details": {"document_id": "missing"},
        }
    }


def test_authentication_error_sets_challenge_header() -> None:
    client = TestClient(_create_test_app())

    response = client.get("/authenticated")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "authentication_error"


def test_framework_errors_use_standard_envelope() -> None:
    client = TestClient(_create_test_app())

    not_found = client.get("/missing")
    invalid = client.get("/validated", params={"limit": "invalid"})

    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "http_error"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
    assert invalid.json()["error"]["details"]


def test_unexpected_errors_do_not_expose_internal_details() -> None:
    client = TestClient(_create_test_app(), raise_server_exceptions=False)

    response = client.get("/unexpected")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_server_error",
            "message": "An unexpected error occurred.",
            "details": None,
        }
    }
    assert "sensitive internal message" not in response.text
