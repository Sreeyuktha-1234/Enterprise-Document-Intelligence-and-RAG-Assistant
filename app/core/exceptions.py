"""Application exceptions and reusable FastAPI exception handlers."""

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


class ApplicationError(Exception):
    """Base class for errors that can be safely returned to API clients."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "application_error"
    default_message = "An application error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details
        self.headers = dict(headers) if headers else None
        super().__init__(self.message)


class DocumentNotFoundError(ApplicationError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "document_not_found"
    default_message = "The requested document was not found."


class DocumentProcessingError(ApplicationError):
    error_code = "document_processing_error"
    default_message = "The document could not be processed."


class UnsupportedFileTypeError(ApplicationError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    error_code = "unsupported_file_type"
    default_message = "The uploaded file type is not supported."


class RetrievalError(ApplicationError):
    error_code = "retrieval_error"
    default_message = "The document retrieval operation failed."


class LLMServiceError(ApplicationError):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "llm_service_error"
    default_message = "The language model service is unavailable."


class AuthenticationError(ApplicationError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "authentication_error"
    default_message = "Authentication is required."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            message,
            details=details,
            headers=headers or {"WWW-Authenticate": "Bearer"},
        )


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build the standard API error response."""

    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "details": details,
                }
            }
        ),
        headers=headers,
    )


async def application_error_handler(
    request: Request, exc: ApplicationError
) -> JSONResponse:
    """Handle known application errors without exposing internal state."""

    logger.warning(
        "Application error on %s %s: %s",
        request.method,
        request.url.path,
        exc.message,
    )
    return _error_response(
        status_code=exc.status_code,
        code=exc.error_code,
        message=exc.message,
        details=exc.details,
        headers=exc.headers,
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Return framework HTTP errors using the standard response shape."""

    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    details = None if isinstance(exc.detail, str) else exc.detail
    logger.warning(
        "HTTP %s on %s %s: %s",
        exc.status_code,
        request.method,
        request.url.path,
        message,
    )
    return _error_response(
        status_code=exc.status_code,
        code="http_error",
        message=message,
        details=details,
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return request validation failures using the standard response shape."""

    logger.warning(
        "Validation error on %s %s",
        request.method,
        request.url.path,
    )
    return _error_response(
        status_code=422,
        code="validation_error",
        message="Request validation failed.",
        details=exc.errors(),
    )


async def unexpected_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Log unexpected errors and return a safe generic response."""

    logger.error(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_server_error",
        message="An unexpected error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all centralized exception handlers on a FastAPI application."""

    app.add_exception_handler(ApplicationError, application_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)
