"""Application health-check API."""

from typing import Literal

from fastapi import APIRouter, status
from pydantic import BaseModel


router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health-check response returned by the application."""

    status: Literal["healthy"]


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check application health",
)
def health_check() -> HealthResponse:
    """Report that the API process is running and accepting requests."""

    return HealthResponse(status="healthy")
