"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.database import engine
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger


settings = get_settings()
configure_logging(logging.DEBUG if settings.DEBUG else logging.INFO)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage resources shared across the application lifecycle."""

    logger.info("Starting %s", settings.APP_NAME)
    try:
        yield
    finally:
        engine.dispose()
        logger.info("Stopped %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "API for enterprise document intelligence and retrieval-augmented "
        "generation."
    ),
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

register_exception_handlers(app)
app.include_router(health_router)
app.include_router(documents_router)
