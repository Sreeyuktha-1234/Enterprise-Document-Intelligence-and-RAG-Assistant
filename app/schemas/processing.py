"""Schemas returned by the document processing API."""

from pydantic import BaseModel, Field


class ProcessingStatistics(BaseModel):
    """Summary of work completed by the ingestion pipeline."""

    pages_extracted: int = Field(ge=0)
    pages_processed: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    characters_extracted: int = Field(ge=0)
    characters_processed: int = Field(ge=0)


class ProcessingResponse(BaseModel):
    """Result returned after successfully processing one document."""

    document_id: int
    status: str
    statistics: ProcessingStatistics
