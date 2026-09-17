"""Pydantic request and response schemas for documents."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentBase(BaseModel):
    """Fields shared by document request schemas."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    original_filename: str = Field(min_length=1, max_length=255)
    file_path: str = Field(min_length=1, max_length=1024)
    file_size: int = Field(ge=0)
    content_type: str = Field(min_length=1, max_length=255)


class DocumentCreate(DocumentBase):
    """Data required to create a document metadata record."""

    status: str = Field(default="uploaded", min_length=1, max_length=50)


class DocumentUpdate(BaseModel):
    """Fields that may be changed on an existing document record."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filename: str | None = Field(default=None, min_length=1, max_length=255)
    original_filename: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )
    file_path: str | None = Field(default=None, min_length=1, max_length=1024)
    file_size: int | None = Field(default=None, ge=0)
    content_type: str | None = Field(default=None, min_length=1, max_length=255)
    status: str | None = Field(default=None, min_length=1, max_length=50)


class DocumentResponse(DocumentBase):
    """Document metadata returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated collection of document metadata."""

    items: list[DocumentResponse]
    total: int = Field(ge=0)
    skip: int = Field(ge=0)
    limit: int = Field(ge=1)
