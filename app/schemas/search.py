"""Pydantic schemas for semantic document search."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings


settings = get_settings()

MAX_QUERY_LENGTH = 4000
MAX_TOP_K = 100


class SearchRequest(BaseModel):
    """Validated semantic search parameters."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(
        default=settings.RETRIEVAL_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        strict=True,
    )

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value


class SearchResult(BaseModel):
    """One relevant document chunk and its source information."""

    chunk_content: str
    filename: str
    page_number: int = Field(ge=1)
    document_id: int = Field(ge=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
    distance: float = Field(ge=0.0)


class SearchResponse(BaseModel):
    """Semantic search results returned to an API client."""

    query: str
    top_k: int = Field(ge=1, le=MAX_TOP_K)
    total: int = Field(ge=0)
    results: list[SearchResult]
