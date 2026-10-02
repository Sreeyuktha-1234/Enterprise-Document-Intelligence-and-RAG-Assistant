"""Pydantic schemas for stateless retrieval-augmented chat."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_QUESTION_LENGTH = 4000


class ChatRequest(BaseModel):
    """One user question for the RAG pipeline."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        strict=True,
    )

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must contain non-whitespace text")
        return value


class ChatSource(BaseModel):
    """A source chunk supporting the generated answer."""

    citation: str = Field(pattern=r"^\[Source [1-9]\d*\]$")
    document_id: int = Field(ge=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    excerpt: str = Field(min_length=1)
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    """A grounded answer and its citation-addressable evidence."""

    answer: str = Field(min_length=1)
    sources: list[ChatSource]
