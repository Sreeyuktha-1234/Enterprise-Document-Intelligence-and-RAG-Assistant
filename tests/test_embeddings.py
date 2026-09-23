"""Tests for reusable Hugging Face LangChain embeddings."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings

from app.services.embedding_service import (
    EmbeddingService,
    get_embedding_backend,
    settings,
)


class FakeEmbeddings(Embeddings):
    """Deterministic in-memory embedding backend for unit tests."""

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(texts)
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text)), 1.0]


def test_embed_langchain_documents_in_batches_without_mutating_metadata() -> None:
    backend = FakeEmbeddings()
    documents = [
        LangChainDocument(
            page_content=f"Document content {index}",
            metadata={"chunk_index": index},
        )
        for index in range(5)
    ]
    service = EmbeddingService(backend=backend, batch_size=2)

    vectors = service.embed_documents(documents)

    assert len(vectors) == 5
    assert [len(batch) for batch in backend.document_batches] == [2, 2, 1]
    assert [document.metadata["chunk_index"] for document in documents] == list(
        range(5)
    )


def test_embed_texts_preserves_input_order() -> None:
    backend = FakeEmbeddings()
    service = EmbeddingService(backend=backend, batch_size=2)

    vectors = service.embed_texts(["a", "bb", "ccc"])

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]


def test_embed_query_uses_backend() -> None:
    backend = FakeEmbeddings()
    service = EmbeddingService(backend=backend)

    vector = service.embed_query("enterprise policy")

    assert vector == [17.0, 1.0]
    assert backend.queries == ["enterprise policy"]


def test_empty_inputs_are_handled_safely() -> None:
    backend = FakeEmbeddings()
    service = EmbeddingService(backend=backend)

    assert service.embed_documents([]) == []
    assert service.embed_texts([]) == []
    with pytest.raises(ValueError, match="non-whitespace"):
        service.embed_query("   ")


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        EmbeddingService(backend=FakeEmbeddings(), batch_size=0)


def test_unexpected_backend_vector_count_is_rejected() -> None:
    backend = Mock(spec=Embeddings)
    backend.embed_documents.return_value = []
    service = EmbeddingService(backend=backend)

    with pytest.raises(RuntimeError, match="unexpected vector count"):
        service.embed_texts(["content"])


def test_hugging_face_model_is_loaded_once_per_configuration() -> None:
    backend = FakeEmbeddings()
    get_embedding_backend.cache_clear()

    with patch(
        "app.services.embedding_service.HuggingFaceEmbeddings",
        return_value=backend,
    ) as constructor:
        first = EmbeddingService()
        second = EmbeddingService()

    assert first.backend is backend
    assert second.backend is backend
    constructor.assert_called_once_with(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )
    get_embedding_backend.cache_clear()
