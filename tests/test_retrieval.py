"""Tests for reusable LangChain semantic retrieval."""

from pathlib import Path

import pytest
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.exceptions import RetrievalError
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import EmptyRetriever, RetrievalService
from app.services.vector_store_service import VectorStoreService


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings suitable for local retrieval tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.casefold()
        return [
            float(normalized.count("risk")),
            float(normalized.count("finance")),
            float(normalized.count("security")),
            float(len(normalized)) / 100.0,
        ]


def _embedding_service() -> EmbeddingService:
    return EmbeddingService(
        model_name="retrieval-test-model",
        backend=FakeEmbeddings(),
        batch_size=2,
    )


def _documents() -> list[LangChainDocument]:
    return [
        LangChainDocument(
            page_content="Risk controls and audit evidence.",
            metadata={
                "document_id": 1,
                "filename": "risk.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        ),
        LangChainDocument(
            page_content="Finance reporting policy.",
            metadata={
                "document_id": 2,
                "filename": "finance.pdf",
                "page_number": 4,
                "chunk_index": 3,
            },
        ),
        LangChainDocument(
            page_content="Security incident response.",
            metadata={
                "document_id": 3,
                "filename": "security.pdf",
                "page_number": 2,
                "chunk_index": 1,
            },
        ),
    ]


def _persisted_service(path: Path) -> VectorStoreService:
    service = VectorStoreService(_embedding_service(), path)
    service.create_vector_store(_documents())
    return service


def test_retrieve_loads_index_and_includes_similarity_metadata(
    tmp_path: Path,
) -> None:
    _persisted_service(tmp_path)
    vector_service = VectorStoreService(_embedding_service(), tmp_path)
    retrieval = RetrievalService(vector_service, top_k=2)

    documents = retrieval.retrieve("risk audit")

    assert len(documents) == 2
    assert documents[0].metadata["document_id"] == 1
    assert documents[0].metadata["filename"] == "risk.pdf"
    assert isinstance(documents[0].metadata["distance"], float)
    assert 0.0 < documents[0].metadata["similarity_score"] <= 1.0


def test_retrieve_supports_per_query_top_k(tmp_path: Path) -> None:
    vector_service = _persisted_service(tmp_path)
    retrieval = RetrievalService(vector_service, top_k=1)

    assert len(retrieval.retrieve("policy")) == 1
    assert len(retrieval.retrieve("policy", top_k=2)) == 2
    assert len(retrieval.retrieve("policy", top_k=20)) == 3


def test_as_retriever_returns_langchain_interface(tmp_path: Path) -> None:
    vector_service = _persisted_service(tmp_path)

    retriever = RetrievalService(vector_service).as_retriever(top_k=2)
    documents = retriever.invoke("security response")

    assert isinstance(retriever, VectorStoreRetriever)
    assert len(documents) == 2
    assert all(isinstance(document, LangChainDocument) for document in documents)


def test_empty_vector_store_returns_no_results(tmp_path: Path) -> None:
    vector_service = VectorStoreService(_embedding_service(), tmp_path)
    retrieval = RetrievalService(vector_service)

    assert retrieval.retrieve("anything") == []
    retriever = retrieval.as_retriever()
    assert isinstance(retriever, EmptyRetriever)
    assert retriever.invoke("anything") == []


def test_incomplete_vector_store_raises_controlled_error(tmp_path: Path) -> None:
    (tmp_path / "index.faiss").write_bytes(b"incomplete")
    retrieval = RetrievalService(
        VectorStoreService(_embedding_service(), tmp_path)
    )

    with pytest.raises(RetrievalError, match="could not be loaded"):
        retrieval.retrieve("risk")


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(tmp_path: Path, top_k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RetrievalService(
            VectorStoreService(_embedding_service(), tmp_path),
            top_k=top_k,  # type: ignore[arg-type]
        )


def test_blank_query_is_rejected(tmp_path: Path) -> None:
    retrieval = RetrievalService(
        VectorStoreService(_embedding_service(), tmp_path)
    )

    with pytest.raises(ValueError, match="non-whitespace"):
        retrieval.retrieve("   ")
