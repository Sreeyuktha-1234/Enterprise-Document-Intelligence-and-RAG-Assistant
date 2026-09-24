"""Tests for the persistent LangChain FAISS vector store."""

from pathlib import Path

import pytest
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings

from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


class FakeEmbeddings(Embeddings):
    """Small deterministic embedding model with a stable vector dimension."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float(len(lowered)),
            float(lowered.count("risk")),
            float(lowered.count("finance")),
            float(sum(ord(character) for character in lowered) % 997),
        ]


def _embedding_service(model_name: str = "test-embeddings") -> EmbeddingService:
    return EmbeddingService(
        model_name=model_name,
        backend=FakeEmbeddings(),
        batch_size=2,
    )


def _documents() -> list[LangChainDocument]:
    return [
        LangChainDocument(
            page_content="Enterprise risk controls and audit evidence.",
            metadata={
                "document_id": 1,
                "filename": "risk.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "source": "data/uploads/risk.pdf",
            },
        ),
        LangChainDocument(
            page_content="Finance policy and quarterly reporting controls.",
            metadata={
                "document_id": 2,
                "filename": "finance.pdf",
                "page_number": 3,
                "chunk_index": 4,
                "source": "data/uploads/finance.pdf",
            },
        ),
    ]


def test_create_and_persist_vector_store_preserves_metadata(tmp_path: Path) -> None:
    service = VectorStoreService(_embedding_service(), tmp_path)

    store = service.create_vector_store(_documents())

    assert store.index.ntotal == 2
    assert (tmp_path / "index.faiss").is_file()
    assert (tmp_path / "index.pkl").is_file()
    assert (tmp_path / "manifest.json").is_file()

    results = store.similarity_search("risk controls", k=2)
    assert {result.metadata["filename"] for result in results} == {
        "risk.pdf",
        "finance.pdf",
    }
    assert {result.metadata["source"] for result in results} == {
        "data/uploads/risk.pdf",
        "data/uploads/finance.pdf",
    }


def test_load_existing_index_after_integrity_validation(tmp_path: Path) -> None:
    creator = VectorStoreService(_embedding_service(), tmp_path)
    creator.create_vector_store(_documents())
    loader = VectorStoreService(_embedding_service(), tmp_path)

    loaded = loader.load_vector_store()

    assert loaded.index.ntotal == 2
    assert loaded.similarity_search("finance", k=1)[0].metadata["document_id"] in {
        1,
        2,
    }


def test_add_documents_persists_and_can_be_reloaded(tmp_path: Path) -> None:
    service = VectorStoreService(_embedding_service(), tmp_path)
    service.create_vector_store([_documents()[0]])
    additional = _documents()[1]

    ids = service.add_documents([additional])
    reloaded = VectorStoreService(_embedding_service(), tmp_path).load_vector_store()

    assert len(ids) == 1
    assert reloaded.index.ntotal == 2
    results = reloaded.similarity_search("finance policy", k=2)
    assert any(result.metadata == additional.metadata for result in results)


def test_add_documents_creates_index_when_none_exists(tmp_path: Path) -> None:
    service = VectorStoreService(_embedding_service(), tmp_path)

    ids = service.add_documents(_documents())

    assert len(ids) == 2
    assert service.vector_store is not None
    assert service.vector_store.index.ntotal == 2
    assert (tmp_path / "manifest.json").is_file()


def test_load_rejects_tampered_docstore_before_deserialization(tmp_path: Path) -> None:
    service = VectorStoreService(_embedding_service(), tmp_path)
    service.create_vector_store(_documents())
    with (tmp_path / "index.pkl").open("ab") as docstore:
        docstore.write(b"tampered")

    with pytest.raises(ValueError, match="integrity verification failed"):
        VectorStoreService(_embedding_service(), tmp_path).load_vector_store()


def test_load_rejects_different_embedding_model(tmp_path: Path) -> None:
    VectorStoreService(_embedding_service("model-a"), tmp_path).create_vector_store(
        _documents()
    )

    with pytest.raises(ValueError, match="different embedding model"):
        VectorStoreService(
            _embedding_service("model-b"),
            tmp_path,
        ).load_vector_store()


def test_create_rejects_empty_or_blank_documents(tmp_path: Path) -> None:
    service = VectorStoreService(_embedding_service(), tmp_path)

    with pytest.raises(ValueError, match="At least one document"):
        service.create_vector_store([])
    with pytest.raises(ValueError, match="non-whitespace"):
        service.create_vector_store([LangChainDocument(page_content="   ")])
