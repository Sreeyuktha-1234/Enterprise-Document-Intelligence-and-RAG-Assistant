"""Persistent LangChain FAISS vector-store operations."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embedding_service import EmbeddingService


logger = get_logger(__name__)
settings = get_settings()

INDEX_NAME = "index"
INDEX_FILENAME = f"{INDEX_NAME}.faiss"
DOCSTORE_FILENAME = f"{INDEX_NAME}.pkl"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1
HASH_CHUNK_SIZE = 1024 * 1024


class VectorStoreService:
    """Create, persist, load, and extend a local LangChain FAISS index."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store_path: str | Path | None = None,
    ) -> None:
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )
        self.vector_store_path = Path(
            vector_store_path or settings.VECTOR_STORE_PATH
        )
        self.vector_store: FAISS | None = None

    def create_vector_store(
        self,
        documents: Sequence[LangChainDocument],
        *,
        persist: bool = True,
    ) -> FAISS:
        """Create a new FAISS index from LangChain documents."""

        self._validate_documents(documents)
        document_ids = [uuid4().hex for _ in documents]
        self.vector_store = self._create_store(documents, document_ids)

        if persist:
            self.save_vector_store()

        logger.info("Created FAISS index with %s documents", len(documents))
        return self.vector_store

    def load_vector_store(self) -> FAISS:
        """Load an application-created index after validating local artifacts."""

        self._verify_persisted_index()
        try:
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path),
                self.embedding_service.backend,
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.exception("Failed to load FAISS index from %s", self.vector_store_path)
            raise ValueError("The persisted FAISS index could not be loaded.") from exc

        logger.info("Loaded FAISS index from %s", self.vector_store_path)
        return self.vector_store

    def add_documents(
        self,
        documents: Sequence[LangChainDocument],
        *,
        persist: bool = True,
    ) -> list[str]:
        """Embed and add documents to the active or persisted FAISS index."""

        self._validate_documents(documents)
        document_ids = [uuid4().hex for _ in documents]

        if self.vector_store is None:
            if self._persisted_artifacts_exist():
                self.load_vector_store()
            else:
                self.vector_store = self._create_store(documents, document_ids)
                if persist:
                    self.save_vector_store()
                return document_ids

        vectors = self.embedding_service.embed_documents(documents)
        text_embeddings = list(
            zip(
                (document.page_content for document in documents),
                vectors,
                strict=True,
            )
        )
        self.vector_store.add_embeddings(
            text_embeddings,
            metadatas=[dict(document.metadata) for document in documents],
            ids=document_ids,
        )

        if persist:
            self.save_vector_store()

        logger.info("Added %s documents to the FAISS index", len(documents))
        return document_ids

    def save_vector_store(self) -> None:
        """Persist the active index and write its integrity manifest last."""

        if self.vector_store is None:
            raise ValueError("No FAISS index is available to persist.")
        if self.vector_store_path.is_symlink():
            raise ValueError("The vector-store directory cannot be a symbolic link.")

        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(
            str(self.vector_store_path),
            index_name=INDEX_NAME,
        )
        self._write_manifest()
        logger.info("Persisted FAISS index to %s", self.vector_store_path)

    def _create_store(
        self,
        documents: Sequence[LangChainDocument],
        document_ids: list[str],
    ) -> FAISS:
        vectors = self.embedding_service.embed_documents(documents)
        text_embeddings = list(
            zip(
                (document.page_content for document in documents),
                vectors,
                strict=True,
            )
        )
        return FAISS.from_embeddings(
            text_embeddings,
            self.embedding_service.backend,
            metadatas=[dict(document.metadata) for document in documents],
            ids=document_ids,
        )

    @staticmethod
    def _validate_documents(documents: Sequence[LangChainDocument]) -> None:
        if not documents:
            raise ValueError("At least one document is required for the FAISS index.")
        if any(not document.page_content.strip() for document in documents):
            raise ValueError("FAISS documents must contain non-whitespace text.")

    def _persisted_artifacts_exist(self) -> bool:
        return any(path.exists() for path in self._artifact_paths().values())

    def _verify_persisted_index(self) -> None:
        if self.vector_store_path.is_symlink():
            raise ValueError("The vector-store directory cannot be a symbolic link.")

        artifacts = self._artifact_paths()
        missing = [name for name, path in artifacts.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The persisted FAISS index is incomplete: " + ", ".join(missing)
            )

        root = self.vector_store_path.resolve()
        for path in artifacts.values():
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("FAISS index artifacts must be regular local files.")

        try:
            manifest: dict[str, Any] = json.loads(
                artifacts[MANIFEST_FILENAME].read_text(encoding="utf-8")
            )
            if manifest["version"] != MANIFEST_VERSION:
                raise ValueError("Unsupported FAISS manifest version.")
            if manifest["embedding_model"] != self.embedding_service.model_name:
                raise ValueError(
                    "The FAISS index was created with a different embedding model."
                )

            expected_hashes = manifest["sha256"]
            for filename in (INDEX_FILENAME, DOCSTORE_FILENAME):
                actual_hash = self._sha256(artifacts[filename])
                expected_hash = expected_hashes[filename]
                if not hmac.compare_digest(actual_hash, expected_hash):
                    raise ValueError(
                        f"FAISS index integrity verification failed for {filename}."
                    )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("The FAISS integrity manifest is invalid.") from exc

    def _write_manifest(self) -> None:
        artifacts = self._artifact_paths()
        manifest = {
            "version": MANIFEST_VERSION,
            "embedding_model": self.embedding_service.model_name,
            "sha256": {
                filename: self._sha256(artifacts[filename])
                for filename in (INDEX_FILENAME, DOCSTORE_FILENAME)
            },
        }
        manifest_path = artifacts[MANIFEST_FILENAME]
        temporary_path = manifest_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)

    def _artifact_paths(self) -> dict[str, Path]:
        return {
            INDEX_FILENAME: self.vector_store_path / INDEX_FILENAME,
            DOCSTORE_FILENAME: self.vector_store_path / DOCSTORE_FILENAME,
            MANIFEST_FILENAME: self.vector_store_path / MANIFEST_FILENAME,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            while chunk := artifact.read(HASH_CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()
