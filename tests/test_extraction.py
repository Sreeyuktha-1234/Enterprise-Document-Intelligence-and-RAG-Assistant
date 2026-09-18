"""Tests for page-aware PDF text extraction."""

import logging
from pathlib import Path

import pymupdf
import pytest

from app.core.exceptions import DocumentProcessingError
from app.services.extraction_service import ExtractionService


def _create_pdf(path: Path, page_texts: list[str]) -> None:
    with pymupdf.open() as pdf:
        for text in page_texts:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text)
        pdf.save(path)


def test_extract_pdf_preserves_page_text_and_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "report.pdf"
    _create_pdf(pdf_path, ["First page", "Second page", ""])

    result = ExtractionService().extract_pdf(pdf_path)

    assert result.source == pdf_path.as_posix()
    assert result.total_pages == 3
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert "First page" in result.pages[0].text
    assert "Second page" in result.pages[1].text
    assert result.pages[2].text == ""
    assert result.pages[0].metadata == {
        "source": pdf_path.as_posix(),
        "page": 0,
        "page_number": 1,
        "total_pages": 3,
    }
    assert "First page" in result.full_text
    assert "Second page" in result.full_text


def test_extraction_result_is_serializable(tmp_path: Path) -> None:
    pdf_path = tmp_path / "report.pdf"
    _create_pdf(pdf_path, ["Page content"])

    payload = ExtractionService().extract_pdf(pdf_path).to_dict()

    assert payload["source"] == pdf_path.as_posix()
    assert payload["total_pages"] == 1
    assert payload["pages"][0]["page_number"] == 1
    assert "Page content" in payload["pages"][0]["text"]


@pytest.mark.parametrize("filename", ["missing.pdf", "corrupted.pdf"])
def test_extract_pdf_handles_unreadable_files(
    tmp_path: Path,
    filename: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pdf_path = tmp_path / filename
    if filename == "corrupted.pdf":
        pdf_path.write_bytes(b"this is not a PDF")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            DocumentProcessingError,
            match="corrupted, unreadable, or unavailable",
        ):
            ExtractionService().extract_pdf(pdf_path)

    assert "PDF extraction failed" in caplog.text
    assert str(pdf_path) in caplog.text


def test_extract_pdf_rejects_password_protected_files(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_path = tmp_path / "source.pdf"
    encrypted_path = tmp_path / "encrypted.pdf"
    _create_pdf(source_path, ["Protected content"])

    with pymupdf.open(source_path) as pdf:
        pdf.save(
            encrypted_path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            DocumentProcessingError,
            match="Password-protected PDFs",
        ):
            ExtractionService().extract_pdf(encrypted_path)

    assert "PDF extraction failed" in caplog.text
