"""Tests for page-aware spaCy text preprocessing."""

from app.services.extraction_service import ExtractedPage, ExtractionResult
from app.services.preprocessing_service import PreprocessingService


def _result(*page_texts: str) -> ExtractionResult:
    total_pages = len(page_texts)
    return ExtractionResult(
        source="data/uploads/report.pdf",
        pages=tuple(
            ExtractedPage(
                page_number=index + 1,
                text=text,
                metadata={
                    "source": "data/uploads/report.pdf",
                    "page": index,
                    "page_number": index + 1,
                    "total_pages": total_pages,
                },
            )
            for index, text in enumerate(page_texts)
        ),
    )


def test_preprocess_normalizes_artifacts_and_preserves_content() -> None:
    result = _result(
        "Revenue\u00ad   Report\n\nThe enter-\nprise earned   $1,250,000.\n-----\n• Keep policy ID A-102."
    )

    cleaned = PreprocessingService().preprocess(result)

    assert cleaned.total_pages == 1
    assert cleaned.pages[0].page_number == 1
    assert cleaned.pages[0].metadata == result.pages[0].metadata
    assert "Revenue Report" in cleaned.pages[0].text
    assert "enterprise earned $1,250,000." in cleaned.pages[0].text
    assert "Keep policy ID A-102." in cleaned.pages[0].text
    assert "-----" not in cleaned.pages[0].text
    assert "  " not in cleaned.pages[0].text


def test_preprocess_removes_repeated_headers_footers_and_page_numbers() -> None:
    result = _result(
        "ACME CONFIDENTIAL\nQuarterly Operations\nPage one content.\nPage 1 of 3",
        "ACME CONFIDENTIAL\nQuarterly Operations\nPage two content.\nPage 2 of 3",
        "ACME CONFIDENTIAL\nQuarterly Operations\nPage three content.\nPage 3 of 3",
    )

    cleaned = PreprocessingService().preprocess(result)

    assert [page.page_number for page in cleaned.pages] == [1, 2, 3]
    assert all("ACME CONFIDENTIAL" not in page.text for page in cleaned.pages)
    assert all("Quarterly Operations" not in page.text for page in cleaned.pages)
    assert all("Page 1 of 3" not in page.text for page in cleaned.pages)
    assert "Page one content." in cleaned.pages[0].text
    assert "Page two content." in cleaned.pages[1].text
    assert "Page three content." in cleaned.pages[2].text


def test_preprocess_keeps_nonrepeated_enterprise_headings() -> None:
    result = _result(
        "Risk Assessment\nMaterial risk details remain.",
        "Financial Controls\nControl evidence remains.",
    )

    cleaned = PreprocessingService().preprocess(result)

    assert "Risk Assessment" in cleaned.pages[0].text
    assert "Material risk details remain." in cleaned.pages[0].text
    assert "Financial Controls" in cleaned.pages[1].text
    assert "Control evidence remains." in cleaned.pages[1].text


def test_preprocess_handles_empty_pages_safely() -> None:
    result = _result("", " \n\t\n")

    cleaned = PreprocessingService().preprocess(result)

    assert cleaned.total_pages == 2
    assert [page.text for page in cleaned.pages] == ["", ""]
    assert cleaned.full_text == "\n\n"


def test_preprocess_handles_empty_document_safely() -> None:
    result = ExtractionResult(source="empty.pdf", pages=())

    cleaned = PreprocessingService().preprocess(result)

    assert cleaned.source == "empty.pdf"
    assert cleaned.pages == ()
    assert cleaned.full_text == ""
