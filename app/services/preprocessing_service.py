"""Page-aware text cleanup for extracted enterprise documents."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import spacy
from spacy.language import Language

from app.core.logging import get_logger
from app.services.extraction_service import ExtractedPage, ExtractionResult


logger = get_logger(__name__)

EDGE_LINE_COUNT = 2
REPEATED_LINE_RATIO = 0.6
MAX_REPEATED_LINE_LENGTH = 160

DEHYPHENATION_PATTERN = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")
INLINE_WHITESPACE_PATTERN = re.compile(r"[ \t\v\f]+")
FORMATTING_ARTIFACT_PATTERN = re.compile(r"^\s*[-_=*]{3,}\s*$")
PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$",
    flags=re.IGNORECASE,
)

REMOVABLE_CHARACTERS = str.maketrans(
    {
        "\x00": None,
        "\u00ad": None,
        "\u200b": None,
        "\u200c": None,
        "\u200d": None,
        "\ufeff": None,
        "\ufffd": None,
    }
)


class PreprocessingService:
    """Clean extracted PDF text while preserving page boundaries and metadata."""

    def __init__(self, nlp: Language | None = None) -> None:
        self.nlp = nlp or spacy.blank("en")
        if not any(
            self.nlp.has_pipe(component)
            for component in ("parser", "senter", "sentencizer")
        ):
            self.nlp.add_pipe("sentencizer")

    def preprocess(self, result: ExtractionResult) -> ExtractionResult:
        """Return cleaned page-aware content ready for the LangChain pipeline."""

        prepared_pages = [self._prepare_lines(page.text) for page in result.pages]
        repeated_edge_lines = self._find_repeated_edge_lines(prepared_pages)

        cleaned_pages = tuple(
            ExtractedPage(
                page_number=page.page_number,
                text=self._clean_page(lines, repeated_edge_lines),
                metadata=dict(page.metadata),
            )
            for page, lines in zip(result.pages, prepared_pages, strict=True)
        )

        logger.info(
            "Preprocessed %s pages from %s; removed %s repeated edge lines",
            len(cleaned_pages),
            result.source,
            len(repeated_edge_lines),
        )
        return ExtractionResult(source=result.source, pages=cleaned_pages)

    @staticmethod
    def _prepare_lines(text: str) -> list[str]:
        if not text:
            return []

        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.translate(REMOVABLE_CHARACTERS)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = DEHYPHENATION_PATTERN.sub("", normalized)

        return [
            INLINE_WHITESPACE_PATTERN.sub(" ", line).strip()
            for line in normalized.split("\n")
        ]

    def _clean_page(self, lines: list[str], repeated_lines: set[str]) -> str:
        if not lines:
            return ""

        edge_indexes = self._edge_indexes(lines)
        retained_lines: list[str] = []
        for index, line in enumerate(lines):
            canonical = self._canonical_line(line)
            if FORMATTING_ARTIFACT_PATTERN.fullmatch(line):
                retained_lines.append("")
            elif index in edge_indexes and (
                canonical in repeated_lines or PAGE_NUMBER_PATTERN.fullmatch(line)
            ):
                retained_lines.append("")
            else:
                retained_lines.append(line)

        normalized_lines = self._normalize_lines_with_spacy(retained_lines)
        return self._collapse_blank_lines(normalized_lines)

    def _normalize_lines_with_spacy(self, lines: list[str]) -> list[str]:
        nonempty_lines = [line for line in lines if line]
        documents = iter(self.nlp.pipe(nonempty_lines))
        normalized: list[str] = []

        for line in lines:
            if not line:
                normalized.append("")
                continue

            document = next(documents)
            sentences = [
                " ".join(sentence.text.split())
                for sentence in document.sents
                if sentence.text.strip()
            ]
            normalized.append(" ".join(sentences))

        return normalized

    @classmethod
    def _find_repeated_edge_lines(cls, pages: list[list[str]]) -> set[str]:
        if len(pages) < 2:
            return set()

        occurrences: Counter[str] = Counter()
        for lines in pages:
            page_candidates = {
                cls._canonical_line(lines[index])
                for index in cls._edge_indexes(lines)
                if cls._is_repeat_candidate(lines[index])
            }
            occurrences.update(page_candidates)

        threshold = max(2, math.ceil(len(pages) * REPEATED_LINE_RATIO))
        return {
            line
            for line, count in occurrences.items()
            if count >= threshold
        }

    @staticmethod
    def _edge_indexes(lines: list[str]) -> set[int]:
        nonempty_indexes = [index for index, line in enumerate(lines) if line]
        return set(
            nonempty_indexes[:EDGE_LINE_COUNT]
            + nonempty_indexes[-EDGE_LINE_COUNT:]
        )

    @classmethod
    def _is_repeat_candidate(cls, line: str) -> bool:
        canonical = cls._canonical_line(line)
        return bool(
            canonical
            and len(canonical) <= MAX_REPEATED_LINE_LENGTH
            and not FORMATTING_ARTIFACT_PATTERN.fullmatch(line)
            and not PAGE_NUMBER_PATTERN.fullmatch(line)
        )

    @staticmethod
    def _canonical_line(line: str) -> str:
        return " ".join(line.casefold().split())

    @staticmethod
    def _collapse_blank_lines(lines: list[str]) -> str:
        collapsed: list[str] = []
        for line in lines:
            if not line and (not collapsed or not collapsed[-1]):
                continue
            collapsed.append(line)
        return "\n".join(collapsed).strip()
