"""Extract page-aware text from supported HAZOP report documents.

PDF extraction is text-layer based. Scanned/image-only pages require OCR and are
reported as empty pages rather than silently treated as successfully extracted.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class ExtractedDocumentText:
    text: str
    page_count: int
    empty_pages: tuple[int, ...]


def extract_document_text(filename: str, content: bytes) -> ExtractedDocumentText:
    """Decode a text document or extract text from each PDF page."""
    suffix = Path(filename).suffix.casefold()
    if suffix in {".txt", ".md", ".log"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Report must be UTF-8 text.") from exc
        if not text.strip():
            raise ValueError("The supplied report is empty.")
        return ExtractedDocumentText(text=text, page_count=1, empty_pages=())

    if suffix != ".pdf":
        raise ValueError("Supported report formats are PDF, TXT, MD and LOG.")

    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise ValueError("Password-protected PDF reports are not supported.")
        page_texts: list[str] = []
        empty_pages: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            extracted = (page.extract_text() or "").strip()
            if not extracted:
                empty_pages.append(page_number)
                continue
            page_texts.append(f"[Page {page_number}]\n{extracted}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF report: {exc}") from exc

    if not page_texts:
        raise ValueError(
            "No selectable text was found in the PDF. Scanned/image-only PDFs "
            "need OCR before they can be processed."
        )

    return ExtractedDocumentText(
        text="\n\n".join(page_texts),
        page_count=len(reader.pages),
        empty_pages=tuple(empty_pages),
    )
