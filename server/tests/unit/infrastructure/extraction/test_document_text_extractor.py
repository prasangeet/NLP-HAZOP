"""Tests for page-aware PDF and UTF-8 report text extraction."""

from app.infrastructure.extraction import document_text_extractor as extractor


class FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakePdfReader:
    def __init__(self, _stream, strict: bool = True):
        assert strict is True
        self.is_encrypted = False
        self.pages = [
            FakePage("T-301: High level in the storage tank."),
            FakePage(""),
            FakePage("The outlet valve is closed."),
        ]


def test_pdf_text_keeps_page_markers_and_reports_empty_pages(monkeypatch):
    monkeypatch.setattr(extractor, "PdfReader", FakePdfReader)

    result = extractor.extract_document_text("t301.pdf", b"%PDF-test")

    assert result.page_count == 3
    assert result.empty_pages == (2,)
    assert "[Page 1]\nT-301: High level in the storage tank." in result.text
    assert "[Page 3]\nThe outlet valve is closed." in result.text


def test_text_report_is_decoded_without_inventing_page_breaks():
    result = extractor.extract_document_text(
        "report.txt", b"A pump failure causes loss of feed flow."
    )

    assert result.text == "A pump failure causes loss of feed flow."
    assert result.page_count == 1
    assert result.empty_pages == ()
