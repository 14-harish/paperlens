
from langchain_core.documents import Document

from src.paperlens.ingestion import (
    load_pdf_documents,
    split_documents,
)


def test_missing_pdf_raises_error():
    """A missing PDF should raise FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        load_pdf_documents("does_not_exist.pdf")


def test_non_pdf_file_raises_error(tmp_path):
    """A non-PDF file should be rejected."""
    import pytest

    text_file = tmp_path / "notes.txt"
    text_file.write_text("Sample text")

    with pytest.raises(ValueError, match="Only PDF"):
        load_pdf_documents(text_file)


def test_pdf_extraction_preserves_page_numbers(tmp_path):
    """Extracted pages should have correct page metadata."""
    import pymupdf

    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page1 = pdf.new_page()
        page1.insert_text(
            (72, 72),
            "First page of a research paper.",
        )

        page2 = pdf.new_page()
        page2.insert_text(
            (72, 72),
            "Second page of a research paper.",
        )

        pdf.save(pdf_path)

    documents = load_pdf_documents(pdf_path)

    assert len(documents) == 2
    assert documents[0].metadata["page"] == 1
    assert documents[1].metadata["page"] == 2
    assert documents[0].metadata["source"] == "sample.pdf"


def test_chunking_preserves_metadata():
    """Splitting should create chunks and retain metadata."""
    document = Document(
        page_content="Machine learning research. " * 200,
        metadata={
            "source": "paper_a.pdf",
            "paper_title": "paper_a",
            "page": 4,
        },
    )

    chunks = split_documents([document])

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["source"] == "paper_a.pdf"
        assert chunk.metadata["page"] == 4
        assert len(chunk.page_content) <= 1000
