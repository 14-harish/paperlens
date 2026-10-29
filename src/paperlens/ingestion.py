
from pathlib import Path
import re
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.paperlens.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf_documents(pdf_path: str | Path) -> list[Document]:
    """Extract text from a PDF, preserving paper and page metadata."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    documents = []

    stop_reading = False

    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text")

            lines = text.splitlines()
            kept_lines = []

            for line in lines:
                heading = line.strip().lower()

                if re.fullmatch(r"(references|bibliography|reference list)", heading):
                    stop_reading = True
                    break

                kept_lines.append(line)

            text = "\n".join(kept_lines).strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "paper_title": pdf_path.stem,
                            "page": page_index + 1,
                        },
                    )
                )

            if stop_reading:
                break

    if not documents:
        raise ValueError(
            f"No extractable text found in {pdf_path.name}. "
            "The PDF may be scanned or image-only."
        )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split page documents into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    return splitter.split_documents(documents)
