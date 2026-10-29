from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.paperlens.config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    TOP_K,
)
from src.paperlens.ingestion import (
    load_pdf_documents,
    split_documents,
)


# Project paths
PAPERS_DIR = BASE_DIR / "data" / "sample_papers"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# Vector database collection
COLLECTION_NAME = "paperlens_papers"

def build_vector_store():
    """Load papers, create embeddings, and persist them in Chroma."""

    # Find all PDFs in the papers directory
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PAPERS_DIR}"
        )

    # Extract pages and split them into chunks
    all_chunks = []

    for pdf_path in pdf_files:
        pages = load_pdf_documents(pdf_path)
        chunks = split_documents(pages)
        all_chunks.extend(chunks)

        print(f"{pdf_path.name}: {len(pages)} pages, "
              f"{len(chunks)} chunks")

    if not all_chunks:
        raise ValueError("No text chunks were extracted from the PDFs.")

    # Load the local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Create and persist the Chroma vector store
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"\nTotal chunks indexed: {len(all_chunks)}")
    print(f"Vector store saved to: {CHROMA_DIR}")

    return vector_store

def search_papers(question: str, k: int = TOP_K):
    """Retrieve the most relevant paper chunks for a question."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    results = vector_store.max_marginal_relevance_search(
    question,
    k=k,
    fetch_k=40,
    lambda_mult=0.7,
)

    return results


def search_all_papers(question: str, k_per_paper: int = 2):
    """Retrieve relevant chunks separately from each research paper."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if k_per_paper < 1:
        raise ValueError("k_per_paper must be at least 1.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    all_results = []

    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        results = vector_store.similarity_search(
            question,
            k=k_per_paper,
            filter={"source": pdf_path.name},
        )

        all_results.extend(results)

    return all_results
