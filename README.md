# PaperLens — Research Paper Question Answering with RAG

PaperLens is a local Retrieval-Augmented Generation (RAG) application that answers questions across five diabetic retinopathy research papers. It retrieves relevant passages from the papers, generates evidence-based answers using an LLM, and provides page-level citations to help users verify the information.

The project demonstrates how document ingestion, text chunking, embedding generation, vector search, and LLM-based answer generation work together in a RAG pipeline.

## Features

- **PDF ingestion:** Extracts text from research papers using PyMuPDF.
- **Text chunking:** Splits extracted text into overlapping chunks for retrieval.
- **Local embeddings:** Generates text embeddings using Sentence Transformers.
- **Vector database:** Stores and retrieves document chunks using ChromaDB.
- **Research paper comparison:** Retrieves relevant passages from individual papers to support cross-paper comparisons.
- **LLM-powered answers:** Uses Groq for answer generation based on retrieved evidence.
- **Source citations:** Includes paper filenames and page numbers in answers.
- **Evidence-aware responses:** Instructs the model to acknowledge when the retrieved evidence does not answer a question.
- **Interactive interface:** Provides a Streamlit interface for asking questions and viewing answers.

## Architecture

```text
Research Paper PDFs
        |
        v
   PDF Extraction
      (PyMuPDF)
        |
        v
    Text Chunking
   (overlapping chunks)
        |
        v
 Local Embedding Model
 (Sentence Transformers)
        |
        v
     ChromaDB
   Vector Storage
        |
        v
  Relevant Chunk Retrieval
        |
        v
   Groq LLM
 (Answer Generation)
        |
        v
 Answer + Page Citations
        |
        v
  Streamlit Interface
```

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| PDF processing | PyMuPDF |
| Text splitting | LangChain Text Splitters |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database | ChromaDB |
| LLM integration | LangChain Groq |
| LLM provider | Groq |
| Testing | pytest |

## Project Structure

```text
paperlens/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   ├── sample_papers/   # Local PDF files
│   └── chroma/          # Generated vector index
├── src/
│   └── paperlens/
│       ├── __init__.py
│       ├── config.py
│       ├── ingestion.py
│       ├── retrieval.py
│       └── comparison.py
└── tests/
    ├── test_ingestion.py
    └── test_retrieval.py
```

The PDF files and generated ChromaDB index are kept locally and are not required to be committed to the Git repository. The index can be generated from the source PDFs.

## Setup and Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd paperlens
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Add your own Groq API key to `.env` and configure the model name supported by your Groq account.

Example `.env.example`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The model name is configurable. If it is unavailable to your account, replace it with a model currently supported by Groq.

**Security:** Never commit your real `.env` file or API key to GitHub.

### 5. Add the research papers

Place the five research paper PDFs in:

```text
data/sample_papers/
```

Use the filenames expected by the project, or ensure the ingestion code supports your chosen filenames.

The papers should be available locally before building the vector index.

### 6. Build the vector index

Run the project's vector-store creation function using the configured Python environment. For example, if you add a small indexing entry point, run:

```bash
python build_index.py
```

The `build_index.py` command is an example entry point; use it only if that script is included in the repository. Otherwise, invoke the existing `build_vector_store()` function from `src/paperlens/retrieval.py` using your project's Python environment.

### 7. Run PaperLens

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open it in your browser to ask questions about the research papers.

## Example Questions

- What deep learning models are used for diabetic retinopathy detection?
- Compare the approaches used in the five research papers.
- What performance metrics are reported by a particular paper?
- What preprocessing techniques are discussed?
- Does the supplied evidence report training time on an NVIDIA A100 GPU?

PaperLens is designed to ground its responses in retrieved passages rather than inventing information when evidence is unavailable.

## Testing

Run the test suite with:

```bash
python -m pytest -q
```

Using `python -m pytest` ensures pytest runs with the active Python environment.

## Design Considerations

- **Local versus hosted computation:** PDF processing, embeddings, and vector retrieval run locally. Groq is used for LLM-based answer generation.
- **Chunking:** Overlapping chunks help preserve context across text boundaries.
- **Per-paper retrieval:** Retrieving passages separately from each paper helps support comparisons across the research collection.
- **Source grounding:** Paper filenames and page metadata are included in the retrieved context to support citations.
- **Evidence limitations:** Answers depend on the quality of PDF extraction and the relevance of retrieved chunks. Citations and generated answers should be checked against the original papers.

## Current Scope and Limitations

PaperLens is a local portfolio prototype built to explore and demonstrate a RAG workflow.

- It currently targets a collection of five research papers.
- The PDF files must be supplied locally.
- LLM answer generation requires a valid Groq API key and an available model.
- Citation accuracy depends on retrieval quality and the source metadata.
- The application has not been presented as a production deployment or a clinically validated medical system.

## Potential Improvements

- Add retrieval and answer-quality evaluation.
- Improve citation verification and source display.
- Add configurable retrieval settings.
- Expand the collection to support more research papers.
- Add automated tests for cross-paper retrieval and answer grounding.

## Disclaimer

PaperLens is an educational project for exploring research literature and RAG systems. It is not intended to provide medical advice or support clinical diagnosis.
