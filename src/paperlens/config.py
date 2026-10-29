import os
from pathlib import Path

from dotenv import load_dotenv

# Find the project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


def validate_config():
    """Check whether required configuration is present."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    if GROQ_API_KEY == "your_groq_api_key_here":
        raise ValueError(
            "Replace the placeholder in .env with your actual API key."
        )

    if not GROQ_MODEL:
        raise ValueError("GROQ_MODEL is missing.")

    if GROQ_MODEL == "your_available_groq_model_here":
        raise ValueError("Choose an available Groq model in .env.")