"""
config.py
---------
Central place for all app-wide constants and configuration values.
Keeping these in one file means you can tune chunking/retrieval behavior
without hunting through ingest.py / query.py.
"""

import os
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")
COLLECTION_NAME = "document_knowledge_base"

# ---------------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------
TOP_K = 4               # number of chunks to retrieve per query


def validate_config():
    """Raise a clear error early if required configuration is missing."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "Gemini API key (get one free at https://aistudio.google.com/app/apikey)."
        )
