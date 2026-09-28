import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# --- Rutas ---
PDFS_DIR = BASE_DIR / "pdfs"
CHROMA_DIR = BASE_DIR / "chroma"
COLLECTION_NAME = "mis_programas"

# --- Embeddings (Modelo ligero accesible vía API) ---
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Chunking & Retrieval ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_K = 5