import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (en la raíz del proyecto)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- API Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Rutas ---
PDFS_DIR = BASE_DIR / "pdfs"
CHROMA_DIR = BASE_DIR / "chroma"
COLLECTION_NAME = "mis_programas"

# --- Embeddings ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
DEFAULT_K = 10
