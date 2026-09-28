from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_K,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    PDFS_DIR,
)

# ---------------------------------------------------------------------------
# Prompt del sistema
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Eres un asistente legal experto en reglamentos y normas.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado.
Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página
del fragmento de donde proviene la información, por ejemplo:
(Fuente: NombreDocumento.pdf, Pág. X).
Si la respuesta no está en el contexto, indica exactamente:
"No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
# Estado global (se carga una sola vez al arrancar)
# ---------------------------------------------------------------------------
_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: Chroma | None = None
_llm: ChatGroq | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"[RAG] Cargando modelo de embeddings: {EMBEDDING_MODEL} ...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[RAG] Embeddings listos.")
    return _embeddings


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_PEGA"):
            raise ValueError(
                "Falta GROQ_API_KEY. Crea un archivo .env en la raíz del proyecto "
                "con tu clave (ver .env.example)."
            )
        _llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.0,
            api_key=GROQ_API_KEY,
        )
        print(f"[RAG] LLM configurado: {GROQ_MODEL}")
    return _llm


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            print(f"[RAG] Cargando base vectorial existente: {CHROMA_DIR}")
            _vector_store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
        else:
            print("[RAG] No hay base vectorial. Ejecuta indexar_documentos() primero.")
            raise FileNotFoundError(
                f"No existe la carpeta chroma/ o está vacía. "
                f"Coloca PDFs en pdfs/ y ejecuta: python -m src.indexar"
            )
    return _vector_store


# ---------------------------------------------------------------------------
# Indexación (se ejecuta una vez o cuando cambian los PDFs)
# ---------------------------------------------------------------------------
def indexar_documentos(force: bool = False) -> int:
    """
    Carga todos los PDF de la carpeta pdfs/, los divide en fragmentos,
    genera embeddings y los guarda en ChromaDB.

    Returns:
        Número de fragmentos indexados.
    """
    global _vector_store

    if not PDFS_DIR.exists() or not any(PDFS_DIR.glob("*.pdf")):
        raise FileNotFoundError(
            f"No se encontraron PDFs en {PDFS_DIR}. "
            "Coloca tus archivos .pdf dentro de la carpeta pdfs/."
        )

    if force and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"[RAG] Base anterior eliminada: {CHROMA_DIR}")

    print(f"[RAG] Cargando PDFs desde {PDFS_DIR} ...")
    loader = PyPDFDirectoryLoader(str(PDFS_DIR))
    documents = loader.load()
    print(f"[RAG] Páginas cargadas: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"[RAG] Fragmentos generados: {len(chunks)}")

    embeddings = get_embeddings()
    print("[RAG] Indexando en ChromaDB (puede tardar unos minutos la primera vez)...")
    _vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    total = _vector_store._collection.count()
    print(f"[RAG] ✓ Indexados {total} fragmentos en {CHROMA_DIR}")
    return total


# ---------------------------------------------------------------------------
# Pipeline de consulta
# ---------------------------------------------------------------------------
def rag_pipeline(pregunta: str, k: int = DEFAULT_K, verbose: bool = False) -> dict[str, Any]:
    """
    Flujo completo: pregunta → recuperación → prompt aumentado → respuesta Groq.

    Returns:
        dict con keys: pregunta, fragmentos, contexto, respuesta, tokens_contexto_aprox
    """
    vector_store = get_vector_store()
    llm = get_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(pregunta)

    contexto = "\n\n---\n\n".join(
        f"[Fuente: {os.path.basename(d.metadata.get('source', '?'))} — "
        f"Pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    prompt = prompt_template.invoke({"context": contexto, "question": pregunta})
    respuesta = llm.invoke(prompt).content

    if verbose:
        print(f"Fragmentos recuperados: {len(docs)}")
        for i, d in enumerate(docs, 1):
            fuente = os.path.basename(d.metadata.get("source", "?"))
            pagina = d.metadata.get("page", "?")
            print(f"  [{i}] {fuente} Pág.{pagina}: {d.page_content[:80]}...")

    return {
        "pregunta": pregunta,
        "fragmentos": docs,
        "contexto": contexto,
        "respuesta": respuesta,
        "tokens_contexto_aprox": len(contexto) // 4,
    }
