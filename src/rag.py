from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
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
    HF_TOKEN,
    PDFS_DIR,
)

CONTEXTUALIZE_Q_PROMPT = """Dada una conversación y la última pregunta del usuario que podría hacer referencia al contexto del historial, formula una pregunta independiente que pueda ser entendida sin el historial de conversación. NO respondas la pregunta, solo reescríbela si es necesario; de lo contrario, devuélvela tal como está.

Historial de conversación:
{chat_history}

Última pregunta: {question}

Pregunta independiente:"""

contextualize_q_prompt = ChatPromptTemplate.from_template(CONTEXTUALIZE_Q_PROMPT)

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

_embeddings: HuggingFaceInferenceAPIEmbeddings | None = None
_vector_store: Chroma | None = None
_llm: ChatGroq | None = None


def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not HF_TOKEN:
            raise ValueError(
                "Falta HUGGINGFACE_TOKEN en las variables de entorno. "
                "Genera uno gratuito en https://huggingface.co/settings/tokens"
            )
        _embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
        )
    return _embeddings


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise ValueError("Falta GROQ_API_KEY en las variables de entorno.")
        _llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.0,
            api_key=GROQ_API_KEY,
        )
    return _llm


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            _vector_store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
        else:
            raise FileNotFoundError(
                f"No existe la carpeta chroma/ o está vacía."
            )
    return _vector_store


def indexar_documentos(force: bool = False) -> int:
    global _vector_store

    if not PDFS_DIR.exists() or not any(PDFS_DIR.glob("*.pdf")):
        raise FileNotFoundError(f"No se encontraron PDFs en {PDFS_DIR}.")

    if force and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    loader = PyPDFDirectoryLoader(str(PDFS_DIR))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    _vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return _vector_store._collection.count()


def _format_chat_history(history: list[dict[str, str]]) -> str:
    formatted = []
    for msg in history:
        role = "Usuario" if msg.get("role") == "user" else "Asistente"
        formatted.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(formatted)


def rag_pipeline(
    pregunta: str,
    historial: list[dict[str, str]] | None = None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    vector_store = get_vector_store()
    llm = get_llm()

    historial_formateado = _format_chat_history(historial or [])

    if historial_formateado.strip():
        prompt_contextualizer = contextualize_q_prompt.invoke(
            {"chat_history": historial_formateado, "question": pregunta}
        )
        pregunta_autonoma = llm.invoke(prompt_contextualizer).content.strip()
    else:
        pregunta_autonoma = pregunta

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(pregunta_autonoma)

    contexto = "\n\n---\n\n".join(
        f"[Fuente: {os.path.basename(d.metadata.get('source', '?'))} — "
        f"Pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    prompt = prompt_template.invoke({"context": contexto, "question": pregunta_autonoma})
    respuesta = llm.invoke(prompt).content

    return {
        "pregunta": pregunta,
        "pregunta_autonoma": pregunta_autonoma,
        "fragmentos": docs,
        "contexto": contexto,
        "respuesta": respuesta,
    }