from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Asegurar imports desde la raíz
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_K
from src.rag import get_llm, get_vector_store, rag_pipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    mensaje_usuario = (data.get("message") or "").strip()

    if not mensaje_usuario:
        return jsonify({"response": "Error: mensaje vacío"}), 400

    # Saludos simples (opcional, sin gastar tokens de Groq)
    msg_lower = mensaje_usuario.lower()
    if any(x in msg_lower for x in ("hola", "buenas", "buenos días", "buenas tardes")):
        return jsonify(
            {
                "response": (
                    "¡Hola! 👋 Soy el asistente del reglamento académico. "
                    "Pregúntame sobre inasistencias, matrícula, repetición de "
                    "asignaturas, requisitos de admisión, etc."
                )
            }
        )
    if any(x in msg_lower for x in ("adiós", "adios", "chao", "hasta luego")):
        return jsonify({"response": "¡Hasta luego! Si necesitas algo más, aquí estaré. 🙌"})

    try:
        resultado = rag_pipeline(mensaje_usuario, k=DEFAULT_K)
        return jsonify({"response": resultado["respuesta"]})
    except FileNotFoundError as e:
        return jsonify(
            {
                "response": (
                    f"⚠️ Base de conocimientos no lista.\n{e}\n\n"
                    "Coloca los PDFs en la carpeta pdfs/ y ejecuta:\n"
                    "  python -m src.indexar"
                )
            }
        ), 503
    except ValueError as e:
        return jsonify({"response": f"⚠️ Configuración incompleta:\n{e}"}), 503
    except Exception as e:
        return jsonify({"response": f"Error interno: {type(e).__name__}: {e}"}), 500


@app.route("/health")
def health():
    """Comprueba que embeddings, Chroma y LLM estén disponibles."""
    status = {"embeddings": False, "chroma": False, "llm": False, "ok": False}
    try:
        get_vector_store()
        status["chroma"] = True
        status["embeddings"] = True
    except Exception as e:
        status["error_chroma"] = str(e)

    try:
        get_llm()
        status["llm"] = True
    except Exception as e:
        status["error_llm"] = str(e)

    status["ok"] = status["chroma"] and status["llm"]
    code = 200 if status["ok"] else 503
    return jsonify(status), code


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
