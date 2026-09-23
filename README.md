# Asistente RAG Académico

Chatbot web con Flask que responde preguntas sobre reglamentos académicos usando **RAG** (Retrieval-Augmented Generation):

- PDFs → fragmentos → embeddings locales → ChromaDB → Groq LLM

## Requisitos

- Python 3.10 o superior
- Cuenta en [Groq](https://console.groq.com) (API key gratuita)

## Instalación (desde cero)

### 1. Crear y activar el entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar la API key de Groq

Crea un archivo `.env` en la raíz del proyecto:

```env
GROQ_API_KEY=gsk_tu_clave_aqui
GROQ_MODEL=openai/gpt-oss-120b
```

> Obtén tu clave en: https://console.groq.com/keys

### 4. Colocar los PDFs

Copia tus documentos PDF dentro de la carpeta:

```
pdfs/
```

### 5. Indexar la base de conocimientos (solo una vez)

```bash
python -m src.indexar
```

Si cambias los PDFs y quieres regenerar todo:

```bash
python -m src.indexar --force
```

### 6. Ejecutar la aplicación

```bash
python app.py
```

Abre en el navegador: **http://127.0.0.1:5000**

## Estructura del proyecto

```text
├── app.py               # Servidor Flask (chat)
├── requirements.txt
├── .env                # API key
├── pdfs/               # PDFs de conocimiento
├── chroma/             # Base vectorial (se genera al indexar)
├── src/
│   ├── config.py       # Rutas, modelo, chunk size
│   ├── rag.py          # Pipeline RAG completo
│   └── indexar.py      # Script de indexación
├── templates/
│   └── index.html      # Interfaz del chat
└── static/css/
    └── style.css
```
