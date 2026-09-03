# Enterprise Document Intelligence and RAG Assistant

Initial FastAPI foundation for an enterprise document intelligence and retrieval-augmented generation assistant. This version provides application configuration, logging, and a health endpoint only. Database, document processing, vector search, embeddings, and LLM capabilities are intentionally deferred.

## Prerequisites

- Python 3.11 or newer
- Git

PostgreSQL, Ollama, FAISS, spaCy, PyMuPDF, LangChain, Transformers, Docker, and MLflow will be added when their corresponding features are implemented.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the current dependencies:

```powershell
python -m pip install -r requirements.txt
```

Copy the example environment file and adjust values if needed:

```powershell
Copy-Item .env.example .env
```

## Run the application

```powershell
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/health` to check application health. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

The health response has this shape:

```json
{
  "status": "healthy",
  "name": "Enterprise Document Intelligence and RAG Assistant",
  "version": "0.1.0"
}
```

## Run tests

```powershell
python -m pytest
```
