# Multi Model RAG

A local multimodal RAG system for PDFs with:
- text extraction
- table extraction
- image extraction + image captioning
- FAISS vector search
- LLM answer generation
- chat-style Streamlit UI

The project uses Ollama models locally:
- `nomic-embed-text` for embeddings
- `llama3` for answer generation
- `llava` for image understanding

## Features

- Chat UI (`streamlit_app.py`) with message history and source display
- PDF ingestion pipeline (`scripts/ingest.py`)
- CLI query demo (`scripts/query_demo.py`)
- FastAPI endpoints (`/health`, `/query`)
- Source-aware answers with page references in prompt context

## Project Structure

```text
rag/
  app/
    main.py
    routes/query.py
  RAG/
    augmentation/prompt_builder.py
    embeddings/ollama_embed.py
    generation/llm.py
    indexing/{pdf_loader,chunker,table_extractor,image_extractor,build_index}.py
    multimodel/{table_parser,image_captioner}.py
    retrieval/retriever.py
  scripts/
    ingest.py
    query_demo.py
  streamlit_app.py
  requirements.txt
```

## Prerequisites

1. Python 3.10+ (3.12 works).
2. Ollama installed and available in PATH.
3. Models pulled in Ollama:
   - `llama3`
   - `nomic-embed-text`
   - `llava`

For table extraction on Windows, install Ghostscript if Camelot requires it.

## Installation

From project root (`rag/`):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Ollama server:

```powershell
ollama serve
```

In another terminal:

```powershell
ollama pull llama3
ollama pull nomic-embed-text
ollama pull llava
```

## Data Preparation

Place your PDFs in:

```text
data/raw/
```

Example:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
```

## Build the Index (Ingestion)

```powershell
python scripts\ingest.py
```

This creates:
- `vectorstore/faiss_index/index.bin`
- `vectorstore/faiss_index/meta.pkl`
- extracted images in `data/images/` (if present in PDFs)

## Run the Chat UI (Recommended)

```powershell
streamlit run streamlit_app.py
```

In the sidebar:
1. Upload PDFs
2. Click `Save PDFs`
3. Click `Run Ingestion`
4. Start chatting in the input box at the bottom

## Run the API

```powershell
uvicorn app.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `GET /query?q=your_question&top_k=5`
- `POST /query` with JSON:

```json
{
  "q": "What is the revenue trend?",
  "top_k": 5
}
```

PowerShell example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/query" `
  -H "Content-Type: application/json" `
  -d "{\"q\":\"Summarize the document\",\"top_k\":5}"
```

## Run CLI Query Demo

```powershell
python scripts\query_demo.py "What is the profit margin?" 5
```

## How It Works

1. `scripts/ingest.py` loads PDFs from `data/raw`.
2. Text is chunked and stored as `type=text`.
3. Tables are extracted and converted to markdown (`type=table`).
4. Images are extracted and captioned with LLaVA (`type=image`).
5. All chunks are embedded and indexed in FAISS.
6. Query flow:
   - embed question
   - retrieve top-k chunks
   - build prompt with context
   - generate answer with `llama3`

## Troubleshooting

- `Index files are missing...`
  - Run: `python scripts\ingest.py`

- `No PDF files found in data/raw`
  - Add PDFs to `data/raw` and re-run ingestion.

- `Table extraction failed: camelot-py is not installed`
  - Install dependencies again: `python -m pip install -r requirements.txt`
  - On Windows, install Ghostscript if needed.

- Ollama connection/model errors
  - Ensure `ollama serve` is running.
  - Verify models with `ollama list`.

- Slow response
  - Local inference depends on hardware. CPU-only runs are slower.

## Notes

- The system is fully local (embedding + generation + image captioning via Ollama).
- You can change model names in:
  - `RAG/embeddings/ollama_embed.py`
  - `RAG/generation/llm.py`
  - `RAG/multimodel/image_captioner.py`
