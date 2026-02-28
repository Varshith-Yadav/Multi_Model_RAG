import os

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
URL = f"{OLLAMA_BASE_URL}/api/embeddings"
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def embed(text: str, model: str = DEFAULT_MODEL):
    payload = {
        "model": model,
        "prompt": text or "",
    }

    response = requests.post(URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    embedding = data.get("embedding")
    if not embedding and data.get("embeddings"):
        embedding = data["embeddings"][0]

    if not embedding:
        raise RuntimeError(f"Empty embedding returned by Ollama for model '{model}'.")

    return embedding

