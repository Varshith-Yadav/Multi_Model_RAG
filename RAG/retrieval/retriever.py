from pathlib import Path
import pickle

import faiss
import numpy as np

from RAG.embeddings.ollama_embed import embed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.bin"
META_PATH = INDEX_DIR / "meta.pkl"

_index = None
_docs = None


class IndexNotReadyError(RuntimeError):
    pass


def index_exists() -> bool:
    return INDEX_PATH.exists() and META_PATH.exists()


def load_index(force_reload: bool = False):
    global _index, _docs

    if not force_reload and _index is not None and _docs is not None:
        return _index, _docs

    if not index_exists():
        raise IndexNotReadyError(
            "Index files are missing. Run ingestion first to create "
            "vectorstore/faiss_index/index.bin and vectorstore/faiss_index/meta.pkl."
        )

    _index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("rb") as f:
        _docs = pickle.load(f)

    return _index, _docs


def retrieve(query: str, k: int = 5):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    index, docs = load_index()

    if index.ntotal == 0 or not docs:
        raise IndexNotReadyError("Index is empty. Ingest PDFs and rebuild the index.")

    top_k = max(1, min(int(k), index.ntotal))
    query_vector = np.array([embed(query)], dtype="float32")
    _, indices = index.search(query_vector, top_k)

    return [docs[i] for i in indices[0] if 0 <= i < len(docs)]
