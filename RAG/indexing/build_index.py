import os
import pickle
from pathlib import Path

import faiss
import numpy as np

from RAG.embeddings.ollama_embed import embed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index"


def build_index(docs, save_path=DEFAULT_SAVE_PATH):
    if not docs:
        raise ValueError("No documents found for indexing.")

    save_dir = Path(save_path)
    os.makedirs(save_dir, exist_ok=True)

    vectors = []
    indexed_docs = []

    for doc in docs:
        content = doc.get("content", "").strip()
        if not content:
            continue

        vector = embed(content)
        if not vector:
            continue

        vectors.append(vector)
        indexed_docs.append(doc)

    if not vectors:
        raise ValueError("No embeddings were generated. Check Ollama embedding setup.")

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))

    faiss.write_index(index, str(save_dir / "index.bin"))

    with (save_dir / "meta.pkl").open("wb") as f:
        pickle.dump(indexed_docs, f)

    return len(indexed_docs)
