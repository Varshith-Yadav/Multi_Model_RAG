import faiss
import numpy as np
import pickle
from RAG.embeddings.ollama_embed import embed

def build_index(docs, save_path="vectorstore/faiss_index"):
    vectors = []

    for d in docs:
        vectors.append(embed(d["content"]))

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, f"{save_path}/index.bin")

    with open(f"{save_path}/meta.pkl", "wb") as f:
        pickle.dump(docs, f)
