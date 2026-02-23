import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from RAG.augmentation.prompt_builder import build_prompt
from RAG.generation.llm import generate
from RAG.retrieval.retriever import IndexNotReadyError, retrieve


def pretty_print_sources(docs):
    print("\nRetrieved Sources:\n")

    for i, doc in enumerate(docs, start=1):
        print(f"--- Source {i} ---")
        print(f"Type   : {doc.get('type')}")
        print(f"Page   : {doc.get('page')}")
        print(f"Source : {doc.get('source')}")

        content = doc.get("content", "")
        snippet = content[:300].replace("\n", " ")
        print(f"Snippet: {snippet}...\n")

        if doc.get("type") == "image":
            print(f"Image path: {doc.get('image_path')}\n")


def run_query(query: str, top_k: int = 5):
    print(f"\nQuery: {query}")

    docs = retrieve(query, k=top_k)
    pretty_print_sources(docs)

    prompt = build_prompt(query, docs)
    print("\nGenerating answer with Llama3...\n")

    answer = generate(prompt)

    print("\n================ ANSWER ================\n")
    print(answer)
    print("\n========================================\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts/query_demo.py \"your question here\" [top_k]")
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    try:
        run_query(query, top_k=top_k)
    except IndexNotReadyError as exc:
        print(f"\n[ERROR] {exc}")
        print("Run ingestion first: python scripts/ingest.py")
        sys.exit(1)
